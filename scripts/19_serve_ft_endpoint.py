"""Serve fine-tuned Qwen 2.5 7B + v3 LoRA adapter as a Modal HTTP endpoint.

Two endpoints in one container:
  POST /generate  — raw FT generation (used by judge eval, prompt-only)
  POST /rag       — retrieval (bge-m3 + Qdrant on /vol) + grounded FT generation

Both share the same warm container (FT model loaded once). RAG endpoint also
holds bge-m3 + Qdrant client. All RAM lives on Modal H100, not local WSL.

Deploy:
    modal deploy scripts/19_serve_ft_endpoint.py

Smoke (after deploy):
    curl -X POST <GENERATE_URL> -H "Content-Type: application/json" \
        -d '{"prompt": "Pakistan ka qaumi tarana kis ne likha?"}'

    curl -X POST <RAG_URL> -H "Content-Type: application/json" \
        -d '{"query": "Pakistan ka qaumi tarana kis ne likha?", "top_k": 3}'
"""
from __future__ import annotations

import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "unsloth", "xformers")
    .pip_install(
        "trl>=0.15",
        "peft",
        "accelerate",
        "bitsandbytes",
        "transformers>=4.46",
        "fastapi[standard]",
        "pydantic>=2",
        "haystack-ai>=2.29.0",
        "qdrant-haystack",
        "qdrant-client",
        "sentence-transformers>=3.0",
        "rank-bm25>=0.2.2",
        gpu="H100",
    )
    .add_local_python_source("src", copy=True)
)

# Shipped default is v3. Both the app name and the adapter dir are env-configurable so a
# different version can be deployed ALONGSIDE the live one without touching it — these
# module globals are read at deploy time and baked into the served function by Modal.
# Deploy the live v3 endpoint (what the Space points at):
#   modal deploy scripts/19_serve_ft_endpoint.py
# Roll back to v2, or stand a v2 endpoint up side-by-side, with:
#   RAG_ADAPTER_DIR=/vol/7b-v2-adapter RAG_APP_NAME=urdu-rag-ft-endpoint-v2 \
#       modal deploy scripts/19_serve_ft_endpoint.py
app = modal.App(os.environ.get("RAG_APP_NAME", "urdu-rag-ft-endpoint"))
vol = modal.Volume.from_name("urdu-llm-vol", create_if_missing=False)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = os.environ.get("RAG_ADAPTER_DIR", "/vol/7b-v3-adapter")
MAX_SEQ_LENGTH = 4096

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 512
DEFAULT_REPETITION_PENALTY = 1.1
SYSTEM_PROMPT = "You are a helpful assistant."

EMBEDDER_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
QDRANT_VOL_PATH = "/vol/qdrant_data_v1"
COLLECTION = "urdu_wikipedia_v1"
BM25_PATH = "/vol/bm25_index_v1.pkl"
EMBED_DIM = 1024
DEFAULT_TOP_K = 3
DEFAULT_DENSE_K = 50
DEFAULT_BM25_K = 50
DEFAULT_RRF_K = 60

RAG_TEMPLATE = """{chunks}

سوال: {query}"""

# Iter 2 (RAG_SYSTEM = SYSTEM_PROMPT) tested 20% smoke vs Iter 1's 35%.
# "Use the provided context" instruction empirically helps despite being
# out-of-training-distribution. Reverted.
RAG_SYSTEM = (
    "You are a helpful assistant. Use the provided context to answer in Urdu. "
    "If the context is insufficient, answer briefly from general knowledge."
)

URDU_PUNCT = "۔،؟!؛٪٫٬"
ASCII_PUNCT = ".,!?;:()[]{}\"'-·–—"
_PUNCT_TABLE = str.maketrans({c: " " for c in URDU_PUNCT + ASCII_PUNCT})


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in text.translate(_PUNCT_TABLE).split() if t]


@app.cls(
    gpu="H100",
    image=image,
    volumes={"/vol": vol},
    scaledown_window=300,
    timeout=600,
    min_containers=0,
)
@modal.concurrent(max_inputs=4)
class FTServer:
    @modal.enter()
    def load(self):
        from unsloth import FastLanguageModel
        from peft import PeftModel
        import torch

        print(f"[load] base={MODEL_NAME} adapter={ADAPTER_DIR}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        FastLanguageModel.for_inference(model)
        free, total = torch.cuda.mem_get_info(0)
        print(f"[load] FT VRAM: {(total - free) / 1024**3:.2f} GB / {total / 1024**3:.1f} GB")
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch

        # RAG side: embedder + Qdrant
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient

        print(f"[load] embedder={EMBEDDER_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDER_MODEL, device="cuda")
        free, total = torch.cuda.mem_get_info(0)
        print(f"[load] FT+embedder VRAM: {(total - free) / 1024**3:.2f} GB / {total / 1024**3:.1f} GB")

        print(f"[load] qdrant path={QDRANT_VOL_PATH}")
        self.qdrant = QdrantClient(path=QDRANT_VOL_PATH)
        self.collection = COLLECTION

        # BM25 sidecar (built by scripts/20c_build_bm25.py)
        import os
        import pickle
        if os.path.exists(BM25_PATH):
            print(f"[load] bm25 from {BM25_PATH}")
            with open(BM25_PATH, "rb") as f:
                bundle = pickle.load(f)
            self.bm25 = bundle["bm25"]
            self.bm25_chunk_ids = bundle["chunk_ids"]
            print(f"[load] bm25 corpus_size={self.bm25.corpus_size}")
        else:
            self.bm25 = None
            self.bm25_chunk_ids = None
            print(f"[load] WARN: {BM25_PATH} missing; hybrid retrieval unavailable")

        # Cross-encoder reranker
        try:
            from sentence_transformers import CrossEncoder
            print(f"[load] reranker={RERANKER_MODEL}")
            self.reranker = CrossEncoder(RERANKER_MODEL, device="cuda", max_length=512)
            free, total = torch.cuda.mem_get_info(0)
            print(f"[load] FT+embedder+reranker VRAM: {(total - free) / 1024**3:.2f} GB / {total / 1024**3:.1f} GB")
        except Exception as e:
            self.reranker = None
            print(f"[load] WARN: reranker load failed: {e!r}")

        print("[load] ready")

    def _generate(
        self, prompt: str, system: str | None,
        max_tokens: int, temperature: float, top_p: float,
        repetition_penalty: float,
        use_adapter: bool = True,
    ) -> dict:
        messages = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            input_text, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LENGTH - max_tokens,
        ).to(self.model.device)

        def _do_generate():
            with self.torch.inference_mode():
                return self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    repetition_penalty=repetition_penalty,
                )

        if use_adapter:
            outputs = _do_generate()
        else:
            # PeftModel.disable_adapter() temporarily bypasses LoRA -> base weights only.
            with self.model.disable_adapter():
                outputs = _do_generate()

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {
            "text": text,
            "tokens_in": int(inputs["input_ids"].shape[1]),
            "tokens_out": int(new_tokens.shape[0]),
            "used_adapter": use_adapter,
        }

    def _retrieve(self, query: str, top_k: int) -> list[dict]:
        from qdrant_client.models import SearchParams
        vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
        # Collection uses default unnamed vector (debug confirmed). Force exact
        # brute-force since indexed_vectors_count=0 — local-mode HNSW didn't build.
        res = self.qdrant.query_points(
            collection_name=self.collection,
            query=vec,
            limit=top_k,
            with_payload=True,
            search_params=SearchParams(exact=True),
        )
        return [self._point_to_dict(h) for h in res.points]

    @staticmethod
    def _point_to_dict(point, score: float | None = None) -> dict:
        payload = point.payload or {}
        meta = payload.get("meta") or {}
        return {
            "score": float(score if score is not None else getattr(point, "score", 0.0)),
            "title": meta.get("title") or payload.get("title"),
            "url": meta.get("url") or payload.get("url"),
            "article_id": meta.get("article_id") or payload.get("article_id"),
            "chunk_idx": meta.get("chunk_idx") or payload.get("chunk_idx"),
            "content": payload.get("content"),
        }

    def _retrieve_hybrid(self, query: str, top_k: int,
                         dense_k: int = DEFAULT_DENSE_K,
                         bm25_k: int = DEFAULT_BM25_K,
                         rrf_k: int = DEFAULT_RRF_K) -> list[dict]:
        """Dense top-N + BM25 top-N → RRF dedupe → cross-encode → top_k."""
        from qdrant_client.models import SearchParams
        import numpy as np

        if self.bm25 is None:
            return self._retrieve(query, top_k)

        # Dense
        vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
        dense_res = self.qdrant.query_points(
            collection_name=self.collection,
            query=vec,
            limit=dense_k,
            with_payload=False,
            search_params=SearchParams(exact=True),
        )
        dense_ids = [str(p.id) for p in dense_res.points]
        dense_scores = {str(p.id): float(p.score) for p in dense_res.points}

        # BM25
        q_tokens = _tokenize(query)
        if not q_tokens:
            return self._retrieve(query, top_k)
        bm25_scores_full = self.bm25.get_scores(q_tokens)
        bm25_top_idx = np.argsort(-bm25_scores_full)[:bm25_k]
        bm25_ids = [self.bm25_chunk_ids[i] for i in bm25_top_idx if bm25_scores_full[i] > 0]

        # RRF fuse
        rrf: dict[str, float] = {}
        for rank, _id in enumerate(dense_ids):
            rrf[_id] = rrf.get(_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, _id in enumerate(bm25_ids):
            rrf[_id] = rrf.get(_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        fused_ids = sorted(rrf, key=lambda k: -rrf[k])[:max(dense_k, bm25_k)]

        # Hydrate payloads (single batch retrieve covers BM25-only ids)
        records = self.qdrant.retrieve(
            collection_name=self.collection,
            ids=fused_ids,
            with_payload=True,
            with_vectors=False,
        )
        rec_by_id = {str(r.id): r for r in records}
        candidates = []
        for _id in fused_ids:
            r = rec_by_id.get(_id)
            if r is None:
                continue
            payload = r.payload or {}
            content = payload.get("content") or ""
            if not content.strip():
                continue
            candidates.append((_id, r, content))

        if not candidates:
            return []

        # Cross-encode rerank
        if self.reranker is not None:
            pairs = [(query, c) for _, _, c in candidates]
            ce_scores = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            ordered = sorted(zip(candidates, ce_scores), key=lambda x: -float(x[1]))
            return [
                self._point_to_dict(r, score=float(s))
                for (_id, r, _c), s in ordered[:top_k]
            ]

        # Fallback: rank by RRF only
        return [self._point_to_dict(r, score=rrf[_id]) for _id, r, _c in candidates[:top_k]]

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, request: dict):
        """POST /generate {prompt, system?, max_tokens?, temperature?, top_p?, repetition_penalty?} → {text, tokens_in, tokens_out}."""
        from fastapi import HTTPException

        prompt = request.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="missing or invalid 'prompt'")

        try:
            return self._generate(
                prompt=prompt,
                system=request.get("system"),
                max_tokens=int(request.get("max_tokens", DEFAULT_MAX_TOKENS)),
                temperature=float(request.get("temperature", DEFAULT_TEMPERATURE)),
                top_p=float(request.get("top_p", DEFAULT_TOP_P)),
                repetition_penalty=float(request.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"generation failed: {e!r}")

    @modal.fastapi_endpoint(method="POST", docs=True)
    def rag(self, request: dict):
        """POST /rag {query, top_k?, mode?='hybrid'|'dense_only', max_tokens?, temperature?, top_p?, repetition_penalty?} → {answer, sources, mode, tokens_in, tokens_out}."""
        from fastapi import HTTPException

        query = request.get("query")
        if not query or not isinstance(query, str):
            raise HTTPException(status_code=400, detail="missing or invalid 'query'")

        top_k = int(request.get("top_k", DEFAULT_TOP_K))
        mode = str(request.get("mode", "hybrid")).lower()
        use_adapter = bool(request.get("use_adapter", True))
        try:
            if mode == "dense_only" or self.bm25 is None:
                effective_mode = "dense_only"
                sources = self._retrieve(query, top_k=top_k)
            else:
                effective_mode = "hybrid_reranked" if self.reranker is not None else "hybrid_rrf"
                sources = self._retrieve_hybrid(
                    query,
                    top_k=top_k,
                    dense_k=int(request.get("dense_k", DEFAULT_DENSE_K)),
                    bm25_k=int(request.get("bm25_k", DEFAULT_BM25_K)),
                    rrf_k=int(request.get("rrf_k", DEFAULT_RRF_K)),
                )
            if not use_adapter:
                effective_mode = f"{effective_mode}+base"
            chunks_text = "\n\n".join(s["content"] for s in sources if s.get("content"))
            rag_prompt = RAG_TEMPLATE.format(chunks=chunks_text, query=query)
            gen = self._generate(
                prompt=rag_prompt,
                system=RAG_SYSTEM,
                max_tokens=int(request.get("max_tokens", DEFAULT_MAX_TOKENS)),
                temperature=float(request.get("temperature", 0.3)),
                top_p=float(request.get("top_p", DEFAULT_TOP_P)),
                repetition_penalty=float(request.get("repetition_penalty", 1.3)),
                use_adapter=use_adapter,
            )
            return {
                "answer": gen["text"],
                "mode": effective_mode,
                "used_adapter": gen.get("used_adapter", True),
                "sources": [
                    {k: v for k, v in s.items() if k != "content"} | {"snippet": (s.get("content") or "")[:400]}
                    for s in sources
                ],
                "tokens_in": gen["tokens_in"],
                "tokens_out": gen["tokens_out"],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"rag failed: {e!r}")

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "adapter": ADAPTER_DIR,
            "embedder": EMBEDDER_MODEL,
            "reranker": RERANKER_MODEL if self.reranker is not None else None,
            "qdrant": QDRANT_VOL_PATH,
            "collection": COLLECTION,
            "bm25_loaded": self.bm25 is not None,
            "bm25_corpus_size": getattr(self.bm25, "corpus_size", None),
        }

    @modal.fastapi_endpoint(method="GET")
    def debug_collection(self):
        info = self.qdrant.get_collection(self.collection)
        sample = self.qdrant.scroll(self.collection, limit=1, with_payload=True, with_vectors=True)
        pt = sample[0][0] if sample[0] else None
        vec = getattr(pt, "vector", None) if pt else None
        vec_info: dict = {"type": str(type(vec).__name__)}
        if isinstance(vec, list):
            vec_info["len"] = len(vec)
            vec_info["first_5"] = vec[:5]
            vec_info["is_nonzero"] = any(abs(v) > 1e-9 for v in vec)
        elif isinstance(vec, dict):
            vec_info["names"] = list(vec.keys())
            for k, v in vec.items():
                if isinstance(v, list):
                    vec_info[f"{k}_len"] = len(v)
                    vec_info[f"{k}_first_5"] = v[:5]
        return {
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "vector_info": vec_info,
            "sample_payload_keys": list((pt.payload or {}).keys()) if pt else [],
        }

    @modal.fastapi_endpoint(method="GET")
    def selfsim(self, n: int = 5, strategy: str = "stratified"):
        """Probe stored-vec vs freshly-encoded-vec for n chunks.

        <0.99 cosine on either normalize variant => wrapper mismatch =>
        reindex with the matching encoder path. Also issues a top-1 query
        with the fresh vector to confirm end-to-end retrieval recovers self.

        Query params:
          n          : number of chunks to probe (default 5, max 50)
          strategy   : random | first | stratified (default stratified)
        """
        import math
        import random
        from qdrant_client.models import SearchParams

        n = max(1, min(int(n), 50))
        pool, _ = self.qdrant.scroll(
            self.collection,
            limit=n * 50,
            with_payload=True,
            with_vectors=True,
        )
        if not pool:
            return {"error": "collection empty", "n": 0, "results": []}

        if strategy == "first":
            picks = pool[:n]
        elif strategy == "random":
            picks = random.sample(pool, k=min(n, len(pool)))
        else:  # stratified
            stride = max(1, len(pool) // n)
            picks = [pool[i] for i in range(0, len(pool), stride)][:n]

        def _cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return dot / (na * nb)

        results = []
        for pt in picks:
            payload = pt.payload or {}
            content = payload.get("content") or ""
            meta = payload.get("meta") or {}
            stored = pt.vector if isinstance(pt.vector, list) else None
            if not content or stored is None:
                results.append({"point_id": str(pt.id), "skipped": True})
                continue

            fresh_norm = self.embedder.encode(
                content, normalize_embeddings=True
            ).tolist()
            fresh_unnorm = self.embedder.encode(
                content, normalize_embeddings=False
            ).tolist()

            cos_norm = _cos(stored, fresh_norm)
            cos_unnorm = _cos(stored, fresh_unnorm)
            stored_norm = math.sqrt(sum(x * x for x in stored))

            top = self.qdrant.query_points(
                collection_name=self.collection,
                query=fresh_norm,
                limit=1,
                with_payload=False,
                search_params=SearchParams(exact=True),
            )
            top1 = top.points[0] if top.points else None
            same_as_self = bool(top1 and str(top1.id) == str(pt.id))

            results.append({
                "point_id": str(pt.id),
                "title": meta.get("title") or payload.get("title"),
                "article_id": meta.get("article_id") or payload.get("article_id"),
                "content_preview": content[:120],
                "stored_vec_norm": stored_norm,
                "fresh_norm_true_cos": cos_norm,
                "fresh_norm_false_cos": cos_unnorm,
                "top1_score": float(top1.score) if top1 else None,
                "top1_id": str(top1.id) if top1 else None,
                "same_as_self": same_as_self,
            })

        valid = [r for r in results if "fresh_norm_true_cos" in r]
        if not valid:
            return {"n": 0, "results": results,
                    "verdict": {"vectors_match": False, "min_cos": 0.0, "threshold": 0.99}}
        min_cos = min(
            max(r["fresh_norm_true_cos"], r["fresh_norm_false_cos"]) for r in valid
        )
        all_self = all(r["same_as_self"] for r in valid)
        return {
            "n": len(valid),
            "strategy": strategy,
            "results": results,
            "verdict": {
                "vectors_match": min_cos >= 0.99 and all_self,
                "min_cos": min_cos,
                "all_same_as_self": all_self,
                "threshold": 0.99,
            },
        }

    @modal.fastapi_endpoint(method="POST")
    def reindex(self, request: dict):
        """Force HNSW index build by lowering indexing_threshold."""
        from qdrant_client.models import OptimizersConfigDiff
        self.qdrant.update_collection(
            collection_name=self.collection,
            optimizer_config=OptimizersConfigDiff(indexing_threshold=1),
        )
        return {"status": "reindex triggered"}
