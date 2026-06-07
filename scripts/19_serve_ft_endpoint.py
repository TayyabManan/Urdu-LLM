"""Serve fine-tuned Qwen 2.5 7B + v2 LoRA adapter as a Modal HTTP endpoint.

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
        gpu="H100",
    )
    .add_local_python_source("src", copy=True)
)

app = modal.App("urdu-rag-ft-endpoint")
vol = modal.Volume.from_name("urdu-llm-vol", create_if_missing=False)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "/vol/7b-v2-adapter"
MAX_SEQ_LENGTH = 4096

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 512
DEFAULT_REPETITION_PENALTY = 1.1
SYSTEM_PROMPT = "You are a helpful assistant."

EMBEDDER_MODEL = "BAAI/bge-m3"
QDRANT_VOL_PATH = "/vol/qdrant_data_v1"
COLLECTION = "urdu_wikipedia_v1"
EMBED_DIM = 1024
DEFAULT_TOP_K = 3

RAG_TEMPLATE = """{chunks}

سوال: {query}"""

RAG_SYSTEM = (
    "You are a helpful assistant. Use the provided context to answer in Urdu. "
    "If the context is insufficient, answer briefly from general knowledge."
)


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
        print("[load] ready")

    def _generate(
        self, prompt: str, system: str | None,
        max_tokens: int, temperature: float, top_p: float,
        repetition_penalty: float,
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
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                repetition_penalty=repetition_penalty,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {
            "text": text,
            "tokens_in": int(inputs["input_ids"].shape[1]),
            "tokens_out": int(new_tokens.shape[0]),
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
        out = []
        for h in res.points:
            payload = h.payload or {}
            meta = payload.get("meta") or {}
            out.append({
                "score": float(h.score),
                "title": meta.get("title") or payload.get("title"),
                "url": meta.get("url") or payload.get("url"),
                "article_id": meta.get("article_id") or payload.get("article_id"),
                "chunk_idx": meta.get("chunk_idx") or payload.get("chunk_idx"),
                "content": payload.get("content"),
            })
        return out

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
        """POST /rag {query, top_k?, max_tokens?, temperature?, top_p?, repetition_penalty?} → {answer, sources, tokens_in, tokens_out}."""
        from fastapi import HTTPException

        query = request.get("query")
        if not query or not isinstance(query, str):
            raise HTTPException(status_code=400, detail="missing or invalid 'query'")

        top_k = int(request.get("top_k", DEFAULT_TOP_K))
        try:
            sources = self._retrieve(query, top_k=top_k)
            chunks_text = "\n\n".join(s["content"] for s in sources if s.get("content"))
            rag_prompt = RAG_TEMPLATE.format(chunks=chunks_text, query=query)
            gen = self._generate(
                prompt=rag_prompt,
                system=RAG_SYSTEM,
                max_tokens=int(request.get("max_tokens", DEFAULT_MAX_TOKENS)),
                temperature=float(request.get("temperature", 0.3)),
                top_p=float(request.get("top_p", DEFAULT_TOP_P)),
                repetition_penalty=float(request.get("repetition_penalty", 1.3)),
            )
            return {
                "answer": gen["text"],
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
            "qdrant": QDRANT_VOL_PATH,
            "collection": COLLECTION,
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

    @modal.fastapi_endpoint(method="POST")
    def reindex(self, request: dict):
        """Force HNSW index build by lowering indexing_threshold."""
        from qdrant_client.models import OptimizersConfigDiff
        self.qdrant.update_collection(
            collection_name=self.collection,
            optimizer_config=OptimizersConfigDiff(indexing_threshold=1),
        )
        return {"status": "reindex triggered"}
