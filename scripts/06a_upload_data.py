"""
Upload formatted training data to Modal Volume.
Run this once before training.

    modal run scripts/06a_upload_data.py
"""

import modal

app = modal.App("upload-data")
vol = modal.Volume.from_name("urdu-llm-vol", create_if_missing=True)


@app.function(volumes={"/vol": vol})
def upload(data: bytes):
    """Receive data and write to volume."""
    import os
    os.makedirs("/vol/data", exist_ok=True)
    with open("/vol/data/train.jsonl", "wb") as f:
        f.write(data)
    # Verify
    lines = sum(1 for _ in open("/vol/data/train.jsonl", encoding="utf-8"))
    print(f"Written {lines:,} lines to /vol/data/train.jsonl")
    vol.commit()


@app.local_entrypoint()
def main():
    print("Reading local data...")
    with open("data/formatted/train.jsonl", "rb") as f:
        data = f.read()
    print(f"Uploading {len(data) / 1024**2:.1f} MB to Modal Volume...")
    upload.remote(data)
    print("Done! Data is on Modal Volume at /vol/data/train.jsonl")
