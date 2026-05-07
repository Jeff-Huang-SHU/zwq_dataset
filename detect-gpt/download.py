from huggingface_hub import snapshot_download

# Minimal downloads to run DetectGPT locally.
# - Prefer safetensors when available.
# - Skip TF/Flax/Rust artifacts to reduce disk usage.
cache_dir = "/home/hjf/.cache"  # keep consistent with run_zwq.py --cache_dir

COMMON_TOKENIZER_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

GPT2_FILES = COMMON_TOKENIZER_FILES + [
    "vocab.json",
    "merges.txt",
    "model.safetensors",
    "pytorch_model.bin",
]

T5_FILES = COMMON_TOKENIZER_FILES + [
    "spiece.model",
    "model.safetensors",
    "pytorch_model.bin",
]

SKIP_PATTERNS = [
    "*.h5",          # TensorFlow
    "*.msgpack",     # Flax
    "*.ot",          # Rust
    "*.ckpt",
]


def dl(repo_id: str, allow_patterns: list[str]) -> None:
    print(f"downloading {repo_id} ...")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=SKIP_PATTERNS,
    )


if __name__ == "__main__":
    dl("gpt2-xl", GPT2_FILES)
    dl("t5-large", T5_FILES)
    dl("t5-small", T5_FILES)
    print("all done")