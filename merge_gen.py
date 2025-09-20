python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_ID   = "mosaicml/mpt-7b-instruct"
ADAPTER   = "out-mpt7b-biz-lora/adapter"
OUT_DIR   = "mpt7b-biz-merged"

# tokenizer: add the same special tokens used during training
tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.add_special_tokens({"additional_special_tokens": ["[SYS]","[INST]","[OUT]"]})

# base model + resize to match tokenizer
base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype="auto")
base.resize_token_embeddings(len(tok))

# attach adapter, then merge it into base weights
merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()

# save tokenizer + merged model to one HF folder
tok.save_pretrained(OUT_DIR)
merged.save_pretrained(OUT_DIR)

print(f"Saved merged model to: {OUT_DIR}")
PY

