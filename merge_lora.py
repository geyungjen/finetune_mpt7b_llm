# merge_lora.py
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "mosaicml/mpt-7b-instruct"
ADAPTER_DIR = "mpt7b-biz-lora/adapter"     # <- from your trainer save path
OUT_DIR = "mpt7b-biz-merged"

torch.set_default_dtype(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

print("Loading base…")
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    attn_implementation="eager",
)

# make sure pad/eos are aligned (MPT uses 0 for eos/bos/pad usually)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base.resize_token_embeddings(len(tok))
base.config.pad_token_id = tok.pad_token_id
base.config.use_cache = False

print("Loading LoRA adapter…")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
print("Merging LoRA into base…")
model = model.merge_and_unload()  # writes LoRA deltas into base weights

print(f"Saving merged model to {OUT_DIR} …")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, safe_serialization=True)  # safetensors
tok.save_pretrained(OUT_DIR)
print("Done.")

