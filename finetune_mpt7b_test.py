import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_ID = "mosaicml/mpt-7b-instruct"
ADAPTER_DIR = "out-mpt7b-biz-lora/adapter"

# 1) Load tokenizer and re-add the same special tokens used in training
tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
num_added = tok.add_special_tokens({"additional_special_tokens": ["[SYS]", "[INST]", "[OUT]"]})

# 2) Load base and resize embeddings to match tokenizer
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=dtype,
    device_map="auto",
)
if num_added > 0:
    base.resize_token_embeddings(len(tok))

# 3) Attach LoRA adapter (embeddings will now match)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()

# 4) Simple generate
prompt = (
    "Extract policy number and effective date:\n"
    "Policy: ZX-55391\nEffective: 2024-08-01\nAnswer:"
)
inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        temperature=0.2,
        pad_token_id=tok.eos_token_id,
    )

print(tok.decode(out[0], skip_special_tokens=True))

