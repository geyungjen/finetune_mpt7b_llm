# quick_generate.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "mpt7b-biz-merged"

tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    attn_implementation="eager",
    device_map="auto",
)
model.eval()

prompt = "<s>[SYS]You are a helpful business assistant.[/SYS]\n[INST]Explain working capital in one sentence[/INST]\n[OUT]"
inp = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **inp,
        max_new_tokens=128,
        temperature=0.2,
        do_sample=True,
        eos_token_id=tok.eos_token_id,
    )
print(tok.decode(out[0], skip_special_tokens=True))

