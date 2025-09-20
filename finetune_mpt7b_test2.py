import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_ID = "mosaicml/mpt-7b-instruct"
ADAPTER_DIR = "out-mpt7b-biz-lora/adapter"

tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
# add the same special tokens used in training
num_added = tok.add_special_tokens({"additional_special_tokens": ["[SYS]", "[INST]", "[OUT]"]})

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=dtype, device_map="auto")
if num_added > 0:
    base.resize_token_embeddings(len(tok))

model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()

prompt = (
    "Extract policy number and effective date.\n"
    "Return exactly one line in the format: Policy=<ID>; Effective=<YYYY-MM-DD>\n\n"
    "Text:\nPolicy: ZX-55391\nEffective: 2024-08-01\n\nAnswer:"
)

inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,                    # deterministic
        no_repeat_ngram_size=6,
        repetition_penalty=1.1,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False,
    )

# Decode only the newly generated portion
gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
gen_text = tok.decode(gen_ids, skip_special_tokens=True)

# Keep just the first line after "Answer:"
first_line = gen_text.splitlines()[0].strip()
print(first_line)

