import json, torch, inspect
from dataclasses import dataclass
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "mosaicml/mpt-7b-instruct"
CTX_LEN = 4096

# --- 1) Tokenizer ---
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

# --- 2) Load data ---
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f.read().splitlines()]

def format_row(r):
    sys = "You are a helpful business assistant."
    inst = r.get("instruction","").strip()
    inp = r.get("input","").strip()
    out = r.get("output","").strip()
    if inp:
        prompt = f"<s>[SYS]{sys}[/SYS]\n[INST]{inst}\n\n{inp}[/INST]\n[OUT]"
    else:
        prompt = f"<s>[SYS]{sys}[/SYS]\n[INST]{inst}[/INST]\n[OUT]"
    return prompt, out

def tokenize(example):
    prompt, out = format_row(example)
    text = prompt + out + tok.eos_token
    return tok(text, truncation=True, max_length=CTX_LEN)

train_data = read_jsonl("train.jsonl")
eval_data  = train_data[:200]

train_ds = Dataset.from_list(train_data).map(tokenize, remove_columns=list(train_data[0].keys()))
eval_ds  = Dataset.from_list(eval_data ).map(tokenize, remove_columns=list(eval_data[0].keys()))

# --- 3) Base model ---
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,                     # <- use 'dtype' (newer HF)
    attn_implementation="eager",     # MPT doesn’t support FA2/SDPA in your setup
)
# If using gradient checkpointing, disable cache
model.config.use_cache = False
model.config.pad_token_id = tok.pad_token_id

# --- 4) LoRA config ---
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["attn.Wqkv", "attn.out_proj"],  # MPT attention names
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# --- 5) Trainer args (version-safe) ---
base_kwargs = dict(
    output_dir="mpt7b-biz-lora",
    num_train_epochs=2,
    learning_rate=2e-4,
    weight_decay=0.0,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    logging_steps=10,
    save_steps=200,
    bf16=(dtype == torch.bfloat16),
    fp16=(dtype == torch.float16),
    gradient_checkpointing=True,
    optim="adamw_torch",
    report_to="none",
)

sig = inspect.signature(TrainingArguments.__init__).parameters
if "evaluation_strategy" in sig:
    base_kwargs["evaluation_strategy"] = "steps"
    base_kwargs["eval_steps"] = 200
elif "eval_strategy" in sig:
    base_kwargs["eval_strategy"] = "steps"
    base_kwargs["eval_steps"] = 200
else:
    # No periodic eval supported in this transformers version
    pass

args = TrainingArguments(**base_kwargs)

data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

# --- 6) Train ---
trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tok,
)
trainer.train()

# --- 7) Save adapter ---
model.save_pretrained("mpt7b-biz-lora/adapter")
tok.save_pretrained("mpt7b-biz-lora/tokenizer")
print("Saved LoRA adapter to mpt7b-biz-lora/adapter")

