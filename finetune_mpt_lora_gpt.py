import json, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "mosaicml/mpt-7b-instruct"
CTX_LEN = 4096

# --- 1) Tokenizer ---
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"  # safer for causal LM training

# --- 2) Load data ---
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f.read().splitlines()]

def format_row(r):
    # Simple SFT-style prompt template
    sys = "You are a helpful business assistant."
    inst = r.get("instruction", "").strip()
    inp = r.get("input", "").strip()
    out = r.get("output", "").strip()
    if inp:
        prompt = f"<s>[SYS]{sys}[/SYS]\n[INST]{inst}\n\n{inp}[/INST]\n[OUT]"
    else:
        prompt = f"<s>[SYS]{sys}[/SYS]\n[INST]{inst}[/INST]\n[OUT]"
    return prompt, out

def tokenize(example):
    prompt, out = format_row(example)
    text = prompt + out + tok.eos_token
    # Return input_ids + attention_mask; labels added by collator
    return tok(text, truncation=True, max_length=CTX_LEN)

train_data = read_jsonl("train.jsonl")
eval_data  = read_jsonl("eval.jsonl") if False else train_data[:200]  # quick sanity

train_ds = Dataset.from_list(train_data).map(tokenize, remove_columns=list(train_data[0].keys()))
eval_ds  = Dataset.from_list(eval_data).map(tokenize,  remove_columns=list(eval_data[0].keys()))

# --- 3) Base model ---
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,                    # new arg name (torch_dtype is deprecated)
    attn_implementation="eager",    # MPT requires eager (no FA2/SDPA yet)
)
model.config.use_cache = False       # required with gradient checkpointing

# --- 4) LoRA config (target attention proj matrices) ---
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["attn.Wqkv", "attn.out_proj"],  # MPT attention proj names
)
model = get_peft_model(model, lora_cfg)

# Sanity check: must be > 0 trainable params
model.print_trainable_parameters()

# --- 5) Trainer args (tune batch/accum to your VRAM) ---
args = TrainingArguments(
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
    evaluation_strategy="steps",   # <-- FIXED
    eval_steps=200,
    save_steps=200,
    bf16=(dtype==torch.bfloat16),
    fp16=(dtype==torch.float16),
    gradient_checkpointing=True,
    optim="adamw_torch",
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

# --- 6) Train ---
trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)
trainer.train()

# --- 7) Save adapter ---
model.save_pretrained("mpt7b-biz-lora/adapter")
tok.save_pretrained("mpt7b-biz-lora/tokenizer")
print("Saved LoRA adapter to mpt7b-biz-lora/adapter")

