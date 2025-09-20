import json, torch, inspect
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "mosaicml/mpt-7b-instruct"
CTX_LEN = 4096

# --- Tokenizer ---
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

# --- Data ---
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

# --- Model ---
# --- Model ---
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,           # <-- fixed here
    attn_implementation="eager",
)
model.resize_token_embeddings(len(tok))   # ensure embeddings match tokenizer
model.config.use_cache = False
model.config.pad_token_id = tok.pad_token_id

# --- LoRA ---
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["attn.Wqkv", "attn.out_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# --- Custom collator: force labels === input_ids (CLM) ---
def clm_collator(batch):
    # batch is list of dicts with input_ids (+ attention_mask)
    input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
    attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]
    # left-pad is tricky for MPT; we used right padding, so pad on right:
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tok.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# --- Trainer args (version-safe) ---
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
    gradient_checkpointing=False,     # <-- turn OFF to avoid the error
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

args = TrainingArguments(**base_kwargs)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=clm_collator,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tok,  # fine for your version (HF warns it’ll be removed in v5)
)

trainer.train()

model.save_pretrained("mpt7b-biz-lora/adapter")
tok.save_pretrained("mpt7b-biz-lora/tokenizer")
print("Saved LoRA adapter to mpt7b-biz-lora/adapter")

