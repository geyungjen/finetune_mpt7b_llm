import json, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

MODEL = "mosaicml/mpt-7b-instruct"
CTX_LEN = 2048  # mpt-7b-instruct is trained for 2048

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def build_prompt(ex):
    sys = "You are a concise, neutral business assistant."
    inst = ex.get("instruction","").strip()
    inp  = ex.get("input","").strip()
    out  = ex.get("output","").strip()
    if inp:
        prompt = f"<s>[SYS]{sys}[/SYS]\n[INST]{inst}\n\n{inp}[/INST]\n[OUT]{out}</s>"
    else:
        prompt = f"<s>[SYS]{sys}[/SYS]\n[INST]{inst}[/INST]\n[OUT]{out}</s>"
    return {"text": prompt}

# ---- Data ----
train = Dataset.from_list([build_prompt(x) for x in read_jsonl("train.jsonl")])
eval_  = train.select(range(min(200, len(train))))  # tiny sanity set

# ---- Tokenizer ----
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# (1) Register special tokens and (2) resize embeddings
extra_tokens = {"additional_special_tokens": ["[SYS]", "[INST]", "[OUT]"]}
num_added = tok.add_special_tokens(extra_tokens)

# ---- Model ----
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=dtype,
)
if num_added > 0:
    model.resize_token_embeddings(len(tok))

model.gradient_checkpointing_enable()

# ---- LoRA ----
lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["Wqkv","out_proj"],  # MPT attention module names
    task_type="CAUSAL_LM"
)

# ---- Collator (loss only after [OUT]) ----
collator = DataCollatorForCompletionOnlyLM(response_template="[OUT]", tokenizer=tok)

# (3) Use SFTConfig instead of TrainingArguments
# (4) Set packing=False since CompletionOnly collator is incompatible with packing
sft_config = SFTConfig(
    output_dir="out-mpt7b-biz-lora",
    num_train_epochs=2,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    bf16=(dtype == torch.bfloat16),
    fp16=(dtype == torch.float16),
    gradient_checkpointing=True,
    max_seq_length=CTX_LEN,
    dataset_text_field="text",
    packing=False,                  # important with CompletionOnly collator
    optim="paged_adamw_32bit",
    report_to="none",
    max_grad_norm=1.0,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    peft_config=lora,
    train_dataset=train,
    eval_dataset=eval_,
    args=sft_config,
    data_collator=collator,
)

trainer.train()
trainer.save_model("out-mpt7b-biz-lora/adapter")
tok.save_pretrained("out-mpt7b-biz-lora/tokenizer")
print("Saved LoRA adapter to out-mpt7b-biz-lora/adapter")

