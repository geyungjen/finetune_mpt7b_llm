import json, re, torch, argparse
from pathlib import Path
from statistics import mean
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    import evaluate
    rouge = evaluate.load("rouge")
except Exception:
    rouge = None

BASE_ID = "mosaicml/mpt-7b-instruct"
ADAPTER_DIR = "out-mpt7b-biz-lora/adapter"

PROMPT_TMPL = (
    "<s>[SYS]You are a concise, neutral business assistant.[/SYS]\n"
    "[INST]{inst}\n\n{inp}[/INST]\n[OUT]"
)

def load_model():
    tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    num_added = tok.add_special_tokens({"additional_special_tokens": ["[SYS]","[INST]","[OUT]"]})

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=dtype, device_map="auto")
    if num_added > 0:
        base.resize_token_embeddings(len(tok))

    model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
    return tok, model

def gen(model, tok, inst, inp, max_new_tokens=80):
    prompt = PROMPT_TMPL.format(inst=inst.strip(), inp=inp.strip())
    dev = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(dev)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                 # deterministic
            no_repeat_ngram_size=6,
            repetition_penalty=1.05,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
        )
    gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True)

    # keep only first line
    return text.splitlines()[0].strip()

def score_extract(pred, gold):
    # normalize spaces
    n = lambda s: re.sub(r"\s+", " ", s.strip())
    exact = int(n(pred) == n(gold))
    # optional regex sanity: Policy=<ID>; Effective=YYYY-MM-DD
    ok = bool(re.match(r"^Policy=.+; Effective=\d{4}-\d{2}-\d{2}$", pred))
    return {"exact": exact, "regex_ok": int(ok)}

def score_text(pred, gold):
    scores = {}
    if rouge:
        r = rouge.compute(predictions=[pred], references=[gold])
        scores["rougeL"] = r.get("rougeL", 0.0)
    # simple length sanity: discourage rambling
    len_pen = max(0.0, 1.0 - max(0, len(pred) - 400)/400)
    scores["len_pen"] = len_pen
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_file", default="eval.jsonl")
    args = ap.parse_args()

    data = [json.loads(l) for l in Path(args.eval_file).read_text().splitlines() if l.strip()]
    tok, model = load_model()

    rows, agg = [], {"extract_exact": [], "extract_regex": [], "rougeL": [], "len_pen": []}

    for i, ex in enumerate(data, 1):
        task = ex.get("task","extract")
        inst, inp, gold = ex["instruction"], ex["input"], ex["target"]
        pred = gen(model, tok, inst, inp)

        if task == "extract":
            s = score_extract(pred, gold)
            agg["extract_exact"].append(s["exact"])
            agg["extract_regex"].append(s["regex_ok"])
        else:  # summarize / draft / other
            s = score_text(pred, gold)
            if "rougeL" in s: agg["rougeL"].append(s["rougeL"])
            agg["len_pen"].append(s["len_pen"])

        rows.append({"i": i, "task": task, "pred": pred, "gold": gold, **s})

    # Print per-example
    for r in rows:
        scores = {key: r[key] for key in ('exact', 'regex_ok', 'rougeL', 'len_pen') if key in r}
        print(f"[{r['i']:03d}] {r['task']:9s}  pred={r['pred']}  | gold={r['gold']} | scores={scores}") 

    # Aggregate
    def avg(xs): return round(mean(xs), 4) if xs else None
    summary = {
        "extract_exact@1": avg(agg["extract_exact"]),
        "extract_regex_ok": avg(agg["extract_regex"]),
        "avg_rougeL": avg(agg["rougeL"]),
        "avg_len_pen": avg(agg["len_pen"]),
        "n": len(rows),
    }
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

