python - <<'PY'
import json, os
from transformers import AutoTokenizer

OUT="mpt7b-biz-merged"  # your merged folder
cfg_path=os.path.join(OUT,"config.json")

with open(cfg_path) as f:
    cfg=json.load(f)

# Ensure model_type + architectures look like MPT
cfg["model_type"]="mpt"
cfg.setdefault("architectures", ["MptForCausalLM"])

# Ensure attn_config exists and has ALiBi enabled
attn=cfg.get("attn_config", {})
attn["alibi"]=True
# Optional but harmless defaults MPT commonly ships with
attn.setdefault("alibi_bias_max", 8)
attn.setdefault("softmax_scale", None)
cfg["attn_config"]=attn

# Keep max seq len at 2048 (MPT-7B-instruct default)
cfg["max_seq_len"]=2048

# Ensure vocab_size matches tokenizer after you added [SYS],[INST],[OUT]
tok=AutoTokenizer.from_pretrained(OUT, use_fast=True)
cfg["vocab_size"]=len(tok)

with open(cfg_path,"w") as f:
    json.dump(cfg, f, indent=2)
print("Patched", cfg_path, "✓")
PY

