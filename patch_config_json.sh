#!/bin/bash

python - <<'PY'
import json, pathlib
p = pathlib.Path("mpt7b-biz-merged/config.json")
cfg = json.loads(p.read_text())

attn = cfg.setdefault("attn_config", {})
attn.setdefault("alibi", True)
attn.setdefault("alibi_bias_max", 8)
attn.setdefault("attn_impl", "triton")
attn.setdefault("attn_pdrop", 0.0)
attn.setdefault("qk_ln", False)
attn.setdefault("clip_qkv", None)
attn.setdefault("softmax_scale", None)
attn.setdefault("rope", False)
attn.setdefault("prefix_lm", False)
attn.setdefault("attn_uses_sequence_id", False)

cfg["learned_pos_emb"] = False
p.write_text(json.dumps(cfg, indent=2))
print("Patched", p)
PY

