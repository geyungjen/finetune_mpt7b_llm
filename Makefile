PYTHON=python3
VENV ?= mpt7b-common

# CUDA 12.1 wheels index
TORCH_IDX=--extra-index-url https://download.pytorch.org/whl/cu121

.PHONY: install_vllm venv install train merge gen serve curl

# create venv if it doesn't exist
venv:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip

# install base deps from requirements.txt (uses the CUDA index)
install: venv
	. $(VENV)/bin/activate && pip install -r requirements.txt $(TORCH_IDX)

# install vLLM (idempotent): only installs if not importable
install_vllm: venv
	. $(VENV)/bin/activate && \
	$(PYTHON) -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('vllm') else 1)" || \
	$(PYTHON) -m pip install "vllm==0.6.6" $(TORCH_IDX)

# optional: your training script name
train:
	. $(VENV)/bin/activate && $(PYTHON) finetune_mpt_lora_min.py

merge:
	. $(VENV)/bin/activate && $(PYTHON) merge_lora.py

gen:
	. $(VENV)/bin/activate && $(PYTHON) quick_generate.py

serve:
	# If you saw torchvision import issues earlier, you can add:
	# export TRANSFORMERS_NO_TORCHVISION=1
	. $(VENV)/bin/activate && vllm serve ./mpt7b-biz-merged \
	  --dtype bfloat16 \
	  --max-model-len 2048 \
	  --gpu-memory-utilization 0.86 \
	  --served-model-name mpt7b-biz-merged \
	  --chat-template ./mpt_chat.jinja

curl:
	curl http://localhost:8000/v1/chat/completions \
	  -H "Content-Type: application/json" \
	  -d '{"model":"mpt7b-biz-merged","messages":[{"role":"system","content":"Be concise and helpful."},{"role":"user","content":"What is working capital?"}],"temperature":0.2,"max_tokens":128,"stop":["<|im_end|>"]}'

