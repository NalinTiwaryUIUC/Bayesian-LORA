venv:
	python3 -m venv .venv

install:
	pip3 install -e .
	pip3 install -r requirements.txt

install-lora:
	pip3 install -r requirements_lora.txt

test:
	pytest -q

format:
	black . && ruff check --fix .

# =============================================================================
# LoRA Experiments with Established Models and Datasets
# =============================================================================

# -----------------------------------------------------------------------------
# Individual Experiments
# -----------------------------------------------------------------------------

# SST-2 Sentiment Analysis
experiment-sst2-bert-sgld:
	python3 scripts/train_lora_hf.py --config configs/experiment_sst2_bert_sgld.yaml

experiment-sst2-roberta-sam-sgld:
	python3 scripts/train_lora_hf.py --config configs/experiment_sst2_roberta_sam_sgld.yaml

# MRPC Paraphrase Detection
experiment-mrpc-distilbert-asgld:
	python3 scripts/train_lora_hf.py --config configs/experiment_mrpc_distilbert_asgld.yaml

# IMDB Sentiment Analysis
experiment-imdb-bert-sam-sgld-r1:
	python3 scripts/train_lora_hf.py --config configs/experiment_imdb_bert_sam_sgld_r1.yaml

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

# Evaluate SST-2 experiments
eval-sst2-bert-sgld:
	python3 scripts/eval_lora_hf.py --config configs/experiment_sst2_bert_sgld.yaml --k 20

eval-sst2-roberta-sam-sgld:
	python3 scripts/eval_lora_hf.py --config configs/experiment_sst2_roberta_sam_sgld.yaml --k 20

# Evaluate MRPC experiments
eval-mrpc-distilbert-asgld:
	python3 scripts/eval_lora_hf.py --config configs/experiment_mrpc_distilbert_asgld.yaml --k 20

# Evaluate IMDB experiments
eval-imdb-bert-sam-sgld-r1:
	python3 scripts/eval_lora_hf.py --config configs/experiment_imdb_bert_sam_sgld_r1.yaml --k 20

# -----------------------------------------------------------------------------
# Quick Commands
# -----------------------------------------------------------------------------

# Run all experiments
experiments-all: experiment-sst2-bert-sgld experiment-sst2-roberta-sam-sgld experiment-mrpc-distilbert-asgld experiment-imdb-bert-sam-sgld-r1

# Evaluate all experiments
eval-all: eval-sst2-bert-sgld eval-sst2-roberta-sam-sgld eval-mrpc-distilbert-asgld eval-imdb-bert-sam-sgld-r1

# -----------------------------------------------------------------------------
# Clean up
# -----------------------------------------------------------------------------

clean:
	rm -rf runs/experiment_*
	rm -rf runs/c10_r18_sgld
	rm -rf runs/c100_wrn2810_*

clean-lora:
	rm -rf runs/experiment_*

clean-cifar:
	rm -rf runs/c10_r18_sgld
	rm -rf runs/c100_wrn2810_*
