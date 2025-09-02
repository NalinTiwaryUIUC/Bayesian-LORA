# LoRA Experiments with Established Models and Datasets

This repository implements **LoRA (Low-Rank Adaptation)** experiments using established HuggingFace models and standard datasets, with Bayesian sampling via various MCMC methods.

## 🎯 **What This Implements**

- **True LoRA**: Only low-rank adaptation parameters are trained (base model frozen)
- **Established Models**: BERT, RoBERTa, DistilBERT from HuggingFace
- **Standard Datasets**: GLUE benchmark tasks (SST-2, MRPC, QNLI, RTE, CoLA, STS-B), IMDB, AG News
- **Bayesian Sampling**: SGLD, ASGLD, SAM-SGLD, SAM-SGLD Rank-1
- **Parameter Efficiency**: Dramatically fewer parameters than full fine-tuning

## 🚀 **Quick Start**

### 1. Install Dependencies
```bash
# Core dependencies
pip3 install -e .
pip3 install -r requirements.txt

# LoRA-specific dependencies
pip3 install -r requirements_lora.txt
```

### 2. Run Experiments
```bash
# SST-2 sentiment analysis with BERT + LoRA + SGLD
make experiment-sst2-bert-sgld

# SST-2 with RoBERTa + LoRA + SAM-SGLD
make experiment-sst2-roberta-sam-sgld

# MRPC paraphrase detection with DistilBERT + LoRA + ASGLD
make experiment-mrpc-distilbert-asgld

# IMDB sentiment analysis with BERT + LoRA + SAM-SGLD Rank-1
make experiment-imdb-bert-sam-sgld-r1
```

### 3. Evaluate Results
```bash
# Evaluate SST-2 ensemble
make eval-sst2-bert-sgld

# Evaluate MRPC ensemble
make eval-mrpc-distilbert-asgld

# Run all experiments and evaluations
make experiments-all
make eval-all
```

## 📋 **Complete Step-by-Step Guide**

### **Step 1: Environment Setup**
```bash
# Navigate to your project directory
cd /path/to/Bayesian-LORA

# Create a virtual environment (if you haven't already)
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# You should see (.venv) in your terminal prompt
```

### **Step 2: Install Dependencies**
```bash
# Install the package in editable mode
pip3 install -e .

# Install core requirements
pip3 install -r requirements.txt

# Install LoRA-specific requirements (HuggingFace, PEFT, etc.)
pip3 install -r requirements_lora.txt
```

### **Step 3: Choose Your Experiment**
You have several options. Let's start with the simplest one:

**SST-2 Sentiment Analysis with BERT + LoRA + SGLD:**
- **Task**: Binary sentiment classification (positive/negative)
- **Model**: BERT-base-uncased with LoRA adaptation
- **Sampler**: Stochastic Gradient Langevin Dynamics
- **Dataset**: Stanford Sentiment Treebank v2

### **Step 4: Run the Experiment**
```bash
# Option A: Using Makefile (recommended)
make experiment-sst2-bert-sgld

# Option B: Direct execution
python3 scripts/train_lora_hf.py --config configs/experiment_sst2_bert_sgld.yaml
```

### **Step 5: What Happens During Training**
The script will:
1. **Load BERT model** and apply LoRA layers
2. **Download SST-2 dataset** automatically
3. **Print model info** (LoRA parameters, base parameters)
4. **Train for 10 epochs** using SGLD
5. **Burn-in phase** (1000 steps) to reach posterior
6. **Collect 20 samples** (every 200 steps)
7. **Save samples** to `runs/experiment_sst2_bert_sgld/`

### **Step 6: Monitor Progress**
You'll see output like:
```
Using device: cuda
Dataset: Stanford Sentiment Treebank v2
Task: sentiment_analysis
Number of labels: 2

Building bert-base-uncased with LoRA...
LoRA Model Info:
  - LoRA parameters: 590,592
  - Base parameters: 109,482,240
  - LoRA ratio: 0.54%

Train batches: 2100
Validation batches: 28

Initial evaluation:
Initial loss: 0.6931, accuracy: 0.5000

Training with sgld for 10 epochs...
Epoch 1/10: Train loss: 0.4567, Val loss: 0.4234, accuracy: 0.7891
...

Sampling with sgld...
Burn-in phase: 1000 steps
Collecting 20 samples (thinning: 200)
Saved sample 1/20: runs/experiment_sst2_bert_sgld/sample_0001.pth
...
```

### **Step 7: Evaluate Results**
```bash
# Option A: Using Makefile
make eval-sst2-bert-sgld

# Option B: Direct execution
python3 scripts/eval_lora_hf.py --config configs/experiment_sst2_bert_sgld.yaml --k 20
```

### **Step 8: Understand the Results**
You'll see:
```
Found 20 LoRA samples in runs/experiment_sst2_bert_sgld
Evaluating ensemble of 20 LoRA samples...
LoRA Ensemble@20 — Acc: 93.4% | NLL: 0.1987
Dataset metrics: {'accuracy': 0.934}

LoRA Analysis:
  - Number of samples: 20
  - Ensemble accuracy: 93.4%
  - Sample range: 0 - 19

LoRA Parameter Statistics:
  - Parameter norm mean: 0.1234
  - Parameter norm std: 0.0456
  - Parameter norm CV: 0.3698
```

## 🔧 **Alternative Experiments You Can Run**

### **1. RoBERTa with SAM-SGLD**
```bash
make experiment-sst2-roberta-sam-sgld
```

### **2. DistilBERT with ASGLD (MRPC)**
```bash
make experiment-mrpc-distilbert-asgld
```

### **3. BERT with SAM-SGLD-R1 (IMDB)**
```bash
make experiment-imdb-bert-sam-sgld-r1
```

## 📊 **Expected Timeline**

- **Setup**: 5-10 minutes (first time)
- **Training**: 2-4 hours (depending on GPU)
- **Evaluation**: 5-10 minutes
- **Total**: 2-4 hours for complete experiment

## 🎯 **What You'll Get**

1. **20 LoRA weight samples** from the posterior
2. **Uncertainty quantification** via ensemble predictions
3. **Parameter efficiency** (only ~590K LoRA params vs 110M+ full model)
4. **Research-ready results** for publication

## 🚨 **Troubleshooting**

If you encounter issues:
```bash
# Check if package is installed
pip3 list | grep bayesian

# Reinstall if needed
pip3 install -e .

# Check GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Check import functionality
python3 -c "from bayesian_lora.models.hf_lora import build_huggingface_lora_model; print('Import successful!')"
```

## 📊 **Available Experiments**

### **Models**
- **BERT-base-uncased** (110M params) - Most common baseline
- **RoBERTa-base** (125M params) - Better performance than BERT
- **DistilBERT-base-uncased** (66M params) - Faster, smaller alternative

### **Datasets**
- **SST-2**: Stanford Sentiment Treebank v2 (67K train, 872 val)
- **MRPC**: Microsoft Research Paraphrase Corpus (3.7K train, 408 val)
- **QNLI**: Question-answering NLI (105K train, 5.5K val)
- **RTE**: Recognizing Textual Entailment (2.5K train, 277 val)
- **IMDB**: Movie reviews (25K train, 25K test)
- **AG News**: Topic classification (120K train, 7.6K test)

### **Samplers**
- **SGLD**: Stochastic Gradient Langevin Dynamics
- **ASGLD**: Adaptive SGLD with momentum
- **SAM-SGLD**: Sharpness-Aware SGLD
- **SAM-SGLD Rank-1**: Directional low-rank noise variant

## 🔧 **Configuration Files**

All experiments use organized YAML configuration files in `configs/`:

```yaml
# Example: configs/experiment_sst2_bert_sgld.yaml
data:
  name: sst2
  batch_size: 32
  max_length: 128

model:
  name: bert-base-uncased
  lora:
    rank: 8
    alpha: 16.0
    dropout: 0.1

sampler:
  name: sgld
  step_size: 5.0e-4
  burn_in: 1000
  thin: 200
  num_samples: 20
```

## 📈 **Expected Results**

Based on literature and standard benchmarks:

| Task | Model | LoRA Rank | Expected Accuracy | LoRA Params |
|------|-------|-----------|-------------------|-------------|
| SST-2 | BERT-base | 8 | 91-93% | ~590K |
| SST-2 | RoBERTa-base | 8 | 93-95% | ~590K |
| MRPC | DistilBERT | 4 | 85-88% | ~295K |
| IMDB | BERT-base | 16 | 89-92% | ~1.2M |

## 🧪 **Running Custom Experiments**

### **Individual Experiments**
```bash
python3 scripts/train_lora_hf.py --config configs/experiment_sst2_bert_sgld.yaml
```

### **Evaluation**
```bash
# Single sample
python3 scripts/eval_lora_hf.py --config configs/experiment_sst2_bert_sgld.yaml --single

# Ensemble
python3 scripts/eval_lora_hf.py --config configs/experiment_sst2_bert_sgld.yaml --k 20
```

## 🔬 **Research Applications**

This setup is perfect for:

1. **Reproducing LoRA Results**: Compare with Hu et al. (2021)
2. **Sampler Comparison**: Evaluate MCMC methods on established tasks
3. **Uncertainty Quantification**: Bayesian LoRA with real datasets
4. **Parameter Efficiency**: Study rank vs. performance trade-offs
5. **Transfer Learning**: Adapt pretrained models to new domains

## 📁 **File Structure**

```
├── src/bayesian_lora/
│   ├── models/hf_lora.py          # HuggingFace LoRA models
│   ├── data/glue_datasets.py      # GLUE and standard datasets
│   └── utils/lora_params.py       # LoRA parameter utilities
├── scripts/
│   ├── train_lora_hf.py           # Main training script
│   └── eval_lora_hf.py            # Evaluation script
├── configs/                        # Experiment configurations
│   ├── experiment_sst2_bert_sgld.yaml
│   ├── experiment_sst2_roberta_sam_sgld.yaml
│   ├── experiment_mrpc_distilbert_asgld.yaml
│   ├── experiment_imdb_bert_sam_sgld_r1.yaml
│   └── experiment_comparison_sst2_bert.yaml
├── requirements_lora.txt           # LoRA dependencies
└── Makefile                       # Organized targets
```

## 🎉 **Key Benefits**

- **Industry Standard**: Uses established models and datasets
- **Reproducible**: Standard evaluation protocols and metrics
- **Efficient**: Only LoRA parameters are trained/sampled
- **Flexible**: Easy to modify for new tasks and models
- **Research Ready**: Credible baselines for publication

## 📚 **References**

- **LoRA**: Hu et al. (2021) - LoRA: Low-Rank Adaptation of Large Language Models
- **SGLD**: Welling & Teh (2011) - Bayesian Learning via Stochastic Gradient Langevin Dynamics
- **ASGLD**: Li et al. (2016) - Preconditioned Stochastic Gradient Langevin Dynamics
- **SAM**: Foret et al. (2021) - Sharpness-Aware Minimization
- **SAM-SGLD**: Chen et al. (2022) - Sharpness-Aware Stochastic Gradient Langevin Dynamics

## 🚨 **Requirements**

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.4+
- Datasets 2.12+
- CUDA (recommended for training)

## 🤝 **Contributing**

To add new experiments:

1. **New Models**: Extend `hf_lora.py`
2. **New Datasets**: Add to `glue_datasets.py`
3. **New Configs**: Create YAML files in `configs/` with `experiment_` prefix
4. **Documentation**: Update this README
