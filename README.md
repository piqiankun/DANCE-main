# DANCE

**DANCE** (Dynamic and Adaptive Neuron Coalition Enhancement) is a graph-guided framework for joint debiasing across high- and low-resource languages. It identifies shared bias-knowledge neurons and profiles their debiasing contributions, general-knowledge sensitivity, and cross-lingual functionality. A signed graph captures synergistic and suppressive interactions, enabling a coalition optimizer to jointly determine neuron-specific enhancement weights. These weights are fixed for each model and language pair and applied to both languages during inference, with model parameters frozen. DANCE aims to reduce social bias while preserving general capabilities.

## Environment

Python 3.10 or newer is recommended. Install dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model checkpoints may be given as a Hugging Face model identifier or as a local path supplied at runtime. Hugging Face authentication is optional and is read from `HF_TOKEN`. The data-preparation utility reads its service credential from `DASHSCOPE_API_KEY`. No credential is stored in this repository.

```bash
export HF_TOKEN="<token>"
export DASHSCOPE_API_KEY="<key>"
```

## Data preparation

Place locally obtained datasets under `data/`. The scripts expect the following descriptive filenames where applicable:

| File | Used for |
| --- | --- |
| `know_bias_data_extend.csv` | Bias-neuron discovery and pairwise interaction analysis. |
| `bias_importance_data.csv` | Bias-importance and contribution experiments. |
| `OBQA11.parquet` | General-importance evaluation. |
| `cross_language_source.jsonl` | Source examples for distractor generation. |
| `cross_language.jsonl` | Prepared cross-language examples. |
| `neuron_bias_contributions.csv` | Functional-module identification. |
| `candidate_neuron_pairs.csv` | Pairwise interaction analysis. |
| `bias_neuron_scores.csv` | Coalition optimization input. |
| `general_neuron_scores.csv` | Coalition optimization input. |
| `cross_language_neuron_scores.csv` | Coalition optimization input. |
| `stable_neuron_pairs.csv` | Coalition optimization input. |

Datasets with third-party licenses must be acquired separately and used according to their original terms.

## Bias-neuron discovery

Compute attribution scores and summarize the resulting neurons:

```bash
CUDA_VISIBLE_DEVICES=0 python discover_bias_neurons.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --condition Gender_identity

CUDA_VISIBLE_DEVICES=0 python analyze_bias_neurons.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --condition Gender_identity
```

Run the same procedure for each language, then intersect the resulting neuron sets to obtain multilingual shared neurons.

## Importance analysis

Measure bias-specific and general-domain importance:

```bash
CUDA_VISIBLE_DEVICES=0 python measure_bias_importance.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct

CUDA_VISIBLE_DEVICES=0 python measure_general_importance.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct
```

## Cross-language analysis

Prepare the data, extract activations, and run the ablation experiment:

```bash
python prepare_cross_language_data.py \
  --input data/cross_language_source.jsonl \
  --output data/cross_language.jsonl

CUDA_VISIBLE_DEVICES=0 python extract_cross_language_activations.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --input_file data/cross_language.jsonl \
  --output_file results/cross_language.activations.jsonl

CUDA_VISIBLE_DEVICES=0 python measure_cross_language_importance.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --input_file results/cross_language.activations.jsonl
```

## Neuron interaction analysis

Generate contribution scores, identify candidate modules, and evaluate pairwise interactions:

```bash
CUDA_VISIBLE_DEVICES=0 python measure_neuron_contributions.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct

python identify_neuron_modules.py \
  --input data/neuron_bias_contributions.csv \
  --output_dir results

for strength in 1.5 2.0 2.5 3.0 3.5; do
  CUDA_VISIBLE_DEVICES=0 python analyze_neuron_interactions.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --pairs_file data/candidate_neuron_pairs.csv \
    --enhance_value "$strength"
done

python analyze_interaction_stability.py
```

## Adaptive coalition weighting

Compute adaptive weights from bias, general-domain, cross-language, and stable-pair statistics:

```bash
python optimize_neuron_coalitions.py \
  --data-dir data \
  --gamma 1.0 \
  --output results/neuron_coalition_weights.csv
```

## Benchmark evaluation

Use the benchmark entry points for the datasets available in `data/`:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_bias_benchmarks.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --dataset crows \
  --all

CUDA_VISIBLE_DEVICES=0 python evaluate_multilingual_benchmarks.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --dataset ban_stereo \
  --all
```

