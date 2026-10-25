# Workflow and artifact handoffs

Run each stage with `python run.py <command> [arguments]`. The launcher forwards
arguments to the existing stage implementation and does not orchestrate a full
experiment automatically. Neural attribution, replacement, and benchmark stages
require local datasets, compatible model dependencies, and suitable hardware.

| Stage | Input | Output / next step |
| --- | --- | --- |
| `prepare-data` | `cross_language_source.jsonl` with `en_input` and `en_target` | Four-option examples in `cross_language.jsonl`; complete the required filtering before profiling. |
| `discover-neurons` | Bias-knowledge CSV questions for each language | Attribution tensors and selected-neuron CSVs in `results/`. |
| `analyze-neurons` | Attribution outputs matching model and options | Group/detail summaries in `results/`; intersect per-language selections. |
| `bias-importance` | Shared-neuron list and `bias_importance_data.csv` | Bias contribution statistics and weights in `results/`. |
| `general-importance` | Shared-neuron list and `OBQA11.parquet` | Baseline and ablation statistics in `results/`. |
| `extract-activations` | Aligned cross-language JSONL and a neuron list | Low-resource activation records at the requested output path. |
| `cross-language-importance` | Activation JSONL and aligned English questions/options | Replacement effects in `results/cross_language_importance.json`. |
| `neuron-contributions` | Bias questions and a neuron list | CSV containing neuron coordinates and per-example attributions. |
| `identify-modules` | Contribution-profile CSV | Timestamped candidate CSV in the requested output directory. |
| `analyze-interactions` | Candidate-pair CSV and bias questions | `pairwise_neuron_synergy_<model>_enhance_<strength>.csv`. |
| `interaction-stability` | Interaction outputs for multiple strengths | `stable_neuron_pairs_all_enhance.csv`. |
| `optimize-coalitions` | Three feature CSVs and `stable_neuron_pairs.csv` | Adaptive weights CSV and explanation JSON. |
| `evaluate-bias` / `evaluate-multilingual` | Benchmark data and configured neuron interventions | Benchmark tables under `results/benchmarks/`. |

## Preparing intermediate inputs

The original scripts use different output schemas. Use the formats expected by
the next stage rather than assuming that every output is directly interchangeable:

- Candidate outputs from `identify-modules` use `layer_i`, `neuron_i`, `layer_j`,
  and `neuron_j`. The pairwise analyzer expects `layer_A`, `neuron_A`, `layer_B`,
  and `neuron_B`; rename these four columns when preparing `candidate_neuron_pairs.csv`.
- Contribution-profile inputs for `identify-modules` need `layer`, `neuron`, and
  at least one `sample_*` column. Preserve the per-example attribution columns.
- The three coalition feature files need `layer`, `neuron`, and `weight` columns.
  Prepare them from the intended bias, general, and cross-language score definitions.
- The stability analyzer writes `stable_neuron_pairs_all_enhance.csv`. Supply the
  retained signed pairs as `stable_neuron_pairs.csv` in the optimizer's input directory.
  Only `synergistic` and `suppressive` relationship labels are accepted by the optimizer.
- The multilingual evaluator's optional factor file uses a list of objects with
  `position` and `ratio` fields. It is separate from the optimizer's explanation JSON
  and must be prepared in the evaluator's expected format.

## Stability analysis options

```bash
python run.py interaction-stability \
  --model-name Llama-3.2-3B-Instruct \
  --results-dir results \
  --enhance-values 1.5 2.0 2.5 3.0 3.5
```

The model-name value must match the basename used in interaction-result filenames.
Settings retain their previous defaults when flags are omitted.
