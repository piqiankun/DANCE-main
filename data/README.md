# Local input data

Acquire datasets under their original licenses and place them here. Dataset
contents are ignored by Git. Do not add private records or credentials.

| File / directory | Used by |
| --- | --- |
| `know_bias_data_extend.csv` | Bias-neuron discovery and pairwise interactions. |
| `know_bias_data_all_combine_extend.csv` | Optional extended discovery mode. |
| `bias_importance_data.csv` | Bias importance and per-example contribution profiling. |
| `OBQA11.parquet` | General-importance ablation. |
| `cross_language_source.jsonl` | Distractor generation; requires `en_input`, `en_target`. |
| `cross_language.jsonl` | Activation extraction; includes aligned probes and answers. |
| `cross_language.activation.jsonl` | Default input for activation replacement. |
| `neuron_bias_contributions.csv` | Community/candidate discovery: `layer`, `neuron`, `sample_*`. |
| `candidate_neuron_pairs.csv` | Pairwise analysis: `layer_A`, `neuron_A`, `layer_B`, `neuron_B`. |
| `bias_neuron_scores.csv` | Coalition feature table: `layer`, `neuron`, `weight`. |
| `general_neuron_scores.csv` | Coalition feature table: `layer`, `neuron`, `weight`. |
| `cross_language_neuron_scores.csv` | Coalition feature table: `layer`, `neuron`, `weight`. |
| `stable_neuron_pairs.csv` | Pair coordinates plus `relationship` (`synergistic` or `suppressive`). |
| `BBQ/`, `PakBBQ/`, `bengali/` | Dataset-specific benchmark records. |
| `crows_pairs_anonymized.csv`, `BanStereoset.csv` | Social-bias benchmarks. |
| `OBQA.parquet`, `arc_easy.parquet`, `arc_challenge.parquet` | English capability benchmarks. |
| `BEnQA.csv`, `bluck.csv`, `urdummlu.json` | Bengali/Urdu capability benchmarks. |

Select the shared-neuron coordinates for the relevant backbone and language pair
in each profiling/evaluation module before an experiment. See `docs/workflow.md`
for intermediate-file preparation.
