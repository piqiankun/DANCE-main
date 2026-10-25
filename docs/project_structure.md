# Project structure and migration

The workflow scripts retain their original filenames inside stage-specific
packages. The root contains only one Python launcher, `run.py`.

| Directory | Responsibility |
| --- | --- |
| `dance/data_preparation/` | Construct cross-language profiling examples. |
| `dance/discovery/` | Discover and summarize bias-knowledge neurons. |
| `dance/profiling/` | Measure neuron functionality across three dimensions. |
| `dance/graph/` | Build candidate pairs, verify interactions, and optimize coalitions. |
| `dance/evaluation/` | Run benchmark evaluation entry points. |
| `dance/core/` | Shared model wrapper and the existing knowledge-neuron implementation. |
| `prompts/` | Evaluation prompt templates, previously in `prompt_instructions/`. |

## Original-to-current paths

| Original file | Current file |
| --- | --- |
| `prepare_cross_language_data.py` | `dance/data_preparation/prepare_cross_language_data.py` |
| `discover_bias_neurons.py` | `dance/discovery/discover_bias_neurons.py` |
| `analyze_bias_neurons.py` | `dance/discovery/analyze_bias_neurons.py` |
| `measure_bias_importance.py` | `dance/profiling/measure_bias_importance.py` |
| `measure_general_importance.py` | `dance/profiling/measure_general_importance.py` |
| `extract_cross_language_activations.py` | `dance/profiling/extract_cross_language_activations.py` |
| `measure_cross_language_importance.py` | `dance/profiling/measure_cross_language_importance.py` |
| `measure_neuron_contributions.py` | `dance/graph/measure_neuron_contributions.py` |
| `identify_neuron_modules.py` | `dance/graph/identify_neuron_modules.py` |
| `analyze_neuron_interactions.py` | `dance/graph/analyze_neuron_interactions.py` |
| `analyze_interaction_stability.py` | `dance/graph/analyze_interaction_stability.py` |
| `optimize_neuron_coalitions.py` | `dance/graph/optimize_neuron_coalitions.py` |
| `evaluate_bias_benchmarks.py` | `dance/evaluation/evaluate_bias_benchmarks.py` |
| `evaluate_multilingual_benchmarks.py` | `dance/evaluation/evaluate_multilingual_benchmarks.py` |
| `llm_utils.py` | `dance/core/llm_utils.py` |
| `kn_code/patch.py` | `dance/core/kn_code/patch.py` |
| `kn_code/__init__.py` | `dance/core/kn_code/__init__.py` |
| `kn_code/setup.py` | `dance/core/kn_code/setup.py` |
| `kn_code/knowledge_neurons.py` | `dance/core/kn_code/knowledge_neurons.py` |
| `kn_code/util/alg_train_dict.py` | `dance/core/kn_code/util/alg_train_dict.py` |
| `kn_code/util/hparams.py` | `dance/core/kn_code/util/hparams.py` |
| `kn_code/util/generate.py` | `dance/core/kn_code/util/generate.py` |
| `kn_code/util/globals.py` | `dance/core/kn_code/util/globals.py` |
| `kn_code/util/nethook.py` | `dance/core/kn_code/util/nethook.py` |
| `kn_code/util/perplexity.py` | `dance/core/kn_code/util/perplexity.py` |
| `kn_code/util/__init__.py` | `dance/core/kn_code/util/__init__.py` |
| `kn_code/util/alg_dict.py` | `dance/core/kn_code/util/alg_dict.py` |
| `kn_code/util/runningstats.py` | `dance/core/kn_code/util/runningstats.py` |
| `kn_code/util/logit_lens.py` | `dance/core/kn_code/util/logit_lens.py` |
| `prompt_instructions/evaluate_s.txt` | `prompts/evaluate_s.txt` |
| `prompt_instructions/evaluate_probs.txt` | `prompts/evaluate_probs.txt` |

## Importing modules

```python
from dance.graph.optimize_neuron_coalitions import CoalitionGraphOptimizer
from dance.core.kn_code import load_model
from dance.core.llm_utils import LLM_KN
```

The `dance` and workflow package initializers are intentionally lightweight.
Importing model utilities still requires the original model dependencies.
The utility registries retained under `dance/core/kn_code/util/` include optional
model-editing integrations from the original code; those integrations require
additional modules that are not distributed in this checkout.

## Running a specific module

From the repository root, the unified launcher and module invocation are equivalent:

```bash
python run.py optimize-coalitions --data-dir data --gamma 1.0
python -m dance.graph.optimize_neuron_coalitions --data-dir data --gamma 1.0
```

Data and prompt files were not embedded in the Python sources. They remain
separate from generated results and local credentials.
