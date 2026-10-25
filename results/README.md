# Generated results

Stage outputs are written here by default and ignored by Git.

- Attribution tensors, neuron summaries, importance scores, contribution profiles,
  candidate pairs, pairwise interaction results, and stable-pair statistics remain
  directly under this directory.
- `benchmarks/` contains category-specific evaluation CSVs (previously `final_res/`).
- `cross_language_importance.json` contains activation-replacement statistics.
- Coalition optimization writes a weights CSV and a sibling explanation JSON.
- `enhance_factor_results.json` is an optional input to multilingual evaluation.
  Its existing format is a list of objects with `position` (an integer neuron index)
  and `ratio` fields. Match the evaluator's neuron-coordinate handling when
  preparing this file; it is not the optimizer's explanation JSON format.

Use explicit output arguments to keep separate runs or language pairs apart.
Do not commit local logs, credentials, or generated dataset content.
