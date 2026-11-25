# GEPA for climate-policy target detection

Can a small, cheap language model learn an expert's judgment from a few dozen labelled examples, and explain its answers?
This repository applies **GEPA** (Genetic-Pareto reflective prompt evolution; Agrawal et al., ICLR 2026,
[arXiv:2507.19457](https://arxiv.org/abs/2507.19457)) to a real expert-labelled task: finding climate targets in
national laws and policies.

## Task and data

[Climate Policy Radar](https://climatepolicyradar.org)'s *national-climate-targets* dataset
([Hugging Face](https://huggingface.co/datasets/ClimatePolicyRadar/national-climate-targets), CC-BY-4.0;
Juhasz et al. 2024, [arXiv:2404.02822](https://arxiv.org/abs/2404.02822)): 2,610 passages from the climate laws,
policies and UNFCCC submissions of 192 countries, labelled by three domain experts. For each passage, three yes/no
labels: **net zero target**, **emission reduction target**, **other target** (any other goal with a number and a
deadline). A copy is included in `data/` under its licence.

## Methods

The starting prompt is the annotators' own written definition ("a target is an aim to achieve a specific outcome, that
is quantifiable and is given a deadline", plus one line per target type). GEPA then loops: the student model (Claude
Haiku) answers a few training passages, its answers are checked against the experts' labels, a stronger model (Claude
Sonnet) reads the student's reasoning on the mistakes and rewrites the prompt, and a rewrite is kept only if it scores
higher on held-out validation passages.

## Results

5-fold cross-validation by country (every method rebuilt in each fold; 150 held-out passages per fold):

| Setup | Labelled passages | How it was built | All 3 labels right (mean ± SD) | Wrongly flagged as targets |
|---|---|---|---|---|
| Classical ML | 1,878-2,200 per fold | TF-IDF word features + logistic regression / linear SVM / random forest, grid search with country-grouped cross-validation | 79% ± 2% | 7% |
| Claude Haiku | 0 | zero-shot: the prompt is the annotators' published definition | 81% ± 6% | 21% |
| Claude Sonnet | 0 | zero-shot: same prompt, larger model | 82% ± 3% | 21% |
| Claude Haiku + GEPA | 75-87 per fold | same small model; GEPA rewrote its prompt, with Claude Sonnet as the reflection model | **86% ± 4%** | 13% |

Rules GEPA added to the prompt (from the main run; see `results/targets_prompts_before_after.md` for the full prompt):

- **Be strict:** count a goal only if it has both a number and a deadline.
- **Energy savings ≠ emission cuts:** cutting energy use is an "other" target, not an emission-reduction target.
- **Vague does not count:** aims like "decarbonise the sector" with no figure are not targets.
- **Plans are not promises:** projects, policies, studies and financing are not targets.

The tuned model also explains every answer; the notebook shows examples, including passages it fixed.

## Reproduce

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/jupyter lab gepa_climate_targets.ipynb      # or: jupyter nbconvert --execute --to notebook --inplace ...
```

- **Exact replay:** every LLM response is stored in `cache/dspy`, so a re-run reproduces every number without making
  new calls. The first run re-fits the classical ML models (a few minutes) and caches them in `cache/ml`.
- **Fresh runs:** LLM calls go through the Claude Code CLI (`claude -p`), so it must be installed and logged in. Delete
  `cache/dspy` (or set `GEPA_CACHE_DIR`) to call the models afresh; answers then vary slightly. Set `RUN_GEPA = True`
  in the notebook to re-run the optimiser (about 6 minutes).

## Layout

| Path | Contents |
|---|---|
| `gepa_climate_targets.ipynb` | The full analysis: data, baselines, GEPA, results, cross-validation |
| `gepa_targets/targets.py` | Country-level split, DSPy classifier, GEPA feedback metric, scoring |
| `gepa_targets/targets_ml.py` | Classical ML baselines with grid search |
| `gepa_targets/targets_cv.py` | 5-fold cross-validation by country |
| `gepa_targets/lm.py` | DSPy language-model engine that calls Claude through `claude -p`, with usage caps |
| `data/` | Climate Policy Radar dataset (CC-BY-4.0) |
| `results/` | GEPA logs, tuned prompts, per-fold results, tables and charts |
| `cache/dspy` | Cached LLM responses for exact replay |

## References

- Agrawal et al. *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.* ICLR 2026. arXiv:2507.19457.
- Juhasz et al. *Identifying Climate Targets in National Laws and Policies using Machine Learning.* 2024. arXiv:2404.02822.
- DSPy: https://dspy.ai
