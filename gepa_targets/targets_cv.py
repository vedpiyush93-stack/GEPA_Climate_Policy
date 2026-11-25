"""5-fold cross-validation for climate-target detection, grouped by country.

Each fold holds out one fifth of the countries. Within a fold every method is rebuilt from scratch using only the
other four fifths, then all methods are scored on the same random sample of held-out passages:

  - classical ML: TF-IDF + logistic regression / linear SVM / random forest, each grid-searched with country-grouped
    inner cross-validation; the family with the best inner score is used (never chosen on the test fold)
  - Claude Haiku and Claude Sonnet with the annotators' definition as the prompt (no labels used)
  - Claude Haiku tuned by GEPA (Claude Sonnet as reflection model) on 300 training + 60 validation passages
"""

import json
import pathlib
import random
import time

import dspy
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

from . import targets, targets_ml
from .lm import claude_lm

OUT = pathlib.Path("results/cv")


def folds(k=5, seed=0, n_test=150, n_train=300, n_val=60):
    """List of dicts with DataFrames: pool (all non-fold passages), train, val, test (sample of the fold)."""
    df = pd.read_parquet(targets.DATA)
    countries = sorted(df.geography.unique())
    random.Random(seed).shuffle(countries)
    groups = {c: i % k for i, c in enumerate(countries)}
    out = []
    for f in range(k):
        in_fold = df.geography.map(groups).eq(f)
        test = df[in_fold].sample(n=min(n_test, int(in_fold.sum())), random_state=seed)
        pool = df[~in_fold].sample(frac=1, random_state=seed)
        out.append({"fold": f, "pool": pool, "train": pool.iloc[:n_train], "val": pool.iloc[n_train:n_train + n_val],
                    "test": test, "test_countries": int(df[in_fold].geography.nunique())})
    return out


def _examples(d):
    return [dspy.Example(passage=" ".join(r.text.split()), geography=r.geography, region=r.region,
                         **{k: bool(getattr(r, v)) for k, v in targets.LABELS.items()}).with_inputs("passage")
            for r in d.itertuples()]


def _ml(fold):
    """Grid-search each family on the pool, keep the family with the best inner CV score, predict the test fold."""
    X, Y, g = targets_ml.norm(fold["pool"].text), fold["pool"][targets_ml.COLS].values, fold["pool"].geography.values
    X_test = tuple(targets_ml.norm(fold["test"].text))
    best = None
    for name in targets_ml.MODELS:   # cached: instant after the first run
        pred, score, params = targets_ml.grid_fit_predict(name, tuple(X), Y, g, X_test, n_jobs=4)
        if best is None or score > best[2]:
            best = (name, pred, score, params)
    df = targets_ml.to_rows(fold["test"], best[1])
    return df, {"family": best[0], "inner_cv": round(best[2], 3), "params": best[3]}


def run_fold(f, threads=4, gepa_budget=300, verbose=True):
    """Run every method on fold f; saves per-passage results and a summary under results/cv/. Cached calls are free."""
    OUT.mkdir(parents=True, exist_ok=True)
    fold = folds()[f]
    train, val, test = _examples(fold["train"]), _examples(fold["val"]), _examples(fold["test"])
    haiku, sonnet = claude_lm("haiku"), claude_lm("sonnet")
    t0, res, info = time.time(), {}, {"fold": f, "test_passages": len(test), "test_countries": fold["test_countries"]}
    res["Classical ML"], info["ml"] = _ml(fold)
    res["Claude Haiku"] = targets.run(targets.TargetClassifier(), test, haiku, threads=threads)
    res["Claude Sonnet"] = targets.run(targets.TargetClassifier(), test, sonnet, threads=threads)
    with dspy.context(lm=haiku):
        gepa = dspy.GEPA(metric=targets.metric, reflection_lm=sonnet, max_metric_calls=gepa_budget, num_threads=threads,
                         track_stats=True, log_dir=str(OUT / f"gepa_fold{f}"), seed=0)
        tuned = gepa.compile(targets.TargetClassifier(), trainset=train, valset=val)
    tuned.save(str(OUT / f"tuned_fold{f}.json"))
    res["Claude Haiku + GEPA"] = targets.run(tuned, test, haiku, threads=threads)
    log = json.load(open(OUT / f"gepa_fold{f}" / "run_log.json"))
    seen = {i for it in log for t in it.get("tasks", []) for i in t.get("subsample_ids", [])}
    info["gepa"] = {"val_scores": tuned.detailed_results.val_aggregate_scores,
                    "labelled_passages_used": len(seen) + len(val),
                    "instructions": [p.signature.instructions for _, p in tuned.named_predictors()][0]}
    info["minutes"] = round((time.time() - t0) / 60, 1)
    rows = []
    for name, d in res.items():
        d.to_pickle(OUT / f"fold{f}_{name.replace(' ', '_').replace('+', 'plus')}.pkl")
        rows.append({"fold": f, **targets.summary(d, name)})
    info["summary"] = rows
    json.dump(info, open(OUT / f"fold{f}.json", "w"), indent=1)
    if verbose:
        print(pd.DataFrame(rows).to_string(index=False), f"\n{info['minutes']} min", flush=True)
    return res, info


def load_results(k=5):
    """Per-fold summaries (DataFrame) and per-passage results {fold: {setup: DataFrame}} from results/cv/."""
    infos = [json.load(open(OUT / f"fold{f}.json")) for f in range(k)]
    per = {f: {r["setup"]: pd.read_pickle(OUT / f"fold{f}_{r['setup'].replace(' ', '_').replace('+', 'plus')}.pkl")
               for r in infos[f]["summary"]} for f in range(k)}
    return pd.DataFrame([r for i in infos for r in i["summary"]]), per, infos
