"""Classical ML baselines for climate-target detection, on the same 300 held-out test passages as the LLM runs.

TF-IDF features + logistic regression / linear SVM / random forest (one binary decision per label).
Hyper-parameters chosen by grid search with 5-fold cross-validation grouped by country (no country in both
the fitting and the scoring fold), scored on exact match of all three labels.
Two training regimes: the same 360 labelled passages GEPA had (300 train + 60 val), and all 1,839 passages
from non-test countries.

"""
import pathlib
import time

import numpy as np
from joblib import Memory
from sklearn.base import clone
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import targets

COLS = list(targets.LABELS.values())
norm = lambda s: s.map(lambda t: " ".join(t.split()))  # noqa: E731

MODELS = {
    "Logistic regression": (
        Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True)),
                  ("clf", OneVsRestClassifier(LogisticRegression(max_iter=3000)))]),
        {"tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__min_df": [1, 2],
         "clf__estimator__C": [0.3, 1, 3, 10, 30], "clf__estimator__class_weight": [None, "balanced"]}),
    "Linear SVM": (
        Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True)), ("clf", OneVsRestClassifier(LinearSVC()))]),
        {"tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__min_df": [1, 2],
         "clf__estimator__C": [0.03, 0.1, 0.3, 1, 3], "clf__estimator__class_weight": [None, "balanced"]}),
    "Random forest": (
        Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True, max_features=20000)),
                  ("clf", RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1))]),
        {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__max_depth": [None, 40],
         "clf__min_samples_leaf": [1, 2], "clf__class_weight": [None, "balanced_subsample"]}),
}


# Grid searches are deterministic, so their results are cached on disk (cache/ml, keyed on the data, the model and this
# function's code). Re-running the notebook then skips them.
_MEM = Memory(str(pathlib.Path(__file__).resolve().parent.parent / "cache" / "ml"), verbose=0)


@_MEM.cache(ignore=["n_jobs"])
def grid_fit_predict(name, texts, Y, groups, test_texts, n_jobs=-1):
    """Grid-search one model family (country-grouped 5-fold CV); return (test predictions, best CV score, best params)."""
    pipe, grid = MODELS[name]
    gs = GridSearchCV(clone(pipe), grid, scoring="accuracy", cv=GroupKFold(n_splits=5), n_jobs=n_jobs)
    gs.fit(list(texts), Y, groups=groups)
    params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in gs.best_params_.items()}
    return gs.predict(list(test_texts)), float(gs.best_score_), params


def to_rows(test, pred):
    df = pd.DataFrame({"geography": test.geography.values, "region": test.region.values})
    for (k, c), j in zip(targets.LABELS.items(), range(3)):
        df[f"gold_{k}"] = test[c].astype(bool).values
        df[f"pred_{k}"] = pred[:, j].astype(bool)
    g = df[[f"gold_{k}" for k in targets.LABELS]].values
    p = df[[f"pred_{k}" for k in targets.LABELS]].values
    df["exact"] = (g == p).all(axis=1)
    return df


def run_all(regimes=None, verbose=True):
    """Grid-search every model under every training regime; return (summary DataFrame, best params, per-passage
    results keyed by setup name)."""
    pool, test = targets.frames()
    X_test = norm(test.text)
    regimes = regimes or {"same 360 as GEPA": pool.iloc[:360], "all 1,839 non-test": pool}
    rows, best, per = [], {}, {}
    t0 = time.time()
    for regime, tr in regimes.items():
        X, Y, groups = norm(tr.text), tr[COLS].values, tr.geography.values
        for name in MODELS:
            pred, cv_score, params = grid_fit_predict(name, tuple(X), Y, groups, tuple(X_test))
            df = to_rows(test, pred)
            s = targets.summary(df, f"{name} ({regime})")
            s.update({"cv_exact_match": round(cv_score, 3), "train_passages": len(tr),
                      "exact_south_asia": round(df[df.region == "South Asia"].exact.mean(), 3),
                      "exact_india": f"{int(df[df.geography == 'IND'].exact.sum())}/{int((df.geography == 'IND').sum())}"})
            rows.append(s)
            best[s["setup"]] = params
            per[s["setup"]] = df
            if verbose:
                print(f"{s['setup']}: test exact {s['exact_match']}, cv {s['cv_exact_match']}, "
                      f"{time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), best, per
