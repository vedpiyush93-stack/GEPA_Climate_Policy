"""Climate-target detection in national laws and policies (Climate Policy Radar dataset, CC-BY-4.0).

Task: for a passage from a national law/policy/UNFCCC submission, decide which kinds of climate target it
contains (multi-label): Net Zero, Reduction (GHG emissions), Other (e.g. renewable energy). Labels are from
3 domain experts (Juhasz et al. 2024, arXiv:2404.02822).

The seed prompt already contains the experts' written definitions, so any gain from GEPA has to come from
learning the judgment calls the definitions leave open.
"""

import random

import dspy
import pandas as pd

DATA = "data/cpr_targets/data/train-00000-of-00001.parquet"
LABELS = {"net_zero": "annotation_NZT", "reduction": "annotation_Reduction", "other": "annotation_Other"}

DEFINITIONS = """A target is an aim to achieve a specific outcome, that is quantifiable and is given a deadline.
Target types:
- net_zero: a commitment to balance GHG emissions with removal, effectively reducing the net emissions to zero.
- reduction: a target referring to a reduction in greenhouse gas emissions, either economy-wide or for a sector.
- other: a target that does not fit into reduction or net zero but satisfies the definition of a target, e.g. renewable energy targets.
A passage can contain several target types, or none."""


class NaiveTargets(dspy.Signature):
    """Does this passage from a government climate policy contain climate targets? Label each type."""

    passage: str = dspy.InputField()
    net_zero: bool = dspy.OutputField()
    reduction: bool = dspy.OutputField()
    other: bool = dspy.OutputField()


class ClassifyTargets(dspy.Signature):
    __doc__ = ("Identify the climate targets in a passage from a national law, policy or UNFCCC submission.\n\n"
               + DEFINITIONS)

    passage: str = dspy.InputField()
    net_zero: bool = dspy.OutputField(desc="passage contains a net zero target")
    reduction: bool = dspy.OutputField(desc="passage contains a GHG emissions reduction target")
    other: bool = dspy.OutputField(desc="passage contains another kind of target (e.g. renewable energy)")


class TargetClassifier(dspy.Module):
    def __init__(self, naive=False):
        super().__init__()
        self.classify = dspy.ChainOfThought(NaiveTargets if naive else ClassifyTargets)

    def forward(self, passage):
        return self.classify(passage=passage)


def load(seed=0, n_test_countries=40, n_val=60, n_train=300, n_test=300):
    """Split by country (never by passage). All South Asian passages go to test, topped up to n_test with
    passages from other held-out countries. Returns DSPy examples (train, val, test)."""
    pool, test = frames(seed, n_test_countries, n_test)

    def ex(r):
        return dspy.Example(passage=" ".join(r.text.split()), geography=r.geography, region=r.region,
                            **{k: bool(getattr(r, v)) for k, v in LABELS.items()}).with_inputs("passage")

    to = lambda d: [ex(r) for r in d.itertuples()]  # noqa: E731
    return to(pool.iloc[:n_train]), to(pool.iloc[n_train:n_train + n_val]), to(test)


def frames(seed=0, n_test_countries=40, n_test=300):
    """(pool, test) DataFrames. pool = every passage from non-test countries, shuffled; the first 300 are GEPA's
    training set and the next 60 its validation set. test = the 300 held-out passages."""
    df = pd.read_parquet(DATA)
    rng = random.Random(seed)
    countries = sorted(df.geography.unique())
    south_asia = sorted(df[df.region == "South Asia"].geography.unique())
    others = [c for c in countries if c not in south_asia]
    rng.shuffle(others)
    test_c = set(south_asia) | set(others[:n_test_countries])
    rest = df[~df.geography.isin(test_c)].sample(frac=1, random_state=seed)
    test = df[df.geography.isin(test_c)]
    sa = test[test.region == "South Asia"]
    test = pd.concat([sa, test[test.region != "South Asia"].sample(n=n_test - len(sa), random_state=seed)])
    return rest, test


def _flags(pred):
    out = {}
    for k in LABELS:
        v = getattr(pred, k, None)
        out[k] = v if isinstance(v, bool) else str(v).strip().lower() in ("true", "yes", "1")
    return out


def score(gold, pred):
    p = _flags(pred)
    wrong = [k for k in LABELS if p[k] != gold[k]]
    return 1.0 - len(wrong) / len(LABELS), wrong, p


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Score = share of the 3 labels right. Feedback = which labels were wrong vs the experts, plus the definition."""
    s, wrong, p = score(gold, pred)
    names = {"net_zero": "Net zero", "reduction": "Reduction", "other": "Other"}
    exp = ", ".join(f"{names[k]}={'yes' if gold[k] else 'no'}" for k in LABELS)
    if not wrong:
        fb = f"Correct. Expert labels: {exp}."
    else:
        parts = [f"{names[k]}: you said {'yes' if p[k] else 'no'}, the experts said {'yes' if gold[k] else 'no'}"
                 for k in wrong]
        fb = (f"Wrong on {len(wrong)} label(s): " + "; ".join(parts) + f". Expert labels: {exp}. "
              "Remember the experts only count an aim that is quantifiable AND has a deadline, and decide which "
              "type it is. Work out what in this passage made the experts decide this way.")
    return dspy.Prediction(score=s, feedback=fb)


def run(program, exs, lm, threads=6):
    with dspy.context(lm=lm):
        res = dspy.Evaluate(devset=exs, metric=lambda g, p, trace=None: score(g, p)[0], num_threads=threads,
                            display_progress=True, provide_traceback=False, failure_score=0.0,
                            max_errors=len(exs))(program)
    rows = []
    for ex, pred, _ in res.results:
        s, wrong, p = score(ex, pred)
        rows.append({"geography": ex.geography, "region": ex.region, "passage": ex.passage,
                     **{f"gold_{k}": ex[k] for k in LABELS}, **{f"pred_{k}": p[k] for k in LABELS},
                     "exact": not wrong, "label_score": s, "wrong": ",".join(wrong),
                     "reasoning": getattr(pred, "reasoning", None)})
    return pd.DataFrame(rows)


def summary(df, name):
    out = {"setup": name, "passages": len(df), "exact_match": round(df.exact.mean(), 3)}
    for k in LABELS:
        g, p = df[f"gold_{k}"], df[f"pred_{k}"]
        tp, fp, fn = int((g & p).sum()), int((~g & p).sum()), int((g & ~p).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[f"F1_{k}"] = round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0
    none = ~(df.gold_net_zero | df.gold_reduction | df.gold_other)
    flagged = df.pred_net_zero | df.pred_reduction | df.pred_other
    out["false_alarm_rate"] = round((none & flagged).sum() / max(none.sum(), 1), 3)  # no-target passages flagged
    return out
