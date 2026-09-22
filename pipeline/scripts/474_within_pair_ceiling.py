"""Stage 474: how much better must the ranker be, in the one place the loss lives?

Stage 200 located the evidence loss precisely: candidate recall costs 0.0032 and
ordering inside the top 20 costs 0.3076. Stage 473 then showed the loss is
uniform across pair regimes, so there is no subpopulation to target. What was
still missing is the quantity that makes the problem tractable to think about -
**how separable are the gold hands from the other candidates of the same pair,
and how separable would they have to be?**

Both are computable, and together they turn "improve the evidence ranker" into a
number.

First, what separates a gold hand from its own pair's other top-20 candidates,
in within-pair z so that pair-level scale is already removed. The discriminators
are real and strong: `hs_max` +0.94 z (t 38.3), `hsdir` +0.69, `ahead_fold`
+0.58, `fold_better_flag` +0.54, `transfer_event` +0.50, `ps_partner` +0.45. And
earliness appears with the opposite sign - `events_before` -0.42,
`contact_before` -0.39, `timeline_position` -0.39 - reproducing stage 65's
finding inside the shortlist. **None of this is missing from the model**; every
one of those columns is already a feature.

Second, and this is the number: the production score's **within-pair AUC among
its own top 20 is 0.9116**. A simulation of MAP@5 at that AUC - gold and non-gold
drawn from unit normals separated so that AUC matches, five gold among twenty
candidates - returns **0.6807** against the engine's actual 0.6892, so the
simple model is faithful to within 0.009 and can be inverted.

Inverting it says what any further work has to achieve:

    MAP@5 0.7000  needs within-pair AUC 0.9204
    MAP@5 0.7677  needs               0.9468      <- what closes the board gap
    MAP@5 0.8000  needs               0.9558
    MAP@5 0.9000  needs               0.9837

So closing the 0.02035 gap to third place means taking the within-pair ranking
error from 0.0884 to 0.0532 - **cutting it by forty per cent**, on a shortlist
the model already chose, using signal whose strongest single component separates
at about one standard deviation.

That is the honest shape of the remaining problem. It is not a missing feature
and not a subpopulation: it is a forty per cent error reduction in a
discrimination that is intrinsically weak. It also explains why every attempt
this week landed between +0.005 and +0.008 MAP - MIL's action head, the CCR
block, the isolation comparator, the fusion - each was moving an AUC of 0.9116
by a thousandth or two.

**And the feature set is already spent.** A fresh discriminator trained directly
on the within-pair problem, all 237 columns of the anatomy table, cross-fit by
table, reaches **0.9032** - *below* the production score's 0.9116 - and 0.8907
once the production score and its rank are removed from its inputs. So the
shipped engine is at or above what any direct use of these features extracts;
the lambdarank objective, the twelve-seed bag and the family specialists are a
better use of the same information than a single classifier is.

The ceiling **of this feature set under this harness** is therefore about
**0.91** within-pair AUC against the **0.945** the board gap requires.

The scope of that statement matters and it is easy to overstate. 0.9116 is not an
information-theoretic bound on the problem. It is what 237 engineered columns and
a supervised ranker reach. Raw action and sequence information could exceed it,
and stage 483 is the reason to think so: the noisy-OR head recovers the trigger
action out of fold, at 98.7% preflop and 95.9% raise on isolation with a median
z-margin of 5.80, without any action ever being labelled. The structure is
learnable; it did not convert into MAP@5 here. What is established is that
**better fitting of these columns closes nothing**, and that finding the
information which would is a larger piece of work than the hours remaining.

    python scripts/474_within_pair_ceiling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C

TOP = 20
SKIP = {"hand_id", "table_id", "phase", "p1", "p2", "pair_id", "behavior_family",
        "is_evidence", "evidence_rank", "s1", "r", "started_at"}


def map5_at_auc(auc: float, n: int = 20, k: int = 5, n_gold: int = 5,
                trials: int = 4000, seed: int = 0) -> float:
    """MAP@5 a ranker with this within-pair AUC would reach on this shortlist.

    Gold and non-gold scores are unit normals separated by `d` with
    `AUC = Phi(d / sqrt(2))`; the ranking is the realised ordering, so the
    simulation reproduces the metric rather than approximating it.
    """
    rng = np.random.default_rng(seed)
    d = norm.ppf(auc) * np.sqrt(2)
    out = np.empty(trials)
    for i in range(trials):
        sc = np.r_[rng.normal(d, 1, n_gold), rng.normal(0, 1, n - n_gold)]
        lab = np.r_[np.ones(n_gold), np.zeros(n - n_gold)]
        rel = lab[np.argsort(-sc)[:k]]
        out[i] = ((np.cumsum(rel) / np.arange(1, k + 1)) * rel).sum() / min(n_gold, k)
    return float(out.mean())


def auc_for(target: float, lo: float, hi: float = 0.9999) -> float:
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if map5_at_auc(mid) < target else (lo, mid)
    return hi


def main() -> None:
    d = pl.read_parquet(C.ARTIFACTS / "evidence_anatomy.parquet").filter(pl.col("r") <= TOP)
    y = d["is_evidence"].to_numpy()
    print(f"top-{TOP} rows {d.height}, pairs {d['pair_id'].n_unique()}, "
          f"gold {int(y.sum())}, other {int((1 - y).sum())}", flush=True)

    feats = [c for c in d.columns if c not in SKIP and d[c].dtype.is_numeric()]
    rows = []
    for c in feats:
        z = (d.select("pair_id", c).with_columns(
            ((pl.col(c) - pl.col(c).mean().over("pair_id"))
             / (pl.col(c).std().over("pair_id") + 1e-9)).alias("z"))["z"].to_numpy())
        a, b = z[y == 1], z[y == 0]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 50 or len(b) < 50:
            continue
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        rows.append({"feature": c, "gold_minus_other_z": round(float(diff), 4),
                     "t": round(float(diff / se) if se > 0 else 0.0, 1)})
    rows.sort(key=lambda r: -abs(r["t"]))
    print("\nwithin-pair separation, gold against the pair's other top-20 candidates:")
    for r in rows[:12]:
        print(f"  {r['feature']:32s} {r['gold_minus_other_z']:+8.3f} z   t {r['t']:+7.1f}")

    aucs = [roc_auc_score(g["is_evidence"].to_numpy(), g["s1"].to_numpy())
            for _, g in d.group_by("pair_id")
            if 0 < g["is_evidence"].sum() < g.height]
    cur = float(np.mean(aucs))
    sim = map5_at_auc(cur)
    print(f"\nwithin-pair AUC of the production score: {cur:.4f} over {len(aucs)} pairs")
    print(f"simulated MAP@5 at that AUC: {sim:.4f}   (engine reads 0.6892)")

    need = {t: auc_for(t, cur) for t in (0.70, 0.7677, 0.80, 0.90)}
    print("\nAUC required, on the same shortlist:")
    for t, a in need.items():
        print(f"  MAP@5 {t:.4f} -> {a:.4f}   (error {1 - a:.4f}, "
              f"a {100 * (1 - (1 - a) / (1 - cur)):.0f}% cut)")

    (C.ARTIFACTS / "474_within_pair_ceiling.json").write_text(json.dumps(
        {"within_pair_auc": cur, "simulated_map5": sim, "pairs": len(aucs),
         "required_auc": {str(k): v for k, v in need.items()},
         "separation": rows[:40]}, indent=2))


if __name__ == "__main__":
    main()
