"""Stage 27 (experiment): how well does the pair model find a family it never saw?

The evaluation set counts `other_coordination` pairs as positives, and no
training pair has that family: in development they sit among the unlabelled
pairs, which the pair model learns as soft negatives. Holding one disclosed
family out - its positives turned into unlabelled pairs - and ranking it as if
it were unknown is the only honest estimate of that risk.

Hand-suspicion features that saw the held-out family's planted hands (the pooled
hand model and that family's own hand model) are dropped so nothing leaks.

First result, same features with the family in and out of training:
    directed transfer 0.958 -> 0.113, soft play 0.916 -> 0.031, isolation 0.960 -> 0.073
while the single feature `ps_top5` (both players' surprise over the pair's five
most improbable hands) reaches 0.833 / 0.503 / 0.791 without any training.

The second part scores mixtures: positives are the two known families plus the
held-out family subsampled to a share s of all positives (s = 0, 0.1, 0.2), which
is what the evaluation set looks like if `other_coordination` makes up s. Each
risk variant is a rank blend of the pair model with a family-agnostic score.

Result. Additive rank blends cost the known families more than they rescue.
Working in top-tail depth, -log10(1 - rank), and keeping
max(model depth, ps_top5 depth - c) is nearly free:
    plain model (unk_w 0.05), s = 0 / 0.05 / 0.1 / 0.2
        no guard   0.9768 / 0.9438 / 0.9113 / 0.8460
        c = 0.5    0.9768 / 0.9559 / 0.9332 / 0.8891
        c = 0.25   0.9762 / 0.9628 / 0.9469 / 0.9151
    production setup (pseudo-labels + family blend, all families trained):
        cleaned AP 0.98123 -> 0.98114 at c = 0.5, 0.97509 at c = 0.25
Unlabelled weight 0.05 beats 0.3 at every share (+0.005 at s = 0).
`--production` runs the production check; c = 0.5 is in 07_submit.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C
from pokercol.cv import cleaned_mask, population_average_precision as ap, table_folds

import importlib
rt = importlib.import_module("14_pair_retest")

SHORT = {"directed_transfer": "hsdir_", "soft_play": "hssoft_", "coordinated_isolation": "hsiso_"}
GENERIC = ["ps_top5", "ps_top3", "pct_ps_top5", "ax_pair_sum"]
SHARES = (0.0, 0.05, 0.1, 0.2)
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)


def rank(v: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(v, kind="stable"), kind="stable") / v.size


def mixture_ap(score, y, fam, keep, held, share, rng_seed=0, drop_held=False) -> float:
    """AP on a population where `share` of the positives come from an unseen family.

    The held-out family's positives that are not drawn into the mixture stay in
    the rows as negatives. Dropping them instead - which this did until
    2026-09-18 - makes any working recovery rule look free: a rule that finds the
    hidden family promotes precisely those rows, and if they are not scored, the
    promotion costs nothing. Stage 108 measured it: 34 or 35 of the 35 pairs a
    calibrated detector promoted were held-out positives, so its cost read
    -0.0012 with them dropped and -0.048 to -0.065 with them kept. Three guards
    were certified as free at share 0 by the old behaviour and then lost on the
    leaderboard. `drop_held=True` reproduces it for comparison only.
    """
    known_pos = keep & (y == 1) & (fam != held)
    held_idx = np.flatnonzero(keep & (y == 1) & (fam == held))
    n_held = int(round(share * known_pos.sum() / (1 - share))) if share else 0
    vals = []
    for r in range(5):
        pick = np.random.default_rng(rng_seed + r).choice(held_idx, min(n_held, held_idx.size), replace=False)
        target = known_pos.copy()
        target[pick] = True
        # The held-out positives that were not drawn are the rows at issue: they
        # stay in and score as negatives unless the caller asks for the old
        # behaviour, which took them out.
        rows = keep & ~(drop_held & (y == 1) & (fam == held) & ~target)
        vals.append(ap(target[rows], score[rows]))
        if not n_held:
            break
    return float(np.mean(vals))


def main() -> None:
    dev = rt.load()
    x_all, feats = rt.matrix(dev)
    y = dev["y"].to_numpy()
    fam = dev["behavior_family"].fill_null("none").to_numpy()
    keep = cleaned_mask(dev)
    folds = table_folds(dev["table_id"].to_numpy(), 5)
    gi = [feats.index(f) for f in GENERIC]
    generic_top5 = rank(np.nan_to_num(x_all[:, gi[0]], nan=-1e9))

    n = len(y)
    tail = lambda r: -np.log10(1.0 - r + 1.0 / n)   # rank -> top-tail depth, 0 .. log10(n)
    cache = C.ARTIFACTS / "leave_family_out_oof.npz"
    stored = dict(np.load(cache)) if cache.exists() else {}
    results = {"single_family": {}, "mixture": {}}
    for held in C.FAMILIES:
        cols = [i for i, f in enumerate(feats)
                if not f.startswith("hs_") and not f.startswith(SHORT[held])]
        x = x_all[:, cols]
        y_lofo = np.where((y == 1) & (fam == held), -1, y)
        held_mask = keep & ((y != 1) | (fam == held))
        held_truth = ((fam == held) & (y == 1))[held_mask]
        models = {}
        for name, unk in (("model unk_w=0.3", 0.3), ("model unk_w=0.05", 0.05)):
            key = f"{held}|{unk}"
            if key not in stored:
                stored[key] = rt.lgb_oof(x, y_lofo, folds, unk_w=unk)
                np.savez(cache, **stored)
            models[name] = stored[key]
        for name, oof in models.items():
            combos = {name: rank(oof)}
            for w in WEIGHTS[1:]:
                combos[f"{name} + {w} x ps_top5"] = rank(oof) + w * generic_top5
                combos[f"{name} + tail {w} x ps_top5"] = tail(rank(oof)) + w * tail(generic_top5)
            for c in (0.0, 0.25, 0.5, 0.75, 1.0):
                combos[f"{name} max(tail, ps_top5 tail - {c})"] = np.maximum(
                    tail(rank(oof)), tail(generic_top5) - c)
            for key, score in combos.items():
                r = results["mixture"].setdefault(key, {})
                for s in SHARES:
                    r.setdefault(str(s), []).append(mixture_ap(score, y, fam, keep, held, s))
                results["single_family"].setdefault(key, {})[held] = ap(held_truth, score[held_mask])
            print(f"{held}: {name} held-out AP {ap(held_truth, oof[held_mask]):.4f}", flush=True)

    summary = {k: {s: float(np.mean(v)) for s, v in r.items()} for k, r in results["mixture"].items()}
    results["mixture_mean"] = summary
    for k, r in summary.items():
        sf = results["single_family"][k]
        print(f"{k:52s} " + "  ".join(f"s={s}: {v:.4f}" for s, v in r.items())
              + "  | held-out " + " ".join(f"{v:.3f}" for v in sf.values()))
    (C.ARTIFACTS / "leave_family_out.json").write_text(json.dumps(results, indent=2))


def production_guard() -> None:
    """Cost of the guard where it matters today: the production pair setup, all
    three families in training, cleaned AP overall and per family, two seeds."""
    p13 = importlib.import_module("13_pseudo_labels")
    dev = rt.load()
    x, feats = rt.matrix(dev)
    y = dev["y"].to_numpy()
    fam = dev["behavior_family"].to_numpy()
    keep = cleaned_mask(dev)
    folds = table_folds(dev["table_id"].to_numpy(), 5)
    n = len(y)
    tail = lambda r: -np.log10(1.0 - r + 1.0 / n)
    g = tail(rank(np.nan_to_num(x[:, feats.index("ps_top5")], nan=-1e9)))
    pseudo = {k: (y == -1) & (np.nan_to_num(p13.nested_scores(x, y, folds, k), nan=0.0) > 0.7)
              for k in range(5)}
    import lightgbm as lgb
    rows: dict[str, list] = {}
    unk_values = [float(v) for v in sys.argv[sys.argv.index("--unk") + 1].split(",")] if "--unk" in sys.argv else [0.3]
    for unk, seed in [(u, sd) for u in unk_values for sd in (C.SEED, C.SEED + 1)]:
        params = {**rt.BASE, "seed": seed}
        glob = np.zeros(n)
        family_scores = {f: np.zeros(n) for f in C.FAMILIES}
        for k in range(5):
            tr = folds != k
            target = (y == 1).astype(np.int8)
            weight = np.where(y == 1, 10.0, np.where(y == 0, 1.0, unk)).astype(float)
            target[tr & pseudo[k]] = 1
            weight[tr & pseudo[k]] = 3.0
            glob[~tr] = lgb.train(params, lgb.Dataset(x[tr], label=target[tr], weight=weight[tr]),
                                  700).predict(x[~tr])
            for f in C.FAMILIES:
                tf = (fam == f).astype(np.int8)
                wf = np.where(fam == f, 10.0, np.where(y == 1, 0.0, np.where(y == 0, 1.0, unk)))
                uf = tr & (np.where(pseudo[k], 0.0, wf) > 0)
                family_scores[f][~tr] = lgb.train(params, lgb.Dataset(x[uf], label=tf[uf], weight=wf[uf]),
                                                  700).predict(x[~tr])
        blend = 0.7 * rank(glob) + 0.3 * np.max(np.stack([rank(v) for v in family_scores.values()]), axis=0)
        tag = "" if unk_values == [0.3] else f"unk={unk} "
        variants = {f"{tag}production": blend}
        for c in ((0.0, 0.25, 0.5, 0.75, 1.0) if not tag else (0.5,)):
            variants[f"{tag}guard c={c}"] = np.maximum(tail(rank(blend)), g - c)
        for name, s in variants.items():
            r = {"cleaned": ap((y == 1)[keep], s[keep])}
            for f in C.FAMILIES:
                m = keep & ((fam == f) | (y != 1))
                r[f] = ap((fam == f)[m], s[m])
            rows.setdefault(name, []).append(r)
    out = {}
    for name, rs in rows.items():
        out[name] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        out[name]["seeds"] = [round(r["cleaned"], 4) for r in rs]
        print(f"production {name:22s} {out[name]}", flush=True)
    (C.ARTIFACTS / ("leave_family_out_production.json" if unk_values == [0.3] else "unknown_weight_production.json")).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    if "--production" in sys.argv:
        production_guard()
    else:
        main()
