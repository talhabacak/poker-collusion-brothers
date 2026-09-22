"""Stage 34 (experiment): route pairs to `other_coordination`?

docs/OTHER_COORDINATION_DETECTION.md: other = strong coordination that the three
known families do not explain. Behaviour MAP is a macro AP over the three
disclosed families where a pair only counts for the family it was routed to.
An unseen-family pair routed to, say, soft play at high risk is a false
positive at the top of the soft-play ranking; routing it to
`other_coordination` removes it. Routing a real soft-play pair there loses it.

Leave-one-family-out, reusing stage 27's out-of-fold pair scores:
    risk       pair model without family F, guarded as in production
    family     behaviour model trained on the two known families only (OOF)
    OOD score  candidates below, family-agnostic by construction

The OOD scorer is also *learned*: for each held-out F a LightGBM is trained on
the other two leave-one-out configurations (their held-out family = OOD
positive, their known-family positives = OOD negative) and applied to F -
so the threshold and the model never see F.

Scored on mixtures where F makes up a share s of positives: behaviour MAP over
the two known families, and the precision of pairs routed to other among the
risk top-k.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C
from pokercol.cv import cleaned_mask, table_folds
from pokercol.metrics import average_precision

import importlib
rt = importlib.import_module("14_pair_retest")
m27 = importlib.import_module("27_leave_family_out")

TOP_K = 800          # routing decisions only matter near the top of the risk ranking
SHARES = (0.0, 0.1, 0.2)


def known_behaviour(x, y, fam, folds, known):
    """OOF probabilities over the known families, trained on their positives."""
    lookup = {f: i for i, f in enumerate(known)}
    proba = np.zeros((len(y), len(known)))
    params = dict(**C.LGB_REPRO, objective="multiclass", num_class=len(known), learning_rate=0.05,
                  num_leaves=15, min_data_in_leaf=10, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=3.0, verbosity=-1, num_threads=C.N_THREADS, seed=C.SEED)
    for k in range(5):
        tr = (folds != k) & (y == 1) & np.isin(fam, known)
        b = lgb.train(params, lgb.Dataset(x[tr], label=np.array([lookup[f] for f in fam[tr]])), 300)
        proba[folds == k] = b.predict(x[folds == k])
    return proba


def behaviour_map_mixture(risk, routed, fam, y, keep, held, known, share, seed=0):
    base = keep & ((y != 1) | (fam != held))
    held_idx = np.flatnonzero(keep & (y == 1) & (fam == held))
    n_known = int((base & (y == 1)).sum())
    vals = []
    for r in range(5 if share else 1):
        m = base.copy()
        if share:
            n = int(round(share * n_known / (1 - share)))
            m[np.random.default_rng(seed + r).choice(held_idx, min(n, held_idx.size), replace=False)] = True
        truth = np.where((y == 1)[m], fam[m], "none")
        aps = [average_precision((truth == f).astype(np.int8), np.where(routed[m] == f, risk[m], 0.0)) for f in known]
        vals.append(float(np.mean(aps)))
    return float(np.mean(vals))


def main() -> None:
    dev = rt.load()
    x, feats = rt.matrix(dev)
    y = dev["y"].to_numpy()
    fam = dev["behavior_family"].fill_null("none").to_numpy()
    keep = cleaned_mask(dev)
    folds = table_folds(dev["table_id"].to_numpy(), 5)
    n = len(y)
    tail = lambda r: -np.log10(1.0 - r + 1.0 / n)
    g = tail(m27.rank(np.nan_to_num(x[:, feats.index("ps_top5")], nan=-1e9)))
    stored = dict(np.load(C.ARTIFACTS / "leave_family_out_oof.npz"))

    configs = {}
    for held in C.FAMILIES:
        known = [f for f in C.FAMILIES if f != held]
        cols = [i for i, f in enumerate(feats) if not f.startswith("hs_") and not f.startswith(m27.SHORT[held])]
        risk = np.maximum(tail(m27.rank(stored[f"{held}|0.05"])), g - 0.5)
        proba = known_behaviour(x[:, cols], y, fam, folds, known)
        pmax = proba.max(axis=1)
        ent = -(proba * np.log(proba + 1e-9)).sum(axis=1)
        top = np.zeros(n, bool)
        top[np.argsort(-risk)[:TOP_K]] = True
        # Family-agnostic OOD features; no family identity enters.
        ood_x = np.stack([risk, g, pmax, ent, g - risk, m27.rank(risk), m27.rank(g)], axis=1)
        configs[held] = dict(known=known, risk=risk, proba=proba, pmax=pmax, ent=ent, top=top, ood_x=ood_x)

    results = {}
    for held, c in configs.items():
        known, risk, top = c["known"], c["risk"], c["top"]
        argmax = np.array(known)[c["proba"].argmax(axis=1)]
        is_held = (y == 1) & (fam == held)
        is_known = (y == 1) & np.isin(fam, known)
        # Learned OOD scorer from the other two configurations.
        others = [h for h in C.FAMILIES if h != held]
        xo = np.vstack([configs[h]["ood_x"][configs[h]["top"] & ((y == 1))] for h in others])
        yo = np.concatenate([((fam == h) & (y == 1))[configs[h]["top"] & (y == 1)] for h in others]).astype(int)
        booster = lgb.train(dict(**C.LGB_REPRO, objective="binary", learning_rate=0.05, num_leaves=7,
                                 min_data_in_leaf=20, verbosity=-1, num_threads=C.N_THREADS, seed=C.SEED),
                            lgb.Dataset(xo, label=yo), 200)
        learned = booster.predict(c["ood_x"])
        scorers = {
            "1 - family max": 1.0 - c["pmax"],
            "risk x (1 - family max) x (1 + 0.2 H)": m27.rank(risk) * (1 - c["pmax"]) * (1 + 0.2 * c["ent"]),
            "learned OOD (other configs)": learned,
        }
        row = {"no other routing": {str(s): behaviour_map_mixture(risk, argmax, fam, y, keep, held, known, s)
                                    for s in SHARES}}
        for name, score in scorers.items():
            for q in (0.9, 0.95, 0.98):
                thr = np.quantile(score[top], q)
                routed = np.where(top & (score >= thr), "other_coordination", argmax)
                sel = top & (score >= thr)
                prec_held = float(is_held[sel].mean()) if sel.any() else 0.0
                prec_nonknown = float((~is_known)[sel].mean()) if sel.any() else 0.0
                row[f"{name} top {int(round((1 - q) * 100))}%"] = {
                    **{str(s): behaviour_map_mixture(risk, routed, fam, y, keep, held, known, s) for s in SHARES},
                    "routed": int(sel.sum()), "share_heldout_family": round(prec_held, 3),
                    "share_not_known_positive": round(prec_nonknown, 3)}
        results[held] = row
        print(f"\nheld out {held}")
        for k, v in row.items():
            print(f"  {k:52s} " + "  ".join(f"{a}: {b:.4f}" if isinstance(b, float) and a[0].isdigit() else f"{a}: {b}"
                                              for a, b in v.items()))
    (C.ARTIFACTS / "other_routing.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
