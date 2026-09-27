"""Stage 533: the action head inside B's v55 reranker, against baseline and against its own placebo.

Stage 532's columns are fed to B's `05b_evidence_reranker.py` (the v55-era copy behind the
0.91748 file, patched only to accept a per-outer-fold block and a strict-holdout protocol;
stage 530 asserts the patched copy reproduces 0.66085 with no hooks set) in the slot the
`OAH_COLS`/`OCH_COLS` blocks occupy: appended to `FEATS`, same lambdarank, same six seeds,
same truncation, same tie-break. Three arms and two protocols:

    arms        baseline   the v55 evidence side, 95 columns
                mil        95 + the eight action-head columns of note 001 §6.2
                placebo    95 + the same eight columns, each pair holding some other pair's whole
                           block resized to its own hands, permutation drawn per head seed
    protocols   full       five folds, all 372 pairs, nested cross-fit blocks
                holdout    fold 4 (B's own run_milhold.sh fold) in no training set of the head,
                           the standardiser, the epoch choice or the reranker's inner loop;
                           scored once at the end; inner (folds 0-3) and holdout side by side

Every reading is the arm minus its own placebo, paired on whole-table resamples (2000), seed by
seed and as the mean over seeds; the raw arm minus baseline is reported next to it because
stage 401 showed that on a lambdarank engine extra columns manufacture gains on their own.
The noise floor of stage 400 (0.0005, exact ties in the top six) applies throughout.

The bar, set before any run: +0.010 MAP real minus placebo in the full protocol, with the
strict holdout reading the same sign and size. If the honest arm comes back inside its placebo
the stage stops and says so; no variant is tried.

    TARIK_INTERIM=<repro>/data/interim TARIK_RUN=<repro> python scripts/533_b_mil_evidence_arm.py

Writes artifacts/533_b_mil_evidence_arm.json and the per-arm OOF dumps under the reproduction
tree. Nothing here submits anything.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C

INTERIM = Path(os.environ.get("TARIK_INTERIM", ""))
RUN = Path(os.environ.get("TARIK_RUN", INTERIM.parent.parent if INTERIM.name == "interim" else ""))
# REPO NOTE: the evidence chain sat at the root of its own tree during the
# competition, so the harness was "src/05b_mil.py" relative to TARIK_RUN. Here
# TARIK_RUN is the repository root; TARIK_HARNESS overrides the path.
HARNESS = os.environ.get("TARIK_HARNESS", "pipeline/scripts/05b_mil.py")
BLOCKS = INTERIM / "533_blocks"
REPORT = C.ARTIFACTS / "533_b_mil_evidence_arm.json"
N_FOLDS = 5
HOLDOUT_FOLD = 4
N_BOOT = 2000
NOISE_FLOOR = 0.0005
BAR = 0.010
MIL_COLS = ["mil_lse_z", "mil_max_z", "mil_z_gap", "mil_top_street", "mil_top_action",
            "mil_top_resp_partner", "mil_top_surprise", "pct_mil_max_z"]
BASE_ENV = {"USE_VAL": "0", "USE_CT": "0", "USE_CTR": "0", "USE_OAH": "1", "USE_OCH": "1",
            "RERANK_SEEDS": "42,7,2024,11,99,5", "RERANK_CV_ONLY": "1",
            "RERANK_IN": "submissions/tarik_v55_best.csv", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
# REPO NOTE: RERANK_IN was "submission_repro_base.csv", a file of the competition tree that no
# stage here writes. The harness takes only risk and behaviour from it, and stage 534 replaces
# both with v55's text, so v55 itself - present with or without the base phase - is the input.
PROTOCOLS = {"full": ("532_mil_full_nested_s{seed}.parquet", -1),
             "holdout": ("532_mil_inner_s{seed}.parquet+532_mil_holdout_s{seed}.parquet", HOLDOUT_FOLD)}


# ---------------------------------------------------------------- the blocks

def to_pq(df: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    df = (df.join(players.rename({"player_id": "player_1", "pidx": "i1"}), on="player_1")
            .join(players.rename({"player_id": "player_2", "pidx": "i2"}), on="player_2"))
    return df.with_columns(pl.min_horizontal("i1", "i2").alias("p"),
                           pl.max_horizontal("i1", "i2").alias("q")).drop("i1", "i2")


def keyed_block(pattern: str, seed: int, hands: pl.DataFrame, lab: pl.DataFrame) -> pl.DataFrame:
    """Stage 532's (pair_id, hand_id) block as (hidx, p, q, outer_fold, eight columns)."""
    parts = [pl.read_parquet(C.ARTIFACTS / p.format(seed=seed)) for p in pattern.split("+")]
    b = pl.concat(parts)
    b = b.join(hands, on="hand_id", how="left").join(lab.select("pair_id", "p", "q"), on="pair_id", how="left")
    assert b["hidx"].null_count() == 0 and b["p"].null_count() == 0, "block keys did not map"
    out = b.select(["hidx", "p", "q", "pair_id", "outer_fold"] + MIL_COLS).sort("outer_fold", "pair_id", "hidx")
    for k in out["outer_fold"].unique().to_list():
        sub = out.filter(pl.col("outer_fold") == k)
        assert sub.select("hidx", "p", "q").n_unique() == sub.height, f"duplicate keys in outer fold {k}"
    return out


def placebo(block: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Stage 401's matched placebo: each pair keeps some other pair's whole block, resized to its
    own number of hands; one permutation for every outer fold, drawn from the head seed."""
    outs = []
    for k in sorted(block["outer_fold"].unique().to_list()):
        sub = block.filter(pl.col("outer_fold") == k).sort("pair_id", "hidx")
        lens = sub.group_by("pair_id", maintain_order=True).len()["len"].to_numpy().astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        # drawn afresh from the same seed per outer fold: identical wherever the pair sets are
        # identical (the full protocol, and outer folds 0-3 of the strict one); the outer-fold-4
        # block of the strict protocol covers all 372 pairs and gets its own draw
        perm = np.random.default_rng(seed).permutation(len(lens))
        take = np.concatenate([np.resize(np.arange(starts[p], starts[p] + lens[p]), lens[i])
                               for i, p in enumerate(perm)]).astype(np.int64)
        vals = sub.select(MIL_COLS)[take]
        outs.append(sub.select("hidx", "p", "q", "pair_id", "outer_fold").with_columns(vals))
    out = pl.concat(outs)
    k0 = out["outer_fold"].min()
    a = out.filter(pl.col("outer_fold") == k0)[MIL_COLS[0]].to_numpy()
    b = block.filter(pl.col("outer_fold") == k0).sort("pair_id", "hidx")[MIL_COLS[0]].to_numpy()
    assert not np.allclose(a, b), "the shuffle did nothing"
    return out


def seed_bag(blocks: list[pl.DataFrame]) -> pl.DataFrame:
    """The mean of the seeds' columns, key by key; the within-pair percentile is recomputed."""
    keys = ["hidx", "p", "q", "pair_id", "outer_fold"]
    acc = blocks[0].select(keys + MIL_COLS).sort(keys)
    for b in blocks[1:]:
        j = acc.join(b.select(keys + MIL_COLS), on=keys, how="inner", suffix="_o")
        assert j.height == acc.height, "seed blocks cover different keys"
        acc = j.with_columns([(pl.col(c) + pl.col(c + "_o")).alias(c) for c in MIL_COLS]).select(keys + MIL_COLS)
    acc = acc.with_columns([(pl.col(c) / len(blocks)).alias(c) for c in MIL_COLS])
    # descriptors of the argmax action are categorical; the mean over seeds is kept as a soft vote
    return acc.with_columns(
        (pl.col("mil_max_z").rank("average").over(["outer_fold", "pair_id"])
         / pl.len().over(["outer_fold", "pair_id"])).cast(pl.Float32).alias("pct_mil_max_z"))


# ---------------------------------------------------------------- the engine

def run_harness(tag: str, block: Path | None, hold: int, log_dir: Path) -> dict:
    env = {**os.environ, **BASE_ENV, "RERANK_DUMP": f"533_{tag}"}
    if block is not None:
        env["MIL_BLOCK"] = str(block)
    if hold >= 0:
        env["HOLDOUT_FOLD"] = str(hold)
    with open(log_dir / f"05b_533_{tag}.log", "w") as out, open(log_dir / f"05b_533_{tag}.err", "w") as err:
        subprocess.run([sys.executable, HARNESS], cwd=RUN, env=env, stdout=out, stderr=err, check=True)
    own = INTERIM / f"reranker_cv_533_{tag}.json"
    cv = json.loads((own if own.exists() else INTERIM / "reranker_cv.json").read_text())
    if block is not None:
        assert cv["mil_cols"] == MIL_COLS, cv["mil_cols"]
    return cv


# ---------------------------------------------------------------- the metrics

def ap5(picks: list, gold: set) -> float:
    seen, hits, s = set(), 0, 0.0
    for rank, h in enumerate(picks[:5], start=1):
        if h not in seen and h in gold:
            hits += 1
            s += hits / rank
        seen.add(h)
    return s / min(len(gold), 5) if gold else 0.0


def per_pair_ap(dump: pl.DataFrame, gold: dict, pairs: list[str]) -> np.ndarray:
    """AP@5 per pair with B's tie-break (s2 desc, hidx asc)."""
    top = (dump.sort(["pair_id", "s2", "hidx"], descending=[False, True, False])
           .group_by("pair_id", maintain_order=True).agg(pl.col("hidx").head(5).alias("picks")))
    picks = dict(top.iter_rows())
    return np.array([ap5(picks[p], gold[p]) for p in pairs])


def paired_bootstrap(a: np.ndarray, b: np.ndarray, table_of: np.ndarray, seed: int = 0) -> dict:
    """mean(a) - mean(b) over pairs, resampling whole tables."""
    uniq, inv = np.unique(table_of, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(uniq.size + 1))
    rng = np.random.default_rng(seed)
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([order[starts[t]:starts[t + 1]] for t in pick])
        d[i] = a[idx].mean() - b[idx].mean()
    point = float(a.mean() - b.mean())
    return {"delta": round(point, 5), "ci95": [round(float(np.percentile(d, 2.5)), 5), round(float(np.percentile(d, 97.5)), 5)],
            "p_improves": round(float((d > 0).mean()), 4), "clears the 0.0005 tie-break floor": bool(abs(point) > NOISE_FLOOR)}


def per_rank(dump: pl.DataFrame, ev: pl.DataFrame, pairs: list[str]) -> dict:
    top5 = (dump.filter(pl.col("pair_id").is_in(pairs))
            .sort(["pair_id", "s2", "hidx"], descending=[False, True, False])
            .group_by("pair_id", maintain_order=True).head(5)
            .with_columns((pl.int_range(pl.len()).over("pair_id") + 1).alias("slot"), pl.lit(1).alias("hit")))
    r = ev.filter(pl.col("pair_id").is_in(pairs)).join(top5.select("pair_id", "hidx", "hit"),
                                                        on=["pair_id", "hidx"], how="left").with_columns(pl.col("hit").fill_null(0))
    by_rank = {str(int(k)): round(float(v), 4) for k, v in
               r.group_by("evidence_rank").agg(pl.col("hit").mean()).sort("evidence_rank").iter_rows()}
    g = top5.join(ev.select("pair_id", "hidx").with_columns(pl.lit(1).alias("g")), on=["pair_id", "hidx"], how="left").with_columns(pl.col("g").fill_null(0))
    by_slot = {str(int(k)): round(float(v), 4) for k, v in g.group_by("slot").agg(pl.col("g").mean()).sort("slot").iter_rows()}
    return {"by organiser rank": by_rank, "at our slot k": by_slot,
            "mean gold hands in the top five": round(float(g.group_by("pair_id").agg(pl.col("g").sum())["g"].mean()), 4)}


def reading(dump: pl.DataFrame, gold: dict, ev: pl.DataFrame, fam_of: dict, pairs: list[str]) -> tuple[dict, np.ndarray]:
    ap = per_pair_ap(dump, gold, pairs)
    fam = np.array([fam_of[p] for p in pairs])
    out = {"pairs": len(pairs), "all": round(float(ap.mean()), 5),
           **{f[:4]: round(float(ap[fam == f].mean()), 5) for f in C.FAMILIES},
           "per-rank recall": per_rank(dump, ev, pairs)}
    return out, ap


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocols", default="full,holdout")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--arms", default="baseline,mil,placebo,bag")
    args = parser.parse_args()
    t0 = time.time()
    if not INTERIM.exists() or not (RUN / HARNESS).exists():
        raise SystemExit("set TARIK_INTERIM and TARIK_RUN to the reproduction tree")
    BLOCKS.mkdir(exist_ok=True)
    log_dir = RUN / "logs"
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    seeds = [C.SEED + int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")

    players = pl.read_parquet(INTERIM / "players.parquet").select("player_id", "pidx")
    hands = pl.read_parquet(INTERIM / "hands.parquet").select("hidx", "hand_id", "tidx")
    lab = to_pq(pl.read_csv(C.DEV_LABELS).filter(pl.col("label") == 1), players).select("pair_id", "p", "q", "behavior_family")
    fam_of = dict(lab.select("pair_id", "behavior_family").iter_rows())
    ev = (pl.read_csv(C.DEV_EVIDENCE).join(hands.select("hidx", "hand_id"), on="hand_id")
          .select("pair_id", "hidx", "evidence_rank"))
    gold = {p: set(h) for p, h in ev.group_by("pair_id").agg(pl.col("hidx")).iter_rows()}

    for proto in args.protocols.split(","):
        pattern, hold = PROTOCOLS[proto]
        entry = report.setdefault(proto, {"holdout fold": hold, "bar": BAR, "noise floor": NOISE_FLOOR})
        dumps: dict[str, pl.DataFrame] = {}
        blocks: dict[int, pl.DataFrame] = {}

        def arm(tag: str, block: pl.DataFrame | None) -> None:
            path = None
            if block is not None:
                path = BLOCKS / f"{proto}_{tag}.parquet"
                block.drop("pair_id").write_parquet(path)
            cv = run_harness(f"{proto}_{tag}", path, hold, log_dir)
            d = pl.read_parquet(INTERIM / f"rerank_oof_533_{proto}_{tag}.parquet")
            dumps[tag] = d
            entry.setdefault("harness cv", {})[tag] = {k: v for k, v in cv.items() if k not in ("mil_cols", "per_family")}
            print(f"[{proto}] {tag}: harness reranker_map5_oof {cv['reranker_map5_oof']:.5f}"
                  + (f"  inner {cv['inner_map5']:.5f} holdout {cv['holdout_map5']:.5f}" if hold >= 0 else "")
                  + f" ({time.time() - t0:.0f}s)", flush=True)

        if "baseline" in arms:
            arm("baseline", None)
        for seed in seeds:
            if not (C.ARTIFACTS / pattern.split("+")[0].format(seed=seed)).exists():
                print(f"[{proto}] no 532 block for seed {seed}, skipping", flush=True)
                continue
            blocks[seed] = keyed_block(pattern, seed, hands.select("hidx", "hand_id"), lab)
            if "mil" in arms:
                arm(f"mil_s{seed}", blocks[seed])
            if "placebo" in arms:
                arm(f"placebo_s{seed}", placebo(blocks[seed], seed))
        if "bag" in arms and len(blocks) > 1:
            bag = seed_bag(list(blocks.values()))
            arm("mil_bag", bag)
            arm("placebo_bag", placebo(bag, C.SEED + 100))
        REPORT.write_text(json.dumps(report, indent=2))
        # arms run by an earlier invocation of this stage are read back from their dumps
        for path in sorted(INTERIM.glob(f"rerank_oof_533_{proto}_*.parquet")):
            tag = path.stem.replace(f"rerank_oof_533_{proto}_", "")
            if tag not in dumps:
                dumps[tag] = pl.read_parquet(path)

        # ---- readings on one pair order, one table map, one bootstrap draw set
        slices = {"all": None} if hold < 0 else {"inner (folds 0-3)": [f for f in range(N_FOLDS) if f != hold],
                                                  "holdout (fold 4)": [hold]}
        any_dump = next(iter(dumps.values()))
        pair_fold = dict(any_dump.select("pair_id", "fold").unique().iter_rows())
        pair_table = dict(any_dump.select("pair_id", "tidx").unique().iter_rows())
        for sname, folds in slices.items():
            pairs = sorted(p for p in gold if folds is None or pair_fold[p] in folds)
            table_of = np.array([pair_table[p] for p in pairs])
            res = entry.setdefault("readings", {}).setdefault(sname, {"pairs": len(pairs)})
            aps: dict[str, np.ndarray] = {}
            for tag, d in dumps.items():
                res[tag], aps[tag] = reading(d, gold, ev, fam_of, pairs)
            # arm minus its own placebo, seed by seed, and the raw arm minus baseline next to it
            contrasts = {}
            for seed in seeds:
                a, b = f"mil_s{seed}", f"placebo_s{seed}"
                if a in aps and b in aps:
                    c = {"mil minus placebo": paired_bootstrap(aps[a], aps[b], table_of, seed)}
                    if "baseline" in aps:
                        c["mil minus baseline"] = paired_bootstrap(aps[a], aps["baseline"], table_of, seed)
                        c["placebo minus baseline"] = paired_bootstrap(aps[b], aps["baseline"], table_of, seed)
                    fam = np.array([fam_of[p] for p in pairs])
                    c["mil minus placebo by family"] = {f[:4]: round(float(aps[a][fam == f].mean() - aps[b][fam == f].mean()), 5) for f in C.FAMILIES}
                    contrasts[f"seed {seed}"] = c
            mil_ok = [s for s in seeds if f"mil_s{s}" in aps and f"placebo_s{s}" in aps]
            if len(mil_ok) > 1:
                ma = np.mean([aps[f"mil_s{s}"] for s in mil_ok], axis=0)
                pa = np.mean([aps[f"placebo_s{s}"] for s in mil_ok], axis=0)
                contrasts["mean over seeds"] = {
                    "mil minus placebo": paired_bootstrap(ma, pa, table_of, 7),
                    "mil minus baseline": paired_bootstrap(ma, aps["baseline"], table_of, 7) if "baseline" in aps else None,
                    "placebo minus baseline": paired_bootstrap(pa, aps["baseline"], table_of, 7) if "baseline" in aps else None,
                    "seeds": mil_ok}
            if "mil_bag" in aps and "placebo_bag" in aps:
                contrasts["seed-bagged columns"] = {
                    "mil_bag minus placebo_bag": paired_bootstrap(aps["mil_bag"], aps["placebo_bag"], table_of, 11),
                    "mil_bag minus baseline": paired_bootstrap(aps["mil_bag"], aps["baseline"], table_of, 11) if "baseline" in aps else None}
            res["contrasts"] = contrasts
            for tag in ("mil", "placebo"):
                got = [res[f"{tag}_s{s}"]["all"] for s in seeds if f"{tag}_s{s}" in res]
                if len(got) > 1:
                    res[f"{tag} seed dispersion"] = {"map5 per seed": got, "mean": round(float(np.mean(got)), 5),
                                                     "sd": round(float(np.std(got, ddof=1)), 5),
                                                     "range": round(float(max(got) - min(got)), 5)}
            print(f"[{proto}] {sname}: " + json.dumps({k: (v["all"] if isinstance(v, dict) and "all" in v else None)
                                                       for k, v in res.items() if isinstance(v, dict) and "all" in v}), flush=True)
            if "mean over seeds" in contrasts:
                print(f"[{proto}] {sname}: mean over seeds mil - placebo {contrasts['mean over seeds']['mil minus placebo']}", flush=True)
        REPORT.write_text(json.dumps(report, indent=2))

    # ---- the bar
    verdict = {"bar": f"+{BAR} MAP real minus placebo (full protocol, mean over seeds), strict holdout agreeing in sign and size"}
    try:
        full = report["full"]["readings"]["all"]["contrasts"]["mean over seeds"]["mil minus placebo"]
        inner = report["holdout"]["readings"]["inner (folds 0-3)"]["contrasts"]["mean over seeds"]["mil minus placebo"]
        hold = report["holdout"]["readings"]["holdout (fold 4)"]["contrasts"]["mean over seeds"]["mil minus placebo"]
        verdict.update({"full: mil minus placebo": full, "inner: mil minus placebo": inner, "holdout: mil minus placebo": hold,
                        "cleared": bool(full["delta"] >= BAR and hold["delta"] > 0 and full["ci95"][0] > 0)})
    except KeyError as e:
        verdict["incomplete"] = f"missing {e}"
    report["verdict"] = verdict
    report["runtime_s"] = round(time.time() - t0, 1)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"verdict: {json.dumps(verdict, indent=1)}\nwritten to {REPORT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
