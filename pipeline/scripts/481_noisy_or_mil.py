"""Stage 481: differentiable noisy-OR multiple-instance learning over actions.

One network scores every action a pair took in a hand,

    p_t = sigma(z_t) = P(action t is the collusive trigger)

and the hand's score is the bag aggregate

    P(H is evidence) = 1 - prod_t (1 - p_t)

trained against the hand-level label with gradients reaching every action. No
action is ever given a label, no label is ever replaced by a model output, and
nothing is relabelled between rounds. That is the whole reason this is allowed to
exist after the teammate's experiments 36 and 50: their MIL refined the *labels*
with the previous round's model, which coupled the folds and inflated standard
out-of-fold by +0.070 while the strict holdout read -0.001. There are no rounds
here and there is nothing to couple, but the same check is run anyway and its
result is reported whatever it says (`--part holdout`).

Pooling
-------
Two aggregates are implemented. Noisy-OR is the one the structure asks for; the
log-sum-exp form

    S(H) = tau * log sum_t exp(z_t / tau)

is the smooth-max alternative that trains more stably when a bag is long. The
choice is made on **folds 0-3 only**, by the mean inner-validation MAP@5 of the
bag score used on its own as a within-pair ranking, and then frozen. Fold 4 takes
no part in that decision, which is what keeps `--part holdout` honest.

Bags
----
Built by stage 480 and used exactly as it labels them:

    positive  the 1,817 listed evidence hands of the 372 disclosed target pairs
    negative  the 223,749 - 45,129 hands of the 1,488 confirmed non-target pairs
    censored  a positive pair's other 43,312 hands - **excluded from the loss**

`training_index` is the single place a training row can come from and it filters
on `bag_label >= 0`; the count of censored bags it drops is asserted against the
frame's own census, so the tempting error cannot happen quietly.

Fold protocol
-------------
Table folds, canonical. A pair lives on one table, so a pair, its hands and its
actions all land in one fold; stage 480 asserts it at zero violations.

    full      five-fold out of fold over all 372 pairs, the reading that is
              comparable to the shipped 0.68921
    inner     folds 0-3 only, four-fold, nested epoch choice
    holdout   fold 4 held out of every fit, every statistic and every choice,
              scored once at the end

Epochs are never chosen on the fold being scored. For an outer fold k the three
training folds are rotated (train two, validate one), the epoch with the best
mean inner-validation bag loss is taken, and the model is refitted on all three.

    python scripts/481_noisy_or_mil.py --part pooling    # choose the aggregate
    python scripts/481_noisy_or_mil.py --part columns    # the OOF hand columns
    python scripts/481_noisy_or_mil.py --part holdout    # the strict holdout

Writes artifacts/481_noisy_or_mil.json and artifacts/481_mil_columns_*.parquet.
Nothing here submits anything.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C
from pokercol.cv import table_folds

import importlib
s480 = importlib.import_module("480_action_bag_frame")

N_FOLDS = 5
HOLDOUT_FOLD = 4                       # the fold tarik's run_milhold.sh held out
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORT = C.ARTIFACTS / "481_noisy_or_mil.json"

# Small data, so a narrow network, heavy dropout and a short budget. None of
# these was swept on anything fold 4 can see; the width and dropout are stage
# 133's, which were selected on folds 0-3 of the same split.
DIM = 128
DROPOUT = 0.3
LR = 2e-3
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 40
BATCH_BAGS = 512
NEG_PER_POS = 20          # negative bags sampled per positive bag, per epoch
TAU = 1.0                 # log-sum-exp temperature
GRAD_CLIP = 5.0

MIL_COLUMNS = ["mil_noisy_or", "mil_max_z", "mil_mean_z", "mil_lse_z", "mil_z_gap",
               "mil_top_street", "mil_top_action", "mil_top_resp_partner",
               "mil_top_surprise", "pct_mil_max_z"]


# --------------------------------------------------------------- frame loading

def load_frame() -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """The action frame, its bag index and its fold vector.

    Rows are sorted by bag so a bag is a contiguous slice, which is what the
    segment reductions below assume.
    """
    if not s480.FRAME.exists():
        raise SystemExit(
            f"{s480.FRAME} is missing - it is a derived cache and `run_all.py` clears "
            f"`data/cache/*.parquet`. Rebuild it with:\n"
            f"    python scripts/480_action_bag_frame.py --part frame")
    rows = pl.read_parquet(s480.FRAME).sort(["pair_id", "hand_id", "action_no", "player_id"])
    bag_key = rows.select("pair_id", "hand_id")
    codes = bag_key.with_row_index("row").group_by(["pair_id", "hand_id"], maintain_order=True)
    bags = codes.agg(pl.col("row").min().alias("start"), pl.len().alias("n"))
    bag_idx = np.repeat(np.arange(bags.height), bags["n"].to_numpy())
    folds = table_folds(rows["table_id"].to_numpy(), N_FOLDS)
    return rows, bag_idx, folds


def bag_table(rows: pl.DataFrame, bag_idx: np.ndarray) -> pl.DataFrame:
    """One row per bag: key, label, fold, family, and where its actions start."""
    first = np.concatenate([[0], np.flatnonzero(np.diff(bag_idx)) + 1])
    out = rows[first].select("pair_id", "hand_id", "table_id", "behavior_family",
                             "is_pos_pair", "bag_label")
    return out.with_columns(
        pl.Series("fold", table_folds(out["table_id"].to_numpy(), N_FOLDS)),
        pl.Series("n_actions", np.bincount(bag_idx)))


def make_gather(sizes: np.ndarray, starts: np.ndarray):
    """`batch of bag ids -> the row indices of their actions`, without a Python loop."""
    def gather(batch: np.ndarray) -> np.ndarray:
        n = sizes[batch]
        base = np.repeat(starts[batch], n)
        ends = np.cumsum(n)
        offset = np.arange(ends[-1]) - np.repeat(ends - n, n)
        return base + offset
    return gather


def training_index(bags: pl.DataFrame, fold_in: np.ndarray) -> tuple[np.ndarray, dict]:
    """The only place a training bag can come from.

    Censored bags - a positive pair's hands that the organiser did not list - are
    dropped here and nowhere else, and the count dropped is returned so the
    caller can assert it against the frame census.
    """
    label = bags["bag_label"].to_numpy()
    eligible = fold_in & (label >= 0)
    dropped = int((fold_in & (label == -1)).sum())
    return np.flatnonzero(eligible), {
        "training bags": int(eligible.sum()),
        "positive": int((eligible & (label == 1)).sum()),
        "negative": int((eligible & (label == 0)).sum()),
        "censored bags excluded from the loss": dropped}


# ------------------------------------------------------------------- the model

class ActionHead(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, DIM), nn.LayerNorm(DIM), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(DIM, DIM), nn.LayerNorm(DIM), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(DIM, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def log1mexp(a: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(a)) for a < 0, in the two stable branches."""
    a = torch.clamp(a, max=-1e-7)
    return torch.where(a > -0.6931472,
                       torch.log(-torch.expm1(a)),
                       torch.log1p(-torch.exp(a)))


def pool(z: torch.Tensor, seg: torch.Tensor, n_bags: int, form: str) -> torch.Tensor:
    """Bag logit from action logits.

    noisy_or   log P - log(1 - P) with P = 1 - prod(1 - sigma(z)). The product is
               held in log space: log prod (1 - sigma(z)) = -sum softplus(z).
    lse        tau * logsumexp(z / tau), the smooth maximum.
    """
    if form == "noisy_or":
        s = torch.zeros(n_bags, device=z.device, dtype=z.dtype)
        s.index_add_(0, seg, F.softplus(z))
        log_none = -s                       # log P(no action is the trigger)
        return log1mexp(log_none) - log_none
    if form == "lse":
        m = torch.full((n_bags,), -1e30, device=z.device, dtype=z.dtype)
        m = m.scatter_reduce(0, seg, z, reduce="amax", include_self=True)
        e = torch.zeros(n_bags, device=z.device, dtype=z.dtype)
        e.index_add_(0, seg, torch.exp((z - m[seg]) / TAU))
        return m + TAU * torch.log(e)
    raise ValueError(form)


# ------------------------------------------------------------- standardisation

def standardiser(x: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Winsorising limits and z-score moments, from the training rows alone."""
    sub = x[rows]
    lo = np.nanquantile(sub, 0.001, axis=0)
    hi = np.nanquantile(sub, 0.999, axis=0)
    med = np.nanmedian(sub, axis=0)
    clipped = np.clip(np.where(np.isnan(sub), med, sub), lo, hi)
    mu = clipped.mean(axis=0)
    sd = clipped.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return lo, hi, med, np.stack([mu, sd])


def apply_standardiser(x: np.ndarray, pars) -> np.ndarray:
    lo, hi, med, ms = pars
    out = np.clip(np.where(np.isnan(x), med, x), lo, hi)
    return ((out - ms[0]) / ms[1]).astype(np.float32)


# ---------------------------------------------------------------------- fitting

def fit(x: np.ndarray, bag_idx: np.ndarray, bags: pl.DataFrame, train_bags: np.ndarray,
        form: str, epochs: int, seed: int,
        val_bags: np.ndarray | None = None) -> tuple[ActionHead, object, list[float]]:
    """Fit the action head on `train_bags`, optionally tracking a validation curve.

    `val_bags` is used for nothing but the returned per-epoch loss curve; the
    weights are never selected inside this function.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    sizes = np.bincount(bag_idx)
    starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    rows_of = make_gather(sizes, starts)

    train_rows = rows_of(train_bags)
    pars = standardiser(x, train_rows)
    xs = torch.from_numpy(apply_standardiser(x, pars)).to(DEVICE)
    label = bags["bag_label"].to_numpy()

    pos = train_bags[label[train_bags] == 1]
    neg = train_bags[label[train_bags] == 0]
    assert (label[train_bags] >= 0).all(), "a censored bag reached the training set"

    model = ActionHead(x.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    curve: list[float] = []
    # T_max is the *budget*, not the number of epochs actually run, so a refit
    # stopped at the chosen epoch sees exactly the learning rate the rotation that
    # chose it saw at that epoch. Annealing over the shorter run instead would
    # make the selected epoch describe a model that was never validated.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    for _ in range(epochs):
        model.train()
        take = rng.choice(neg, size=min(neg.size, NEG_PER_POS * pos.size), replace=False)
        epoch_bags = rng.permutation(np.concatenate([pos, take]))
        for i in range(0, epoch_bags.size, BATCH_BAGS):
            batch = epoch_bags[i:i + BATCH_BAGS]
            idx = rows_of(batch)
            seg = torch.from_numpy(np.repeat(np.arange(batch.size), sizes[batch])).to(DEVICE)
            z = model(xs[torch.from_numpy(idx).to(DEVICE)])
            logit = pool(z, seg, batch.size, form)
            y = torch.from_numpy(label[batch].astype(np.float32)).to(DEVICE)
            # Balanced within the batch: the sampled ratio is NEG_PER_POS to one.
            w = torch.where(y > 0, float(NEG_PER_POS), 1.0)
            loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        sched.step()
        if val_bags is not None:
            curve.append(bag_loss(model, xs, sizes, starts, label, val_bags, form))
    return model, pars, curve


@torch.no_grad()
def bag_logits(model: ActionHead, xs: torch.Tensor, sizes, starts, which: np.ndarray,
               form: str, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Bag logits and per-action logits for the bags given."""
    model.eval()
    gather = make_gather(sizes, starts)
    out = np.zeros(which.size, dtype=np.float32)
    per_action: list[np.ndarray] = []
    for i in range(0, which.size, chunk):
        batch = which[i:i + chunk]
        idx = gather(batch)
        seg = torch.from_numpy(np.repeat(np.arange(batch.size), sizes[batch])).to(DEVICE)
        z = model(xs[torch.from_numpy(idx).to(DEVICE)])
        out[i:i + batch.size] = pool(z, seg, batch.size, form).float().cpu().numpy()
        per_action.append(z.float().cpu().numpy())
    return out, np.concatenate(per_action)


@torch.no_grad()
def bag_loss(model, xs, sizes, starts, label, which, form) -> float:
    logits, _ = bag_logits(model, xs, sizes, starts, which, form)
    y = label[which].astype(np.float32)
    w = np.where(y > 0, float(NEG_PER_POS), 1.0)
    t = torch.from_numpy(logits)
    ls = F.binary_cross_entropy_with_logits(t, torch.from_numpy(y), reduction="none").numpy()
    return float((ls * w).sum() / w.sum())


# ---------------------------------------------------- the within-pair proxy metric

def proxy_map5(bags: pl.DataFrame, score: np.ndarray, which: np.ndarray) -> float:
    """MAP@5 of the bag score used on its own to rank a positive pair's hands.

    Only the bags of positive pairs can be ranked, so `which` is intersected with
    them. This is the selection metric for the pooling form and for the epoch
    count, and it is only ever read on folds the model did not train on.
    """
    sub = bags[which].with_columns(pl.Series("s", score))
    sub = sub.filter(pl.col("is_pos_pair") > 0)
    if sub.height == 0:
        return 0.0
    total, n = 0.0, 0
    ordered = sub.sort(["pair_id", "s", "hand_id"], descending=[False, True, False])
    for _, g in ordered.group_by("pair_id", maintain_order=True):
        gold = g["bag_label"].to_numpy() == 1
        rel = int(gold.sum())
        n += 1
        if rel == 0:
            continue
        hits, acc = 0, 0.0
        for i in range(min(5, gold.size)):
            if gold[i]:
                hits += 1
                acc += hits / (i + 1)
        total += acc / min(rel, 5)
    return total / n if n else 0.0


# ------------------------------------------------------------------- protocols

VAL_NEG_CAP = 20_000


def validation_bags(bags: pl.DataFrame, va_mask: np.ndarray, seed: int) -> np.ndarray:
    """Every positive bag of the validation fold plus a capped negative sample.

    The cap is for speed alone; it is drawn once per call with a fixed seed and
    reads nothing but `bag_label`, which the validation fold is allowed to carry -
    it is a fold inside the training half, never the fold being scored.
    """
    va, _ = training_index(bags, va_mask)
    label = bags["bag_label"].to_numpy()[va]
    pos, neg = va[label == 1], va[label == 0]
    if neg.size > VAL_NEG_CAP:
        neg = np.random.default_rng(seed).choice(neg, VAL_NEG_CAP, replace=False)
    return np.sort(np.concatenate([pos, neg]))


def nested_epochs(x, bag_idx, bags, folds_avail: list[int], fold_of_bag, form, seed,
                  report_into: dict | None = None) -> int:
    """Rotate the training folds - train on all but one, validate on it - and take
    the epoch with the best mean validation bag loss. The fold being scored by the
    caller is not in `folds_avail`, so nothing here sees it."""
    curves = []
    for v in folds_avail:
        tr_mask = np.isin(fold_of_bag, [f for f in folds_avail if f != v])
        tr, _ = training_index(bags, tr_mask)
        va = validation_bags(bags, fold_of_bag == v, seed)
        _, _, curve = fit(x, bag_idx, bags, tr, form, MAX_EPOCHS, seed, val_bags=va)
        curves.append(curve)
    mean_curve = np.mean(np.array(curves), axis=0)
    # The curve is noisy epoch to epoch and its bare argmin is not a stable
    # choice, so the minimum is taken of a three-epoch moving average. Both
    # curves go in the report.
    k = np.convolve(mean_curve, np.ones(3) / 3.0, mode="same")
    k[0], k[-1] = mean_curve[0], mean_curve[-1]
    best = int(np.argmin(k)) + 1
    if report_into is not None:
        report_into["validation curve"] = [round(float(v), 5) for v in mean_curve]
        report_into["smoothed curve"] = [round(float(v), 5) for v in k]
        report_into["epochs chosen"] = best
    return best


def score_fold(x, bag_idx, bags, sizes, starts, train_mask, score_mask, form, seed, epochs):
    """Fit on `train_mask`'s eligible bags, score every bag in `score_mask`."""
    tr, census = training_index(bags, train_mask)
    model, pars, _ = fit(x, bag_idx, bags, tr, form, epochs, seed)
    xs = torch.from_numpy(apply_standardiser(x, pars)).to(DEVICE)
    which = np.flatnonzero(score_mask)
    logits, z = bag_logits(model, xs, sizes, starts, which, form)
    return which, logits, z, census


def action_columns(rows: pl.DataFrame, bags: pl.DataFrame, which: np.ndarray,
                   z: np.ndarray, sizes, starts) -> pl.DataFrame:
    """The hand-level columns the reranker will be offered.

    Everything here is a statistic of the *action* scores inside the bag: the
    noisy-OR aggregate, size-free summaries, and a description of the action the
    head put first, which is the object the competition says exists.
    """
    street = rows["street"].to_numpy()
    act = np.argmax(rows.select(s480.ACT_ONEHOT).to_numpy(), axis=1)
    resp = rows["resp_partner"].to_numpy()
    surp = rows["surprise"].to_numpy()
    n = sizes[which]
    seg_start = np.concatenate([[0], np.cumsum(n)[:-1]])
    z = z.astype(np.float64)
    top_z = np.maximum.reduceat(z, seg_start)
    # second best: mask the argmax out and reduce again, so no sort is needed.
    # The first maximum of each segment, found by letting the earliest index win.
    seg_id = np.repeat(np.arange(which.size), n)
    cand = np.flatnonzero(z >= np.repeat(top_z, n))
    flat_top = np.zeros(which.size, dtype=np.int64)
    flat_top[seg_id[cand][::-1]] = cand[::-1]
    masked = z.copy()
    masked[flat_top] = -np.inf
    second = np.maximum.reduceat(masked, seg_start)
    second = np.where(n > 1, second, top_z)
    log_none = -np.add.reduceat(np.logaddexp(0.0, -z), seg_start)      # log prod (1 - p)
    lse = top_z + TAU * np.log(np.add.reduceat(np.exp((z - np.repeat(top_z, n)) / TAU), seg_start))
    rows_flat = make_gather(sizes, starts)(which)
    cols = {
        "mil_noisy_or": (np.log(-np.expm1(np.minimum(log_none, -1e-12))) - log_none),
        "mil_max_z": top_z,
        "mil_mean_z": np.add.reduceat(z, seg_start) / n,
        "mil_lse_z": lse,
        "mil_z_gap": top_z - second,
        "mil_top_street": street[rows_flat[flat_top]],
        "mil_top_action": act[rows_flat[flat_top]],
        "mil_top_resp_partner": resp[rows_flat[flat_top]],
        "mil_top_surprise": surp[rows_flat[flat_top]],
        "pct_mil_max_z": np.zeros(which.size),
    }
    out = bags[which].select("pair_id", "hand_id").with_columns(
        [pl.Series(k, np.asarray(v, dtype=np.float32)) for k, v in cols.items()])
    return out.with_columns(
        (pl.col("mil_max_z").rank("average").over("pair_id")
         / pl.len().over("pair_id")).cast(pl.Float32).alias("pct_mil_max_z"))


# ------------------------------------------------------------------------ parts

def part_pooling(rows, bag_idx, bags, x, sizes, starts, report, seed=C.SEED) -> str:
    """Choose noisy-OR or log-sum-exp on folds 0-3 alone, and freeze it."""
    t0 = time.time()
    inner = [f for f in range(N_FOLDS) if f != HOLDOUT_FOLD]
    fold_of_bag = bags["fold"].to_numpy()
    is_pos = bags["is_pos_pair"].to_numpy().astype(bool)
    out = {}
    for form in ("noisy_or", "lse"):
        scores, losses, chosen_epochs = [], [], []
        for k in inner:
            others = [f for f in inner if f != k]
            sub = {}
            epochs = nested_epochs(x, bag_idx, bags, others, fold_of_bag, form, seed, sub)
            which, logits, _, _ = score_fold(
                x, bag_idx, bags, sizes, starts, np.isin(fold_of_bag, others),
                (fold_of_bag == k) & is_pos, form, seed, epochs)
            scores.append(proxy_map5(bags, logits, which))
            losses.append(sub["validation curve"][epochs - 1])
            chosen_epochs.append(epochs)
        out[form] = {"mean inner proxy MAP@5": round(float(np.mean(scores)), 5),
                     "per fold": [round(float(s), 5) for s in scores],
                     "epochs per fold": chosen_epochs,
                     "mean inner validation loss": round(float(np.mean(losses)), 5)}
        print(f"{form}: {out[form]} ({time.time() - t0:.0f}s)", flush=True)
    chosen = max(out, key=lambda f: out[f]["mean inner proxy MAP@5"])
    out["chosen"] = chosen
    out["chosen on"] = "folds 0-3 only; fold 4 took no part"
    # One global epoch budget, the median of the nested choices made on folds 0-3.
    # It is a single scalar and fold 4 is not in any rotation that produced it,
    # which is what lets the holdout arm refit without a per-fold epoch search.
    out["epoch budget"] = int(np.median(out[chosen]["epochs per fold"]))
    report["pooling"] = out
    return chosen


# ------------------------------------------------- the nested cross-fit contract
#
# The MIL column is label-derived, so if the model that produced a *training*
# row's column was fitted on the reranker's outer fold, that fold's evidence
# labels reach the ranker through its training rows and the measurement inflates.
# Stage 400 named the arrangement and stage 134 named the same hazard as finding
# A3. The contract here is stage 400's `crossfit(nested=True)`:
#
#     for reranker outer fold k, a row in fold j reads a column produced by a
#     model trained on `universe - {k, j}`
#
# so no column a ranker ever sees was made by a model that saw either the fold it
# is scored on or the row's own fold. The unnested arrangement is built too, as a
# reference, because the size of the inflation belongs on the record.

def blocks(rows, bag_idx, bags, x, sizes, starts, universe: list[int],
           outer_folds: list[int], form: str, seed: int, epochs: int,
           nested: bool, log: dict) -> pl.DataFrame:
    """MIL columns for every positive-pair bag, once per reranker outer fold."""
    fold_of_bag = bags["fold"].to_numpy()
    is_pos = bags["is_pos_pair"].to_numpy().astype(bool)
    cache: dict[tuple[int, ...], tuple] = {}
    parts = []
    for k in outer_folds:
        for j in sorted(set(fold_of_bag.tolist())):
            score_mask = (fold_of_bag == j) & is_pos
            if not score_mask.any():
                continue
            drop = {k, j} if nested else {j}
            train = tuple(sorted(set(universe) - drop))
            if train not in cache:
                tr, census = training_index(bags, np.isin(fold_of_bag, train))
                model, pars, _ = fit(x, bag_idx, bags, tr, form, epochs, seed)
                xs = torch.from_numpy(apply_standardiser(x, pars)).to(DEVICE)
                cache[train] = (model, xs)
                log[f"seed {seed} trained on {train}"] = census
            model, xs = cache[train]
            which = np.flatnonzero(score_mask)
            logits, z = bag_logits(model, xs, sizes, starts, which, form)
            part = action_columns(rows, bags, which, z, sizes, starts)
            parts.append(part.with_columns(pl.lit(k, dtype=pl.Int8).alias("outer_fold"),
                                           pl.lit(j, dtype=pl.Int8).alias("row_fold")))
            if k == outer_folds[0]:
                log.setdefault(f"seed {seed} proxy map5 by row fold", {})[str(j)] = round(
                    proxy_map5(bags, logits, which), 5)
            # the contract, asserted rather than described
            assert k not in train or not nested, f"outer fold {k} trained the block it scores"
            assert j not in train, f"row fold {j} trained its own column"
    return pl.concat(parts).sort("outer_fold", "pair_id", "hand_id")


def part_columns(rows, bag_idx, bags, x, sizes, starts, report, form, seeds, epochs) -> None:
    """Nested five-fold column blocks for the headline reading, once per seed."""
    t0 = time.time()
    log: dict = {}
    for seed in seeds:
        out = blocks(rows, bag_idx, bags, x, sizes, starts, list(range(N_FOLDS)),
                     list(range(N_FOLDS)), form, seed, epochs, True, log)
        out.write_parquet(C.ARTIFACTS / f"481_mil_full_nested_s{seed}.parquet")
        print(f"  seed {seed}: nested blocks written ({time.time() - t0:.0f}s)", flush=True)
    # The leaky reference, one seed, so the inflation has a measured size.
    un = blocks(rows, bag_idx, bags, x, sizes, starts, list(range(N_FOLDS)),
                list(range(N_FOLDS)), form, seeds[0], epochs, False, log)
    un.write_parquet(C.ARTIFACTS / f"481_mil_full_unnested_s{seeds[0]}.parquet")
    report["columns (five-fold, nested)"] = {"epochs": epochs, "seeds": seeds, "fits": log}


def part_holdout(rows, bag_idx, bags, x, sizes, starts, report, form, seed, epochs) -> None:
    """The check that decides whether any number here is real.

    Fold 4 is held out of every fit, every standardiser, the epoch budget and the
    pooling decision. `inner` is the four-fold nested reading over folds 0-3;
    `holdout` scores fold 4 once, from models that only ever saw folds 0-3.
    """
    t0 = time.time()
    fold_of_bag = bags["fold"].to_numpy()
    inner = [f for f in range(N_FOLDS) if f != HOLDOUT_FOLD]
    log: dict = {}
    a = blocks(rows, bag_idx, bags, x, sizes, starts, inner, inner, form, seed, epochs, True, log)
    a.filter(pl.col("row_fold") != HOLDOUT_FOLD).write_parquet(
        C.ARTIFACTS / f"481_mil_inner_s{seed}.parquet")
    b = blocks(rows, bag_idx, bags, x, sizes, starts, inner, [HOLDOUT_FOLD], form, seed,
               epochs, True, log)
    b.write_parquet(C.ARTIFACTS / f"481_mil_holdout_s{seed}.parquet")
    # fold 4 trained nothing, anywhere in either block
    trained_on = [t for t in log if t.startswith(f"seed {seed} trained on")]
    assert all(str(HOLDOUT_FOLD) not in t.split("on ")[1] for t in trained_on), \
        "the holdout fold appears in a training set"
    report["holdout protocol"] = {
        "holdout fold": HOLDOUT_FOLD, "epochs": epochs,
        "holdout pairs": int(bags.filter((pl.col("fold") == HOLDOUT_FOLD)
                                         & (pl.col("is_pos_pair") > 0))["pair_id"].n_unique()),
        "inner pairs": int(bags.filter((pl.col("fold") != HOLDOUT_FOLD)
                                       & (pl.col("is_pos_pair") > 0))["pair_id"].n_unique()),
        "training sets used": sorted(t.split("on ")[1] for t in trained_on),
        "fits": log, "seconds": round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in report["holdout protocol"].items() if k != "fits"},
                     indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="all",
                        choices=("pooling", "epochs", "columns", "holdout", "all"))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()

    t0 = time.time()
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    rows, bag_idx, folds = load_frame()
    bags = bag_table(rows, bag_idx)
    x = rows.select(s480.FEATURES).to_numpy().astype(np.float32)
    sizes = np.bincount(bag_idx)
    starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    frame_census = json.loads(s480.REPORT.read_text())["frame"]
    _, census = training_index(bags, np.ones(bags.height, dtype=bool))
    assert census["censored bags excluded from the loss"] == \
        frame_census["bag_label counts"]["-1"], "censoring does not match the frame census"
    report["setup"] = {
        "instance rows": int(rows.height), "bags": int(bags.height),
        "features": len(s480.FEATURES), "device": str(DEVICE),
        "censoring": census,
        "architecture": {"dim": DIM, "dropout": DROPOUT, "lr": LR,
                         "weight decay": WEIGHT_DECAY, "max epochs": MAX_EPOCHS,
                         "batch bags": BATCH_BAGS, "neg per pos": NEG_PER_POS, "tau": TAU},
        "fold isolation violations": frame_census["fold isolation violation count"]}
    print(json.dumps(report["setup"], indent=2), flush=True)

    form = report.get("pooling", {}).get("chosen")
    if args.part in ("pooling", "all") or form is None:
        form = part_pooling(rows, bag_idx, bags, x, sizes, starts, report)
        REPORT.write_text(json.dumps(report, indent=2))
    if args.part in ("epochs", "all") or "epoch budget" not in report.get("pooling", {}):
        # One global epoch budget, from the folds 0-3 rotation of the chosen form.
        # It is a single scalar, fold 4 is in no rotation that produced it, and it
        # is what lets every later fit refit without its own epoch search.
        sub: dict = {}
        inner = [f for f in range(N_FOLDS) if f != HOLDOUT_FOLD]
        best = nested_epochs(x, bag_idx, bags, inner, bags["fold"].to_numpy(),
                             form, C.SEED, sub)
        report.setdefault("pooling", {}).update(
            {"epoch budget": int(best), "epoch budget curve": sub["validation curve"],
             "epoch budget chosen on": "folds 0-3 rotation only; fold 4 took no part"})
        REPORT.write_text(json.dumps(report, indent=2))
    epochs = int(report["pooling"]["epoch budget"])
    print(f"pooling: {form}, epoch budget {epochs} ({time.time() - t0:.0f}s)", flush=True)

    if args.part in ("holdout", "all"):
        for s in [C.SEED + int(v) for v in args.seeds.split(",")]:
            part_holdout(rows, bag_idx, bags, x, sizes, starts, report, form, s, epochs)
            report.setdefault("holdout protocol by seed", {})[str(s)] = report.pop(
                "holdout protocol")
        REPORT.write_text(json.dumps(report, indent=2))
    if args.part in ("columns", "all"):
        seeds = [C.SEED + int(s) for s in args.seeds.split(",")]
        part_columns(rows, bag_idx, bags, x, sizes, starts, report, form, seeds, epochs)
        REPORT.write_text(json.dumps(report, indent=2))
    report["runtime_s"] = round(time.time() - t0, 1)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"written to {REPORT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
