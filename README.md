# Detect Suspicious Value Transfers in Poker — team Brothers

Reproduction repository for the two submissions team **Brothers** selected for final
scoring in the Kaggle competition
[Detect Suspicious Value Transfers in Poker](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker).

| file | role | public | private |
|---|---|---|---|
| `tarik_v55_mil_evidence.csv` | primary final — the team's leaderboard result | 0.91834 | **0.92237** |
| `hybrid_riskA_behB_evMIL.csv` | second final | 0.91112 | 0.91281 |

Private leaderboard: **11th**; public 14th. 371 teams on the board.

* Method: [`WRITEUP.md`](WRITEUP.md) (the Kaggle solution write-up).
* Five evidence case reviews: [`CASE_REVIEWS.md`](CASE_REVIEWS.md).
* Stage-by-stage map of the code: [`pipeline/README.md`](pipeline/README.md).
* Both submitted files, byte for byte: [`submissions/submitted/`](submissions/submitted).
* Checksums and scores of everything: [`verify/checksums.json`](verify/checksums.json).

All signals are derived from gameplay only — `hands`, `seats` and `actions`: cards,
actions, amounts, board, and the order of hands inside the scored phase. Identifier
formats, row or file ordering, player metadata and any other artefact outside the
gameplay logs are not read by any stage that contributes to a submitted column.

---

## How the team worked, and how this repository was made

The two of us worked **independently for almost the whole competition**: no shared
code, no shared feature ideas, no shared diagnostics. That was deliberate. Two people
reading each other's work early converge on one way of seeing the problem, and the
second person stops being a second opinion. So each of us built a full pipeline —
data preparation, features, models, evidence retrieval — from the raw logs on our own.

Only in the final days did we compare, and then we combined what each had measured to
be better. The opponent-aware policy residuals came from one chain into the other and
moved the board 0.90858 → 0.91748. The action-level MIL evidence head was rebuilt
inside the other chain's tables and folds and produced the primary final. The second
final keeps the first chain's risk column, because after all the column-by-column
probes that was the piece that still differed most from the primary. The write-up
calls the two chains "engine A" and "engine B"; that is history, not layout. **In this
repository they are one pipeline in one tree.**

This repository is a **cleaned-up copy** of the working repository. That one holds 291
stages, most of them experiments that never reached a submission, together with the
measurement logs behind every decision. Everything needed to rebuild the two selected
files is here; if the organiser wants the working repository as it stood during the
competition, including the dead ends, we can provide it.

---

## Layout

    run_all.py             every stage in order, with --list / --dry-run / --from / --only
    pipeline/
      scripts/             all stages (and the modules they import); see pipeline/README.md
      src/pokercol/        shared library: card evaluator, folds, metrics, paths
      tests/               unit tests, including a brute-force reference for the evaluator
    verify/                checksums, the submission comparison, the case-review facts
    submissions/submitted/ the three files as sent to Kaggle
    docs/                  how the two finals were chosen, and what was measured wrong

---

## Setup

Python 3.13, one environment for the whole repository.

    git clone https://github.com/talhabacak/poker-collusion-brothers.git
    cd poker-collusion-brothers
    python -m venv .venv && .venv/Scripts/activate      # Windows
    #  python3 -m venv .venv && source .venv/bin/activate  # Linux/macOS
    pip install -r requirements.txt

Then the competition data, into `data/raw/`:

    kaggle competitions download -c detect-suspicious-value-transfers-in-poker -p data/raw
    unzip data/raw/detect-suspicious-value-transfers-in-poker.zip -d data/raw

`data/raw/` must hold the eight files `hands.parquet`, `seats.parquet`,
`actions.parquet`, `players.parquet`, `development_labels.csv`,
`development_evidence.csv`, `evaluation_pairs.csv`, `sample_submission.csv`. Their
sha256 are in `verify/checksums.json`; `run_all.py` records the ones it sees in its
manifest, so a changed input cannot go unnoticed.

**Machine.** Everything here was built on 20 logical cores, 32 GB of RAM and an
RTX 5070 Ti. The MIL head uses PyTorch and is small enough to run on CPU;
`CUDA_VISIBLE_DEVICES=` forces that. Free disk: about 25 GB — 5.4 GB of caches for the
risk chain, 9 GB of intermediates for the base chain, 0.3 GB of raw data, the rest
headroom.

---

## Running it

    python run_all.py --list        # every stage, what it writes, whether that exists
    python run_all.py --dry-run     # the plan and each stage's environment, run nothing
    python run_all.py               # risk, mil, hybrid, verify -> both final files
    python run_all.py --with-base              # rebuild v55 itself first
    python run_all.py --from risk12            # resume at a stage
    python run_all.py --only mil,hybrid        # a subset of phases
    python run_all.py --with-measurements      # stage 533's full four-arm protocol

| phase | what it builds | ends in |
|---|---|---|
| base | the pair risk, behaviour and evidence columns of v55 | `submissions/tarik_v55_best.csv` |
| risk | the caches behind v64's risk column, and that column | `submissions/_risk_v64_union_oppaware.csv` |
| mil | the action-level MIL head, and the primary final | `submissions/tarik_v55_mil_evidence.csv` |
| hybrid | the second final: v64's risk under the primary's other columns | `submissions/hybrid_riskA_behB_evMIL.csv` |
| verify | format validation and sha256 against the submitted files | `artifacts/run_manifest.json` |

The `mil` phase reads `submissions/tarik_v55_best.csv`. If the `base` phase has not
been run, `run_all.py` copies the submitted v55 out of `submissions/submitted/` and
says so.

**Runtimes.** Each stage's seconds are written to `artifacts/run_manifest.json`, so
the numbers below are what a rerun will report back. Measured on the machine above:
stage 150 (opponent-aware policy, 49M action-opponent rows) 1,448 s, stage 20 322 s,
stage 02b 233 s, the rest of the risk phase's measured stages about 420 s together —
call that phase an hour, with 04e, 49 and 156 not separately timed. Stage 534 took
319 s; the hybrid and verify phases take seconds. The base phase is the long one: its
short chain is about 35 minutes on 16 cores, and the v55 configuration adds the two
opponent-aware blocks on top of that.

**Determinism.** Seeds are fixed (`20260914` in the risk chain, `42` in the base
chain), LightGBM runs in deterministic mode with a fixed thread count, folds are
grouped by table through one canonical map, and training frames are sorted on stable
keys before bagging. Stages 534 and 608 copy every column they do not change as
*text*, and assert it: the primary final differs from v55 in the five evidence columns
only, and the second final differs from the primary in `risk_score` only. Those
asserts run on every rebuild.

---

## What is and is not rebuilt here

Being exact about this is more useful to a reviewer than a blanket claim.

**Reproduced here, end to end, from the raw files.** The caches and v64's risk column
(`risk`), the MIL action head and the primary final (`mil`), the second final
(`hybrid`). The last two assert byte identity of the columns they carry through, and
`verify` compares both results against the sha256 of the submitted files.

**Reproduced under one caveat: v55 (the `base` phase).** The run order and every flag
in this phase are `docs/base_chain/run_A3.sh` — the script in commit `823dce0`
("Opponent-aware policy residuals: 0.90858 -> 0.91748") that built the file — copied
across rather than reconstructed: `U_WEIGHT=0.0 MIXED_NEG_W=0 USE_VAL=0 USE_CT=0
USE_CTR=0 USE_OA=1 USE_OC=1 OA_COLS=""` on the pair model, `USE_OAH=1 USE_OCH=1` with
six reranker seeds on the evidence side. The stage files themselves are that commit's,
byte for byte, except `01_prep.py`, which carries a structural patch for Windows
described in `pipeline/README.md`. The script also produced a fourth-family variant of
the file; v55 is the one without it, and that is what the phase ends on.

What has *not* been done on this machine is a full rerun of that chain with a byte
comparison against v55. What was checked is its evidence side: rebuilt from these
flags in a clean tree it reproduced its recorded development readings exactly (0.66085
for the v55 configuration, 0.65164 for the baseline arm, maximum absolute difference
0.0 against the original out-of-fold dumps). The v55 file itself is included byte for
byte, so every stage after it is exactly reproducible regardless.

**Library versions.** The manifest of an earlier full run records lightgbm 4.7.0 while
the environment that built the finals reports 4.6.0, and the base chain was developed
under newer pins still (listed in `requirements.txt`). A gradient-boosting library can
change the last digits of a score between minor versions without changing a ranking,
and the metric reads only rankings. `verify/check_submissions.py` therefore reports
both: the sha256, and — if it differs — how many rows moved in each column and the
rank correlation with the submitted file.

**Changes to competition-era code.** Five, all mechanical, each marked with a
`REPO NOTE` comment: the root directory comes from `POKERCOL_ROOT` so the two chains
share one data directory; stage 533 finds the reranker harness through
`TARIK_HARNESS`; stage 156 falls back to `sample_submission.csv` as its row template
when the competition-era template is absent, asserting the `pair_id` order is
identical, and its docstring records that `build_combo.py` is not here because neither
selected file uses the columns it attached; stage 608 reads the risk column straight
from stage 156's output. Two more were found by a clean rerun from a fresh clone: stage
`01_pair_universe` resolves its library path from its own location rather than the
working directory, and stage 533 hands the reranker harness v55 as `RERANK_IN` in
place of a competition-tree file no stage writes (the harness takes only risk and
behaviour from it, and stage 534 replaces both with v55's text). `run_all.py` also
creates `logs/` and passes stage 534 `--user-approved`, the flag that records the
decision to ship the MIL head below its bar. No feature, seed, threshold or model
parameter is touched.
The one rename is `05_pair_model.py` of the base chain, which became
`05_pair_risk_behaviour.py` because both chains had a file of that name.

---

## Verifying without rerunning anything

    python verify/check_submissions.py
    python pipeline/scripts/08_validate_submission.py submissions/submitted/tarik_v55_mil_evidence.csv

The validator checks the column order, the 112,540 rows and their pair ids, the
behaviour vocabulary, the risk range, that no evidence hand is repeated inside a row,
and that every evidence hand is an evaluation-phase hand in which both players of the
pair were seated.

The case reviews are checkable the same way, from the public log rather than from the
model:

    python verify/case_facts.py --submission submissions/submitted/tarik_v55_mil_evidence.csv \
        --pairs P9B9FA14D2D0E,P8E67334B0E04,PD91D4EB58627,P074E4EC05254,P2AD7AE0B444B

It prints, for each submitted evidence hand, the board as it stood at each decision,
both players' best five at that moment, who the folding player was answering, and
whether they gave up the better hand — and, per pair, how often that happens with the
partner against how often it happens with every other opponent, divided by exposure.

The hand evaluator and the bootstrap have unit tests, including a brute-force
reference for the seven-card path:

    POKERCOL_ROOT=$PWD python -m pytest pipeline/tests -q      # 12 passed

---

## Also in here

* [`docs/FINAL_SELECTION.md`](docs/FINAL_SELECTION.md) — the risk/behaviour/evidence
  cross on the board, the prediction registered before the second final was sent, and
  the selection rule that was locked before its score was seen.
* [`REVIEW.md`](REVIEW.md) — the checklist the team's second author worked through:
  the commands that diff his stages against his own commit, the flag set against his
  own run script, and the claims this repository makes about his chain.
* [`docs/base_chain/`](docs/base_chain) — the base chain's own record, copied
  verbatim from the branch it was developed on: its README and run order, its
  86 KB experiment log, its own write-up and case reviews (both superseded, and
  marked as such), and the six run scripts that lead to v55.
* [`docs/TWO_CHAINS.md`](docs/TWO_CHAINS.md) — why the two of us worked without
  sharing code until the last days, where each chain stood alone, what crossed
  over when they were joined, and every file submitted after the merge with what
  it settled.
* [`docs/MEASUREMENT_NOTES.md`](docs/MEASUREMENT_NOTES.md) — the two validation
  defects that had been shaping decisions, what the public split can and cannot
  settle, the things that were measured and rejected, and the call we got wrong: a
  mechanism recorded as dead on readings from a chain whose development metric does
  not transfer, which scored +0.0037 when rebuilt in the chain where it does.

---

## Credits and licence

Team Brothers: Talha Bacak and Tarık Bacak. Code released under the MIT licence
(`LICENSE`). The competition data is not redistributed here.
