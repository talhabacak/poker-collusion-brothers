# Detect Suspicious Value Transfers in Poker — team Brothers

Reproduction repository for the two submissions team **Brothers** selected for final
scoring in the Kaggle competition
[Detect Suspicious Value Transfers in Poker](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker).

| file | role | public | private |
|---|---|---|---|
| `tarik_v55_mil_evidence.csv` | primary final — the team's leaderboard result | 0.91834 | **0.92237** |
| `hybrid_riskA_behB_evMIL.csv` | second final | 0.91112 | 0.91281 |

Private leaderboard: **11th**; public 14th. 371 teams on the board.

* Method: [`WRITEUP.md`](WRITEUP.md) (the Kaggle solution write-up, ~1,340 words).
* Five evidence case reviews: [`CASE_REVIEWS.md`](CASE_REVIEWS.md).
* Both submitted files, byte for byte: [`submissions/submitted/`](submissions/submitted).
* Checksums and scores of everything: [`verify/checksums.json`](verify/checksums.json).

All signals are derived from gameplay only — `hands`, `seats` and `actions`: cards,
actions, amounts, board, and the order of hands inside the scored phase. Identifier
formats, row or file ordering, `players.csv` metadata and any other artefact outside
the gameplay logs are not read by any stage that contributes to a submitted column.

---

## Layout

    run_all.py             every stage, in order, with --list / --dry-run / --from / --only
    engine_a/              pipeline A: caches, v64's risk column, the MIL evidence head,
                           the two build steps that produced the final files
      src/pokercol/        shared library (cards, folds, metrics, config)
      scripts/             numbered stages; the numbers are the project's own stage ids
    engine_b/              pipeline B: v55's risk, behaviour and evidence columns
      src/                 its author's code, unchanged apart from the note below
    verify/                checksums, the submission comparison, the case-review facts
    submissions/submitted/ the three files as sent to Kaggle
    docs/                  how the two finals were chosen, and what was measured wrong

The competition ran with two independent pipelines that merged into one team on
19 September. Engine B is the work of the team's second member; it is included here
because the primary final carries its risk, behaviour and candidate columns, and the
organiser asks for everything needed to reproduce the selected submission.

---

## Setup

Python 3.13 for engine A, Python 3.12 for engine B (its pins are older; a single
3.13 environment runs phases A, C, D and V, which is everything the default run
needs).

    git clone <this repository>
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

To rebuild v55 as well (phase B), make a second environment from
`requirements-engine-b.txt` and run that phase with it.

**Machine.** Everything here was built on 20 logical cores, 32 GB of RAM and an
RTX 5070 Ti. The MIL head (phase C) uses PyTorch and is small enough to run on CPU;
`CUDA_VISIBLE_DEVICES=` forces that. Free disk: about 25 GB — 5.4 GB of engine A
caches, 9 GB of engine B intermediates, 0.3 GB of raw data, the rest headroom.

---

## Running it

    python run_all.py --list        # every stage, what it writes, whether that exists
    python run_all.py --dry-run     # the plan and each stage's environment, run nothing
    python run_all.py               # phases A, C, D, V -> both final files
    python run_all.py --with-engine-b          # rebuild v55 first (phase B)
    python run_all.py --from a156              # resume at a stage
    python run_all.py --only C,D               # a subset of phases
    python run_all.py --with-measurements      # stage 533's full four-arm protocol

Phases, and what each produces:

| phase | what it builds | ends in |
|---|---|---|
| B | engine B's pair, behaviour and evidence columns | `submissions/tarik_v55_best.csv` |
| A | engine A's caches and v64's risk column | `submissions/_risk_v64_union_oppaware.csv` |
| C | the action-level MIL head in engine B's tables, and the primary final | `submissions/tarik_v55_mil_evidence.csv` |
| D | the second final: v64's risk under the primary's other columns | `submissions/hybrid_riskA_behB_evMIL.csv` |
| V | format validation and sha256 against the submitted files | `artifacts/run_manifest.json` |

Phase C reads `submissions/tarik_v55_best.csv`. If phase B has not been run,
`run_all.py` copies the submitted v55 out of `submissions/submitted/` and says so.

**Runtimes.** Each stage's seconds are written to `artifacts/run_manifest.json`, so
the numbers below are what a rerun will report back. Measured on the machine above:
stage 150 (opponent-aware policy, 49M action-opponent rows) 1,448 s, stage 20 322 s,
stage 02b 233 s, the rest of phase A's measured stages about 420 s together — call
phase A an hour, with 04e, 49 and 156 not separately timed. Phase C's stage 534 took
319 s; phases D and V take seconds. Phase B is the long one: engine B's own README
puts its short chain at about 35 minutes on 16 cores, and the v55 configuration adds
the two opponent-aware blocks on top of that.

**Determinism.** Seeds are fixed (engine A `20260914`, engine B `42`), LightGBM runs
in deterministic mode with a fixed thread count, folds are grouped by table through
one canonical map, and training frames are sorted on stable keys before bagging.
Stages 534 and 608 copy every column they do not change as *text*, and assert it: the
primary final differs from v55 in the five evidence columns only, and the second
final differs from the primary in `risk_score` only. Those asserts run on every
rebuild.

---

## What is and is not rebuilt here

Being exact about this is more useful to a reviewer than a blanket claim.

**Reproduced here, end to end, from the raw files.** Engine A's caches, v64's risk
column (phase A), the MIL action head and the primary final (phase C), the second
final (phase D). Phases C and D assert byte identity of the carried columns, and
phase V compares both results against the sha256 of the submitted files.

**Reproduced under one caveat: v55 (phase B).** Engine B's code is here and its run
order is `run_all.py`'s phase B, with the flags recorded from its author's
`run_A3.sh` — the script that built the 0.91748 file — and the clean-chain settings
its pair side inherits: `USE_OA=1` on the pair model, `USE_OAH=1 USE_OCH=1` and six
reranker seeds on the evidence side, `U_WEIGHT=0 MIXED_NEG_W=0`, and the
fourth-family rule off. What was verified on this machine is the *evidence* side of
that chain: rebuilt from those flags in a clean tree it reproduced its recorded
development readings exactly (0.66085 for the v55 configuration, 0.65164 for the
baseline arm, maximum absolute difference 0.0 against the original out-of-fold
dumps). The full v55 file was produced by its author on their own machine and is
included byte for byte, so every stage after it is exactly reproducible regardless.

**Library versions.** The manifest of an earlier full run records lightgbm 4.7.0
while the environment that built the finals reports 4.6.0; engine B's own pins are
newer again. A gradient-boosting library can change the last digits of a score
between minor versions without changing a ranking, and the metric reads only
rankings. `verify/check_submissions.py` therefore reports both: the sha256, and — if
it differs — how many rows moved in each column and the rank correlation with the
submitted file.

---

## Verifying without rerunning anything

    python verify/check_submissions.py
    python engine_a/scripts/08_validate_submission.py submissions/submitted/tarik_v55_mil_evidence.csv

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

    POKERCOL_ROOT=$PWD python -m pytest engine_a/tests -q      # 12 passed

---

## Also in here

* [`docs/FINAL_SELECTION.md`](docs/FINAL_SELECTION.md) — the risk/behaviour/evidence
  cross on the board, the prediction registered before the second final was sent, and
  the selection rule that was locked before its score was seen.
* [`docs/MEASUREMENT_NOTES.md`](docs/MEASUREMENT_NOTES.md) — the two validation
  defects that had been shaping decisions, what the public split can and cannot
  settle, the things that were measured and rejected, and the call we got wrong: a
  mechanism recorded as dead on readings from an engine whose development metric does
  not transfer, which scored +0.0037 when rebuilt in the engine where it does.

---

## Credits and licence

Engine A and this repository: Talha Bacak. Engine B: the second member of team
Brothers, included with the team's submission and unchanged except where a `REPO
NOTE` comment marks an adjustment for this repository's layout. Four such notes
exist, all mechanical: the engine A root is taken from `POKERCOL_ROOT` so both
engines share one data directory; stage 533 finds engine B's harness through
`TARIK_HARNESS`; stage 156 falls back to `sample_submission.csv` as its row template
when the competition-era template file is absent (their `pair_id` order is asserted
identical); stage 608 accepts either of the two byte-identical v64 risk files.

Code is released under the MIT licence (`LICENSE`). The competition data is not
redistributed here.
