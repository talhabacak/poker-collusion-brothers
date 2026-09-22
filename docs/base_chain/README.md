# The base chain's own record

Everything in this folder is Tarık Bacak's, taken from the branch the base chain was
developed on: commit `edddbb5`, with `run_A3.sh` as it stood at `823dce0`, the commit
that built v55. It is here so the base chain arrives with its own documentation rather
than only as code, and so a reviewer can check the claims the team's write-up makes
about it against the notes they were taken from.

**On language.** The chain's author wrote in Turkish. This repository is in English, so
`AUTHOR_README.md` and `EXPERIMENTS.md` are translations and say so in their first lines;
experiment ids, numbers, file names, metric readings and decisions are unchanged, and the
untouched originals are in the competition repository at the commit named above. In the
run scripts, four progress `echo` lines were translated the same way — every command,
flag, file name and ordering in them is exactly as it was.

| file | what it is | still current? |
|---|---|---|
| `AUTHOR_README.md` | the chain's own README: the problem, the data, the metric, the prize rules and the first notes on the approach | yes as a description of the problem and the data |
| `EXPERIMENTS.md` | the experiment log: 117 numbered experiments with their local reading, leaderboard result and decision (KEPT / REJECTED / PENDING), plus the 14 rules derived from them | yes — this is the record behind every "measured and rejected" in the team's docs |
| `SOLUTION_WRITEUP.md` | the chain's own solution write-up | **no** — it describes the v33/v34 chain, two selection rounds before v55, and predates the opponent-aware blocks, the MIL evidence column and the team merge. The write-up for the submitted files is [`../../WRITEUP.md`](../../WRITEUP.md) |
| `case_reviews.md` | five case reviews written for the chain's own submission at the time | **no** — they are drawn from a file the team did not select. The five reviews for the selected submission are [`../../CASE_REVIEWS.md`](../../CASE_REVIEWS.md), built from `tarik_v55_mil_evidence.csv` |

## The run scripts

Six of the chain's shell scripts, in the order the v55 configuration was reached:

| script | what it ran |
|---|---|
| `run_clean.sh` | the clean chain — the configuration v41/v45 were built from |
| `run_oa.sh` | the opponent-aware policy residuals, pair side |
| `run_oah.sh` | the same block on the hand side, for the evidence reranker |
| `run_oc.sh` | the opponent-conditioned policy and its residuals |
| `run_A2.sh` | the evidence side with the opponent-aware hand block only |
| **`run_A3.sh`** | **the script that built v55** — both blocks on both sides, six reranker seeds |

`run_A3.sh` is the one the `base` phase of `run_all.py` follows flag for flag. The paths
in these scripts are their author's (`/home/tarik/...`, `.venv/bin/python`); they are kept
as they were rather than rewritten, because their value here is as the record of what was
run, not as something to execute. `run_all.py` is the executable version.

One line of `run_A3.sh` is deliberately not in the `base` phase: it also builds
`submission_A3_both_blocks.csv`, the same file with the fourth-family rule applied. v55 is
the file without that rule, and v55 is what the selected submission is built on.

## What is not here

The chain's `reference/official_metric.py` is the competition's own metric implementation
and belongs to the host, so it is not redistributed; its source is the official metric
notebook on Kaggle. The chain's other ~30 run scripts cover experiments that
`EXPERIMENTS.md` records as rejected and that no submitted file depends on.
