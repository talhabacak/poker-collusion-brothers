# Two chains: why the work was split, and what happened when it was joined

## Why independently

Team Brothers is two people. For almost the whole competition we worked with **no
shared code, no shared features and no shared diagnostics**, on purpose. Two people
who read each other's work early converge on one way of seeing the problem, and the
second person stops being a second opinion — the same blind spot then gets twice the
compute. So each of us built a full pipeline from the raw logs: data preparation,
features, models, evidence retrieval, validation.

It cost duplicated effort. What it bought was two independent estimates of every
question that mattered, and — as it turned out — the ability to tell a fact about a
*mechanism* from a fact about an *engine*. Several conclusions reversed when the same
idea was measured in the other chain; one of them is written up in
[`MEASUREMENT_NOTES.md`](MEASUREMENT_NOTES.md).

Throughout this repository the two are called the **base chain** (Tarık Bacak; stages
`01_prep` … `05b_evidence_reranker`, the pipeline behind v55) and the **risk chain**
(Talha Bacak; stages `00b` … `156`, plus the MIL head at `531`–`534` and the assembly
steps `534`/`608`). The write-up calls them engine B and engine A. They are one tree
here because the submitted files are a cross of both, not because the work was joint.

## Where each stood alone

The accounts merged into one team on **19 September**, two days before the deadline.
At that point, on the public board:

| chain | best file alone | public |
|---|---|---|
| base chain | v45 — evidence reranker, fourth family by behaviour-head ambiguity | 0.90858 |
| risk chain | v41 / v44-0 — clean chain, no guard | 0.90731 |

Within 0.0013 of each other, from completely different code. That is the first thing
the split bought: a genuine replication, not a re-run.

## What crossed over

Three things, each measured as a single-column change before it was kept.

**Opponent-aware policy residuals, risk chain → base chain.** The risk chain had
measured this block at +0.0087 / +0.0100 in its own stack. Ported into the base
chain's pair model and reranker it moved the board **0.90858 → 0.91748 (+0.0089)**.
The same mechanism priced the same way in two codebases that share no line of code.

**The action-level MIL evidence head, risk chain → base chain's tables.** Built as
stages 531–534, but trained in the base chain's frames and its `tidx % 5` folds,
against a column-matched placebo: +0.00729 [+0.00260, +0.01250], strict holdout
agreeing in sign and size. It produced the **primary final**, 0.91834 public /
0.92237 private.

**The risk chain's pair column, kept for the second final.** After the cross below,
the base chain's columns were better on all three terms, so the second final was not
about strength but about difference: v64's risk column has Spearman 0.505 against the
primary and shares 679 of its top 1000 pairs.

## Everything tried after the merge

Every file below was submitted between the merge and the 22:00 UTC deadline on
20 September, and each differs from its neighbours in named columns only, so each
reading is a single-column measurement. R = risk, B = behaviour, E = evidence;
subscript A = risk chain, B = base chain.

| when (UTC) | file | composition | public | private | what it settled |
|---|---|---|---|---|---|
| 09-19 00:03 | v64 | R_A + B_A + E_A | 0.90543 | 0.90812 | the risk chain's best complete file |
| 09-19 00:44 | v45 | base chain, pre-port | 0.90858 | 0.90316 | the base chain's best complete file |
| 09-19 20:41 | v70 | R_A + B_A + E_A(iso) | 0.89540 | 0.90087 | an evidence swap inside the risk chain: ~0 |
| 09-20 00:01 | **v55** | R_B + B_B + E_B | 0.91748 | 0.92109 | the ported block: **+0.0089** |
| 09-20 01:55 | x_v55_ourEv | R_B + B_B + E_A | 0.91233 | 0.91880 | E_A − E_B = **−0.00515** |
| 09-20 12:58 | **tarik_v55_mil_evidence** | R_B + B_B + E_MIL | **0.91834** | **0.92237** | E_MIL − E_B = +0.00086 → **primary final** |
| 09-20 19:47 | tarik_v55_mil_evidence_behA | R_B + B_A + E_MIL | 0.91756 | 0.92221 | B_A − B_B = −0.00078 |
| 09-20 20:23 | **hybrid_riskA_behB_evMIL** | R_A + B_B + E_MIL | **0.91112** | 0.91281 | R_A − R_B = −0.00722 → **second final** |

Read together: the base chain's column is the better one on every term, while the
risk chain's risk column led on development (0.926 against 0.902 cleaned AP) and
inverted on the board. Without two chains there would have been no way to see that —
a single pipeline cannot measure its own development metric against anything.

## After the deadline

Two more files were built the same night, after 22:00 UTC. They are on the board and
they count for nothing, and they are recorded here because leaving them out would
flatter the result:

| file | what it is | public | private |
|---|---|---|---|
| v58 (late) | routed family specialists rebuilt **inside the base chain's** reranker | 0.92121 | 0.92288 |
| v58_amb (late) | the same with the fourth-family rule on | 0.92197 | 0.92319 |

Against the selected file that is +0.00287 / +0.00363 public and +0.00051 / +0.00082
private. The mechanism had been recorded as dead months of work earlier, on readings
taken in the risk chain, whose development evidence metric was later measured as
0.054 MAP optimistic. Rebuilt where the metric transfers, it paid. That is the
project's clearest lesson and it is stated in full in
[`MEASUREMENT_NOTES.md`](MEASUREMENT_NOTES.md).

## Provenance of the base chain's files

Its stages come from commit `823dce0` on the branch the base chain was developed on —
the commit whose message is "Opponent-aware policy residuals: 0.90858 -> 0.91748" —
and its own run script is included as [`run_A3.sh`](base_chain/run_A3.sh). Every flag the `base`
phase of `run_all.py` sets is read out of that script rather than inferred, which
matters for one of them: `USE_OC=1` turns on the opponent-conditioned residuals *in
the pair model*, and the working copy used for the cross-engine measurements predates
the code that reads it.

## What the reviewer sees in this repository

The selected files need the base chain (for v55's risk and behaviour columns, and for
the tables the MIL head trains in) and the risk chain (for the MIL head, v64's risk
column and the two assembly steps). Both are here, in one pipeline, in run order.
`pipeline/README.md` says which stage belongs to which chain.

This repository is a cleaned-up copy of the working repository: 291 stages there, most
of them experiments that never reached a submission, with the measurement log behind
every decision. If the organiser wants that tree as it stood during the competition,
dead ends included, we can provide it.
