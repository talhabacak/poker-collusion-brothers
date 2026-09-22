# How the two finals were chosen

Written during the competition, reproduced here unchanged in substance. Every number
is a public-leaderboard reading unless it says otherwise.

## The three-component cross

A submission has three scored columns: risk (R, 70%), behaviour (B, 10%) and the five
evidence hands (E, 20%). Both engines could produce all three, so each file below
differs from its neighbours in named columns only, and the differences read as
single-column measurements.

| file | composition | public |
|---|---|---|
| v55 | R_B + B_B + E_B | 0.91748 |
| **tarik_v55_mil_evidence** | R_B + B_B + E_MIL | **0.91834** — primary final |
| x_v55_ourEv | R_B + B_B + E_A | 0.91233 |
| tarik_v55_mil_evidence_behA | R_B + B_A + E_MIL | 0.91756 |
| **hybrid_riskA_behB_evMIL** | R_A + B_B + E_MIL | **0.91112** — second final |
| v64 | R_A + B_A + E_A | 0.90543 |

Read as differences, holding the other two columns byte-identical:

| component | reading | held fixed |
|---|---|---|
| E_MIL − E_B | +0.00086 | R_B, B_B — not informative on its own at this resolution |
| E_A − E_B | **−0.00515** | R_B, B_B |
| B_A − B_B | −0.00078 | R_B, E_MIL |
| B_B − B_A | −0.00032 | R_A, E_MIL |
| R_A − R_B | **−0.00722** | B_B, E_MIL |

Engine B's column is the better one on every term. Engine A's risk column leads on
development (0.926 against 0.902 cleaned AP) and loses on the board by about 0.010
Pair AP — an inversion engine A's own diagnostics had flagged before the measurement
existed, and the reason no development-only argument was allowed to pick a final.

## Why the primary is the MIL file rather than the best public number

It was not selected because it scored 0.91834. Its case was written down before it
was sent: the offline reading in the engine whose development metric transfers
one-for-one to the board (+0.00729 over a column-matched placebo, interval
[+0.00260, +0.01250]), a predicted range of 0.918–0.920, and the note that
0.20 × 0.0073 ≈ +0.0015 is below the evidence term's public standard error of 0.0021,
so the board could not settle it either way. The board landed inside the predicted
range.

This matters because of the winner's curse. The team had 57 scored submissions; the
maximum of many correlated noisy draws is inflated by roughly +0.007 to +0.021 given
the measured public standard error of the pair term (0.0026–0.0075). A file chosen
*after* seeing a good public number carries that bias; a file whose prediction was
registered first and then met carries the ordinary noise of one draw.

## Why the second final is the hybrid and not v64

Only two files count, and the better of the two is what scores, so the second one
should differ from the primary where diversification pays — the pair ranking — and
nowhere else. v64 carried three engine A components, two of which the board had
already measured as worse (evidence −0.005, behaviour −0.0008). The hybrid keeps
v64's risk column, which is the diverse part (Spearman 0.505 against the primary,
679 of the top 1000 pairs shared), and takes the primary's behaviour and evidence.

It was predicted before it was sent, from the single-column probes:

    v64 + (E_MIL − E_A) + (B_B − B_A) = 0.90543 + 0.00601 + 0.00078 = 0.9122
    pre-registered range 0.909–0.915

and the selection rule was locked before the file existed:

    >= 0.91834          -> hybrid primary, MIL second
    0.90543 – 0.91834   -> MIL primary, hybrid second
    <  0.90543          -> MIL primary, v64 second

It scored **0.91112**, the middle band, so the second final became the hybrid. The
private board later read 0.92237 for the primary and 0.91281 for the hybrid.

## One thing that was never done

Eighteen unlabelled evaluation-like pairs sat at risk 0.9966–0.9982 in development
and carried half of the behaviour term's false-positive damage. They were never
trained against. `unlabelled` is not `negative`: about 150 development pairs of that
description are almost certainly undisclosed targets, and the evaluation set holds
the same population, so a model fitted to push them down would be fitted to push
undisclosed positives down — and the cleaned surrogate would improve while the board
fell. They were used as an audit cohort and nothing else.
