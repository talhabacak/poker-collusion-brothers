# Detect Suspicious Value Transfers in Poker — the base chain's working notes

*English translation of the base chain's own README (`README.md` at commit `edddbb5` of
the competition repository, written in Turkish by its author). Numbers, column names and
counts are unchanged; only the prose is translated. It documents the problem, the data
and the metric as the chain's author read them. The executable run order is in
[`../../pipeline/README.md`](../../pipeline/README.md).*

The competition: on fully synthetic No-Limit Texas Hold'em (NLHE) data, find **pairs of
players acting together**, name the behaviour type, and for every alert show **concrete
hands** a reviewer can open and look at.

> The data is entirely synthetic. It contains no real players, customers, payments or
> production data.

---

## 1. What is the problem?

Play that looks suspicious usually has an innocent explanation: tilt, inexperience,
playing at the same hours, an unusual strategy, or plain luck. Colluding players, on the
other hand, do not cheat continuously. They sprinkle the hands they manipulate in among
ordinary play.

The system is asked to do three things:

1. **Pair ranking:** produce a risk score between `0` and `1` for every pair of players.
2. **Behaviour classification:** say which type of coordination a suspicious pair falls
   into.
3. **Evidence retrieval:** give the five strongest hands in which the coordination is
   visible, ordered from strongest to weakest.

A high risk score alone is not enough. Part of the score comes from finding the right
hands, so stopping at coarse signals like "these two play together a lot" or "a lot of
chips flowed between them" will not do.

---

## 2. The target behaviours

| Class (`predicted_behavior`) | Meaning | Possible trace in the action log |
|---|---|---|
| `directed_transfer` | One of the players **deliberately** loses value to the other (chip dumping). | Big bet/call with a weak hand, folding a much better hand to the partner, a pointless all-in in the partner's favour. |
| `soft_play` | The partners **do not show normal aggression** against each other. | Only check/call with a strong hand against the partner, limping instead of raising, passivity heads-up. |
| `coordinated_isolation` | The partners **squeeze the other players**, while limiting conflict between themselves. | Squeeze/re-raise to push a third player out of the pot, then passive play between the partners. |
| `other_coordination` | An undisclosed **fourth mechanism**. It **never appears** in the public positive labels. | Unknown. Used when there is coordination but it does not fit the three patterns above. |
| `none` | No target behaviour. | — |

The "possible trace" column is our reading, not the competition's official definition.
The one officially known thing is this: every labelled evidence hand contains a
**visible, behaviour-specific action** inside `actions.parquet`. The scenario merely
being active in the background does not count as evidence.

### Coordination is episodic

- A colluding pair mixes the hands it manipulates in with ordinary ones.
- The relationship does not have to be active across the whole timeline. It may be active
  in the development period and over by the evaluation period, or the other way round.
- So statistics averaged over all hands can drown the signal. **Window-based and
  hand-based** analysis matters.

### Trap patterns (not targets)

The data deliberately contains patterns that resemble the target behaviour without being
it:

- **Tilt:** playing aggressively or badly after losing.
- **Weak play:** playing badly in general and losing chips to everyone.
- **Similar strategies:** two players playing in the same style.
- **Repeated opponent selection:** always being at the same table with the same people.
- **Streaks:** winning or losing runs that come from luck.
- **Strategy changes:** a player's style changing over time.

On top of that, players play at different times and in different volumes. Some pairs share
far more hands than others, so metrics must be **normalised by the number of shared
hands** rather than used as raw counts.

---

## 3. The data

### Overall shape

| Quantity | Value |
|---|---|
| Hands | 2,000,000 (6-handed NLHE) |
| Players | 12,000 |
| Player-hand rows | 12,000,000 |
| Actions | 18,609,028 |
| Pairs to be scored | 112,540 |
| Pools / tables | 400 pools × 30 players, each pool one `table_id` |
| Hands per pool | ~5,000 |
| Time range | 2026-01-01 → 2026-02-02 (UTC) |

**Time split:** inside each pool the first 60% of the hands are the `development` period
and the last 40% the `evaluation` period (`hands.phase`).

- `development`: 1,200,000 hands
- `evaluation`: 800,000 hands

Players only meet inside their own pool. The number of possible pairs is C(30,2) = 435 per
pool, 174,000 in total, of which 112,540 enter the evaluation.

### Directory layout

```
detect-suspicious-value-transfers-in-poker/
├── README.md                     ← this file
└── data/raw/
    ├── players.parquet           (~137 KB)
    ├── hands.parquet             (~48 MB)
    ├── seats.parquet             (~95 MB)
    ├── actions.parquet           (~151 MB)
    ├── development_labels.csv
    ├── development_evidence.csv
    ├── evaluation_pairs.csv
    └── sample_submission.csv
```

Join keys: the gameplay tables on `hand_id`, the player information on `player_id`.

### `players.parquet`: 12,000 rows

| Column | Type | Description / values |
|---|---|---|
| `player_id` | str | Player identifier |
| `account_age_days` | int | Account age in days |
| `experience_hands_bucket` | str | `new` (1,411), `developing` (3,299), `experienced` (4,588), `veteran` (2,702) |
| `preferred_stake` | str | `micro` (6,595), `low` (4,198), `mid` (1,207) |
| `region_bucket` | str | `americas`, `europe`, `apac`, `other` |
| `client_family` | str | `desktop`, `mobile`, `web` |

### `hands.parquet`: 2,000,000 rows

| Column | Type | Description |
|---|---|---|
| `hand_id` | str | Hand identifier |
| `table_id` | str | Table (= pool), 400 distinct values |
| `started_at` | timestamp (UTC) | When the hand started |
| `phase` | str | `development` / `evaluation` |
| `button_seat` | int | The dealer button's seat |
| `small_blind`, `big_blind` | int | The blinds. BB values: 2 (1.09M hands), 4 (715K), 10 (195K) |
| `board_cards` | str | The board cards dealt (e.g. `7h Th 7d Kc`). Short or empty if the hand ended early |
| `final_pot` | int | Final pot size |
| `players_dealt` | int | Number of players dealt in |
| `players_at_showdown` | int | Number of players who reached showdown |

### `seats.parquet`: 12,000,000 rows (6 players per hand)

| Column | Type | Description |
|---|---|---|
| `hand_id`, `player_id` | str | Keys |
| `seat_no` | int | Seat number |
| `starting_stack` | int | Stack at the start of the hand |
| `hole_card_1`, `hole_card_2` | str | The player's hole cards. **Present for everyone, including players who folded**, which is what makes hand-strength analysis possible |
| `total_contribution` | int | Total chips put into the pot |
| `net_chips` | int | Net win/loss on the hand |
| `folded` | bool | Whether they folded |
| `went_to_showdown` | bool | Whether they reached showdown |
| `won_share` | float | Share of the pot won (0–1 for split pots) |

### `actions.parquet`: 18,609,028 rows

| Column | Type | Description |
|---|---|---|
| `hand_id` | str | Hand |
| `action_no` | int | Order of the action within the hand (starts at 0) |
| `street` | str | `preflop` (13.2M), `flop` (2.85M), `turn` (1.61M), `river` (0.91M) |
| `player_id` | str | Who acted |
| `action` | str | `fold` (9.65M), `call` (2.97M), `raise` (2.47M), `check` (1.84M), `bet` (1.50M), `all_in` (176K) |
| `amount` | int | Amount put in with this action |
| `amount_to` | int | The total bet level after the action |
| `pot_before` | int | Pot before the action |
| `stack_before` | int | Stack before the action |
| `to_call` | int | Amount required to call at the moment of the decision |
| `players_active` | int | Players still in the hand at the moment of the decision |

With the decision-point context (`pot_before`, `to_call`, `stack_before`,
`players_active`) and the hole cards together, the question **"was this action
sensible?"** can be asked per hand, against pot odds and hand strength.

### `development_labels.csv`: 1,860 rows

| Column | Description |
|---|---|
| `pair_id` | Pair identifier |
| `player_1`, `player_2` | The players |
| `label` | `1` = target, `0` = not a target |
| `label_status` | `confirmed_target` (372) / `confirmed_non_target` (1,488) |
| `behavior_family` | `directed_transfer` (148), `soft_play` (132), `coordinated_isolation` (92), `none` (1,488) |

⚠️ **This is a positive-unlabelled (PU) setting:** the file does **not** list every
coordinated pair in the development period. A pair that is not in the list is **not
negative, it is unknown**. Training with the unlisted pairs treated as negatives misleads
the model.

The `confirmed_non_target` pairs are probably "hard negatives" containing the trap
patterns (tilt, similar strategy and so on). That makes them very valuable for learning
the negative class.

### `development_evidence.csv`: 1,817 rows

| Column | Description |
|---|---|
| `pair_id` | A public positive pair |
| `evidence_rank` | 1–5, strongest to weakest |
| `hand_id` | The evidence hand (development period) |
| `behavior_family` | Behaviour type |

Of the 372 positive pairs, 340 have five evidence hands, 21 have four and 11 have three.
These hands are the only supervised signal that answers "what does an evidence hand look
like?". They can be used to train an **evidence scorer** at the hand level.

### `evaluation_pairs.csv`: 112,540 rows

| Column | Description |
|---|---|
| `pair_id` | The pair to be scored |
| `player_1`, `player_2` | The players |
| `shared_hands` | Hands played together in the evaluation period. Min 38, median 76, mean ~86, max 419 |

Publicly labelled pair ids, and pairs **containing a public positive player**, have been
removed from this list. So a player who is positive in development never appears in the
evaluation pairs.

---

## 4. Submission format

The file must be called `submission.csv` and contain **one row for every `pair_id`** in
`evaluation_pairs.csv`. `sample_submission.csv` is the template.

| Column | Description |
|---|---|
| `pair_id` | Copied unchanged |
| `risk_score` | Between 0 and 1. Higher = more likely coordination |
| `predicted_behavior` | `none`, `directed_transfer`, `soft_play`, `coordinated_isolation`, `other_coordination` |
| `evidence_hand_1` … `evidence_hand_5` | `hand_id`s from the evaluation period, strongest to weakest. Unused positions take `NO_EVIDENCE` |

Example row:

```csv
pair_id,risk_score,predicted_behavior,evidence_hand_1,evidence_hand_2,evidence_hand_3,evidence_hand_4,evidence_hand_5
P00005AC2A509,0.87,soft_play,H2D8EAC9EC7DA02,HF67EF16BB8EF76,HD9B5F56C491E2D,NO_EVIDENCE,NO_EVIDENCE
```

**Rules:**

- No empty cells. Kaggle rejects them.
- Repeating the same `hand_id` in one row **invalidates the submission**.
- **Both players** of the pair must be seated in an evidence hand.
- Unknown, non-shared, development-period or irrelevant hands do not raise an error but
  **earn no evidence credit**.

---

## 5. Evaluation

The leaderboard score has three components.

### 5.1 Pair AP

Average Precision computed from `risk_score` over all evaluation pairs:

$$\text{AP} = \sum_n (R_n - R_{n-1})\, P_n$$

with $P_n$ and $R_n$ the precision and recall at the n-th threshold. Equal risk scores are
ordered deterministically by `pair_id`, so avoiding ties is worthwhile.

### 5.2 Evidence MAP@5

- Averaged over the **true target pairs** only.
- For every target pair, AP@5 is computed by comparing the five ordered hands submitted
  with the hidden "planted evidence" hands.
- A target pair that is missed (no evidence given) contributes **0**.
- `NO_EVIDENCE` is ignored.

### 5.3 Behavior MAP

- The macro average of one-vs-rest AP over the **three** disclosed families
  (`directed_transfer`, `soft_play`, `coordinated_isolation`).
- The class score for a family is `risk_score` if that family is predicted, `0` otherwise.
- A class that is never predicted contributes **0**. Since the average is always taken
  over three families, never predicting one of them is expensive.
- `other_coordination` is not part of this component.

> How the components are combined into the final score (equal weights or different ones)
> is not stated in the description. To be confirmed from the official metric notebook.

### The leaderboard split

- Public LB: ~30% of the evaluation pairs
- Private LB: ~70%
- The split is stratified by behaviour family. The labels and the LB assignment are
  hidden; the metric code is public.

### Practical consequences of the metric

- **Giving evidence costs nothing.** Since Evidence MAP@5 is computed over the true target
  pairs only, writing evidence hands on negative pairs carries no penalty. It is
  sensible to write **five candidate hands on every row**, including low-risk pairs.
- **The order of the evidence matters.** MAP@5 is order-sensitive; the strongest evidence
  must be in position 1.
- **An evidence hand must always be a shared hand.** Candidates must be chosen from
  evaluation hands in which both players were seated.
- **`other_coordination` is a risky label.** Since that prediction gives no score to any
  of the three families, if the pair really is one of the three known families, Behavior
  MAP loses points. Pair AP and the evidence component are unaffected. It should be used
  only where there is a strong signal that genuinely does not fit the three patterns.
- **The class decision and the risk score are coupled.** Because the class score in
  Behavior MAP is `risk_score`, a misclassified high-risk pair both loses points in its
  true family and creates a false positive in the wrong one.

---

## 6. Rules and winner verification

**Prohibited:** coordination must be inferred from **poker activity**. The following may
not be used:

- ID formats (patterns inside `pair_id`/`player_id`)
- Row or file ordering
- The generator's internals
- Any artefact unrelated to gameplay

**To be eligible for a prize:** within **7 days** of the private leaderboard being
published, and regardless of standing, the following must be published:

1. A Kaggle solution write-up of at most **1,500 words**.
2. A **public notebook or repository** containing the code that can reproduce the selected
   submission.
3. **Five short case reviews** from the submitted evidence. Each must contain the
   `pair_id`, the hand IDs, the observed behaviour and **a plausible innocent alternative
   explanation**.

The organiser may run the code in a clean environment. A submission that cannot be
reproduced, or that breaks the rules, is not eligible.

For that reason the pipeline must be **reproducible from the start**: fixed seeds, fixed
dependency versions and clear run instructions.

---

## 7. First notes on the approach

These are starting ideas, not a settled plan:

1. **Hand-level signals (the evidence engine):** for every hand the pair shares, produce
   scores measuring how "abnormal" the actions the two players take against each other are
   given hand strength and pot odds. Examples: passivity with a strong hand against the
   partner, putting a lot of money in with a weak hand against the partner, joint pressure
   on a third player. The development evidence hands can be used to train or calibrate
   that scorer.
2. **The player's own baseline:** compare a player's behaviour against the partner with
   **the same player's** behaviour against other opponents. That removes traps like
   similar strategy and weak play.
3. **Pair-level aggregation:** aggregate hand scores per pair. Because the structure is
   episodic, use top-k, time windows, density or change-point statistics instead of the
   mean. Normalise the results by `shared_hands`.
4. **PU learning:** do not treat unlabelled pairs as negatives. Use `confirmed_non_target`
   pairs as hard negatives and `confirmed_target` pairs as positives, and handle the
   unlabelled ones separately.
5. **Validation:** build a CV in the development period that imitates the evaluation
   conditions by excluding pairs containing public positive players. Write a local copy of
   the competition's three metrics (taking the official metric notebook as reference).
6. **`other_coordination` through anomaly detection:** a separate branch for relationships
   that do not fit the three known patterns but are clearly abnormal at the pair level.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **NLHE** | No-Limit Texas Hold'em |
| **BB** | Big blind |
| **AP** | Average Precision (for the pair ranking) |
| **MAP@5** | Mean Average Precision for an evidence ranking of at most five hands |
| **PU** | Positive-Unlabelled learning: only some positives are labelled and the rest are unknown |
| **Chip dumping** | One player deliberately losing chips to another (`directed_transfer`) |
| **Soft play** | Colluding players not playing aggressively against each other |
| **Squeeze / isolation** | Aggressive raises made to push opponents out of the pot |
| **Tilt** | Play that deteriorates for emotional reasons, usually after a loss |
| **Showdown** | The cards being turned up after the last betting round |
| **Development / Evaluation** | The first 60% / last 40% of the hands in each pool |
