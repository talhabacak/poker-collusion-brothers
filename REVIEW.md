# Review checklist

An internal checklist for the team's second author, kept in the repository so that what
was verified, and by whom, is visible. Nothing here needs to be written — it is all
verification.

The repository exists to satisfy the three things the organiser asked for within seven
days of the private leaderboard: a write-up of at most 1,500 words, code that reproduces
the selected submission, and five evidence case reviews. All three are in place. If a
sentence anywhere states something you disagree with, say so and it gets corrected.

---

## 1. Is the code your code? (2 minutes)

The thirteen base-chain stages in `pipeline/scripts/` were taken from commit `823dce0`,
the commit that built v55. In the competition repository, with the `tarik` branch
fetched:

```bash
git fetch origin tarik
for f in 01_prep 02_policy 03_pairhand 03b_relational 03c_extra 13_oppaware \
         15_oppcond_policy 04_hand_scorer 04c_action_model 04b_stage1b \
         05b_evidence_reranker 06_validate_submission; do
  diff <(tr -d '\r' < <REPO>/pipeline/scripts/$f.py) <(git show 823dce0:src/$f.py | tr -d '\r') \
    > /dev/null && echo "SAME $f" || echo "DIFFERS $f"
done
diff <(tr -d '\r' < <REPO>/pipeline/scripts/05_pair_risk_behaviour.py) \
     <(git show 823dce0:src/05_pair_model.py | tr -d '\r')      # expect no difference
```

- The one exception is **`01_prep.py`**: on Windows `Pool` workers are spawned rather than
  forked, so the body of the script moved into `main()` under an
  `if __name__ == "__main__"` guard. It is stated in the docstring. No feature, seed,
  filter or constant changed — this is the only code change in the repository and it is
  worth a look from you.
- `05_pair_model.py` was renamed **`05_pair_risk_behaviour.py`** because the risk chain
  also has a file of that name. The contents are identical.
- `05b_mil.py` is the hooked copy of your reranker (stage 533 measures through it); run
  with none of its hooks set it reproduces 0.66085 exactly.

## 2. Are the flags right? (1 minute)

The `base` phase of `run_all.py` follows `docs/base_chain/run_A3.sh` line by line:

```
U_WEIGHT=0.0 MIXED_NEG_W=0 USE_VAL=0 USE_CT=0 USE_CTR=0 USE_OA=1 USE_OC=1 OA_COLS=""
USE_OAH=1 USE_OCH=1 RERANK_SEEDS=42,7,2024,11,99,5
SUB_OUT=submission_v55_base.csv → RERANK_IN=submission_v55_base.csv
```

```bash
python run_all.py --dry-run --with-base | grep base1[12]
```

`10_ambiguous_family.py` (the fourth-family rule) is **not** in the phase: v55 is the file
without it, and `run_A3.sh` produced that variant as a separate file.

## 3. Your own records (5 minutes)

`docs/base_chain/` carries, from `edddbb5`: `AUTHOR_README.md`, `EXPERIMENTS.md`,
`SOLUTION_WRITEUP.md`, `case_reviews.md` and the six run scripts in the v55 lineage.

The repository is in English, so the two Turkish documents were translated — ids,
numbers, file names and decisions unchanged, with a note to that effect in their first
lines. Four progress `echo` lines in the run scripts were translated the same way.
**Please check the translations read as you meant them**, particularly the rules
(8-14) in `EXPERIMENTS.md`.

The folder's README marks `SOLUTION_WRITEUP.md` and `case_reviews.md` as superseded, since
the first describes the v33/v34 chain and the second is drawn from a file the team did not
select. Any objection to those two being labelled that way?

## 4. The claims this repository makes about your chain (the real review)

Tell me if any of these is wrong or overstated:

| where | claim |
|---|---|
| `WRITEUP.md` | porting the opponent-aware block moved the board 0.90858 → 0.91748 (+0.0089) |
| `WRITEUP.md` | risk/behaviour are LightGBM with unlabelled pairs at weight 0; evidence is lambdarank, six seeds, top-50 candidates |
| `docs/TWO_CHAINS.md` | at the merge your best was v45 at 0.90858 and mine v41/v44-0 at 0.90731 — 0.0013 apart |
| `docs/TWO_CHAINS.md` | the composition and public/private score of every file submitted after the merge |
| `docs/MEASUREMENT_NOTES.md` | the late v58 read 0.92121/0.92288 and v58_amb 0.92197/0.92319; against the selected file that is +0.00051 / +0.00082 private |
| `README.md` | rebuilt in a clean tree, the evidence side of v55 reproduced 0.66085 / 0.65164 exactly |
| `pipeline/README.md` | which stage writes what |

## 5. The one open item only you can close

A full rerun of the base chain on this machine, with a byte comparison against
`tarik_v55_best.csv`, has **not** been done — the README says so plainly. If your
environment is still standing, running `run_A3.sh` once and comparing the sha256 of the
result against

```
41688e38733a38fed7efdb3c98662877e813e333abb9f0c3934dbdd4f2e2cadf   tarik_v55_best.csv
```

would close it. If it matches, the "not rerun here" caveat comes out of the README and the
reproduction chain is verified end to end. If it does not match we want to know too — the
likely cause is library versions, and your pins are recorded in `requirements.txt`.

## 6. Publication

The repository is **private** for now. It goes public once you have approved it, and the
write-up is then published on Kaggle with its URL posted as a reply to discussion 742244.
**The deadline is 28 September.**

- Is there anywhere you would rather not be named? (`LICENSE`, the credits in `README.md`,
  the stage headings in `pipeline/README.md`, `docs/TWO_CHAINS.md`)
- `docs/base_chain/EXPERIMENTS.md` is your full experiment log. Any objection to it being
  public?
