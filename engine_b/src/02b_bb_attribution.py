"""Step 2b: fold/ call attribution in unopened pots. 43.6% of folds have no aggressor because nobody raised yet;
the player being 'faced' is the big blind. Fill last_agg_pidx with the big-blind player's pidx for those rows so every
'facing partner' feature downstream (steps 3, 3b, 3c, 4c) sees folds/calls to the partner's blind.
Keeps a backup of the original file; toggle back by restoring it."""
import polars as pl, shutil, os, time
t0 = time.time()
OUT = 'data/interim/'
src = OUT + 'actions_scored.parquet'; bak = OUT + 'actions_scored_orig.parquet'
if os.path.exists(OUT + 'NO_BB_ATTR'):   # measured on v14: neutral-to-negative (stage-1 0.488->0.468, behavior 0.968->0.956); disabled
    print('BB attribution disabled by flag data/interim/NO_BB_ATTR; actions_scored.parquet left unchanged'); raise SystemExit(0)
if not os.path.exists(bak): shutil.copy(src, bak)
A = pl.read_parquet(bak)
S = pl.read_parquet(OUT + 'seats_feat.parquet').filter(pl.col('relpos') == 2).select('hidx', pl.col('pidx').alias('bb_pidx'))
A = A.join(S, on='hidx', how='left')
fill = pl.col('last_agg_pidx').is_null() & (pl.col('to_call') > 0) & (pl.col('pidx') != pl.col('bb_pidx'))
n = A.select(fill.sum()).item()
A = A.with_columns(pl.when(fill).then(pl.col('bb_pidx')).otherwise(pl.col('last_agg_pidx')).alias('last_agg_pidx')).drop('bb_pidx')
A.write_parquet(src)
print(f'[{time.time()-t0:.1f}s] filled {n} rows ({n/len(A):.3f} of actions) with big-blind attribution; backup at {bak}')
