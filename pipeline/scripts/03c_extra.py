"""Step 3c: pair-hand features that do NOT require the partner to still be in the hand.
Isolation evidence is mostly about third players folding to the pair's aggression, which the partner-active
condition in step 3 filters away. Computed for every pair of every hand, both phases."""
import polars as pl, numpy as np, time, os
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; XD = OUT + 'extra/'; os.makedirs(XD, exist_ok=True)
NCH = 10
seats = pl.read_parquet(OUT + 'seats_feat.parquet')
acts = pl.read_parquet(OUT + 'actions_scored.parquet')
for ch in range(NCH):
    S = seats.filter(pl.col('tidx') % NCH == ch).select('hidx', 'pidx', 'big_blind')
    A = acts.filter(pl.col('tidx') % NCH == ch).sort(['hidx', 'action_no'])
    A = A.with_columns((pl.col('amount').cum_sum().over(['hidx', 'pidx']) - pl.col('amount')).alias('invested_before'))
    others = S.select('hidx', pl.col('pidx').alias('o'))
    # (1) third player folds facing aggression from a pair member -> credit the pair (aggressor, any other seated player)
    T = (A.filter((pl.col('y') == 0) & pl.col('last_agg_pidx').is_not_null())
         .select('hidx', 'big_blind', 'invested_before', pl.col('pidx').alias('actor'), pl.col('last_agg_pidx').alias('r'))
         .join(others, on='hidx').filter((pl.col('o') != pl.col('actor')) & (pl.col('o') != pl.col('r')))
         .with_columns(pl.min_horizontal('r', 'o').alias('p'), pl.max_horizontal('r', 'o').alias('q'),
                       (pl.col('invested_before') / pl.col('big_blind')).alias('dead'))
         .group_by(['hidx', 'p', 'q']).agg(pl.len().alias('n3_fold_to_pair'), pl.col('dead').sum().alias('third_dead_bb'),
                                           (pl.col('dead') > 1).sum().alias('n3_fold_after_invest')))
    # (2) aggression by a pair member while more than two players are live, and re-raises over a third player
    G = (A.filter(pl.col('y') >= 3).select('hidx', 'players_active', 'st', pl.col('pidx').alias('a'), 'last_agg_pidx')
         .join(others, on='hidx').filter(pl.col('o') != pl.col('a'))
         .with_columns(pl.min_horizontal('a', 'o').alias('p'), pl.max_horizontal('a', 'o').alias('q'),
                       (pl.col('last_agg_pidx').is_not_null() & (pl.col('last_agg_pidx') != pl.col('o')) & (pl.col('last_agg_pidx') != pl.col('a'))).alias('over_third'))
         .group_by(['hidx', 'p', 'q']).agg((pl.col('players_active') > 2).sum().alias('n_pair_agg_multiway'),
                                           pl.col('over_third').sum().alias('n_pair_reraise_over_third'),
                                           ((pl.col('st') == 0) & pl.col('last_agg_pidx').is_null()).sum().alias('n_pair_open_raise')))
    D = (T.join(G, on=['hidx', 'p', 'q'], how='full', coalesce=True)
         .with_columns([pl.col(c).fill_null(0).cast(pl.Float32) for c in ['n3_fold_to_pair', 'third_dead_bb', 'n3_fold_after_invest',
                                                                         'n_pair_agg_multiway', 'n_pair_reraise_over_third', 'n_pair_open_raise']]))
    D.write_parquet(XD + f'chunk{ch:02d}.parquet')
    log('chunk', ch, D.shape)
log('done')
