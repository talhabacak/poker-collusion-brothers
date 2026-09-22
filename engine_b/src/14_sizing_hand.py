"""Step 14: bet-sizing evidence features, per hand and pair.

The teammate's evidence ablation puts their sizing block at -0.0244 MAP@5 when removed: after the
chronology block it is the largest single contributor they have, and we carry almost none of it. Our
policy model already fits a bet-size regressor whose standardised residual `size_z` says how far a bet
departed from the size the population picks in that spot; it is used at action level but never reaches
the evidence ranker.

Collusive value transfer has a sizing signature in both directions: the beneficiary bets larger than the
spot warrants because the money is meant to move, and the donor calls sizes that the population folds to.
So per (hand, pair) we keep sizing summaries for both members, split by whether the partner was still in
the hand and by whether the actor was facing the partner's aggression.

Writes data/interim/sizing_hand.parquet.
"""
import polars as pl, numpy as np, time, os
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; NCH = int(os.environ.get('SZ_CHUNKS', '20'))

S = pl.read_parquet(OUT + 'seats_feat.parquet').select('hidx', 'pidx', 'tidx', 'fold_no')
A = pl.read_parquet(OUT + 'actions_scored.parquet').select(
    'hidx', 'tidx', 'action_no', 'pidx', 'st', 'y', 'size_z', 'amount', 'pot_before', 'to_call',
    'big_blind', 'last_agg_pidx', 'stack_before')
A = A.with_columns((pl.col('amount') / (pl.col('pot_before') + 1e-6)).alias('pot_frac'),
                   (pl.col('to_call') / (pl.col('stack_before') + 1e-6)).alias('call_stack'),
                   (pl.col('amount') / (pl.col('big_blind') + 1e-6)).alias('amt_bb'))
log('actions', A.shape)

outs = []
for ch in range(NCH):
    a = A.filter(pl.col('tidx') % NCH == ch)
    s = S.filter(pl.col('tidx') % NCH == ch).select('hidx', pl.col('pidx').alias('opp'), 'fold_no')
    X = a.join(s, on='hidx').filter(
        (pl.col('opp') != pl.col('pidx')) &
        (pl.col('fold_no').is_null() | (pl.col('fold_no') > pl.col('action_no'))))
    agg_ = pl.col('y').is_in([3, 4, 5])
    call_ = pl.col('y').is_in([1, 2]) & (pl.col('to_call') > 0)
    fac = pl.col('last_agg_pidx') == pl.col('opp')          # actor answering THIS opponent
    X = X.with_columns(
        pl.when(agg_).then(pl.col('size_z')).alias('sz'),
        pl.when(agg_).then(pl.col('pot_frac')).alias('pf'),
        pl.when(agg_ & fac).then(pl.col('size_z')).alias('sz_fac'),
        pl.when(call_ & fac).then(pl.col('call_stack')).alias('cs_fac'),
        pl.when(call_ & fac).then(pl.col('amt_bb')).alias('paid_fac'))
    g = X.group_by(['hidx', 'pidx', 'opp']).agg(
        pl.col('sz').max().alias('sz_max'), pl.col('sz').mean().alias('sz_mean'), pl.col('sz').sum().alias('sz_sum'),
        (pl.col('sz') > 1.5).sum().alias('sz_big'), pl.col('pf').max().alias('pf_max'),
        pl.col('sz_fac').max().alias('szf_max'), pl.col('sz_fac').sum().alias('szf_sum'),
        pl.col('cs_fac').max().alias('csf_max'), pl.col('paid_fac').sum().alias('paidf_sum'))
    outs.append(g); log('chunk', ch, g.height)

H = pl.concat(outs).with_columns(pl.min_horizontal('pidx', 'opp').alias('p'), pl.max_horizontal('pidx', 'opp').alias('q'))
C = [c for c in H.columns if c not in ('hidx', 'pidx', 'opp', 'p', 'q')]
P = H.group_by(['hidx', 'p', 'q']).agg(
    [pl.col(c).max().alias('szh_' + c + '_hi') for c in C] + [pl.col(c).min().alias('szh_' + c + '_lo') for c in C])
P = P.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float32) for c in P.columns if c.startswith('szh_')])
P.write_parquet(OUT + 'sizing_hand.parquet')
log('written', P.shape, len(P.columns), 'columns')
