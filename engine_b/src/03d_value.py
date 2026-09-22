"""Step 3d: counterfactual value, not probability. Surprisal asks how unlikely an action was; this asks how many
chips it moved relative to what the population policy would have committed in the same spot, signed toward the
partner. 'Value transfer' is the competition's own framing and we have only been measuring probability so far."""
import polars as pl, numpy as np, time, os
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; VD = OUT + 'value/'; os.makedirs(VD, exist_ok=True)
NCH = 10
seats = pl.read_parquet(OUT + 'seats_feat.parquet')
acts = pl.read_parquet(OUT + 'actions_scored.parquet')
for ch in range(NCH):
    S = seats.filter(pl.col('tidx') % NCH == ch)
    A = acts.filter(pl.col('tidx') % NCH == ch)
    bb = pl.col('big_blind')
    # what the policy would have put in: fold commits nothing, passive commits the call amount, aggression commits
    # a pot-sized-ish raise (to_call + 0.75*pot is the population's typical sizing)
    A = A.with_columns(((pl.col('p_pass') * pl.col('to_call')) + pl.col('p_agg') * (pl.col('to_call') + 0.75 * pl.col('pot_before'))).alias('exp_chips'))
    A = A.with_columns(((pl.col('amount') - pl.col('exp_chips')) / bb).alias('over_commit'),          # + = put in more than policy
                       ((pl.col('exp_chips') - pl.col('amount')) / bb).alias('under_commit'),         # + = put in less (passive/fold)
                       (pl.col('pot_before') / bb).alias('pot_bb2'))
    part = S.select('hidx', pl.col('pidx').alias('b'), pl.col('fold_no').alias('b_fold_no'), pl.col('net_chips').alias('b_net'), pl.col('big_blind').alias('bb2'))
    AP = (A.rename({'pidx': 'a'}).join(part, on='hidx').filter(pl.col('b') != pl.col('a'))
          .filter(pl.col('b_fold_no').is_null() | (pl.col('b_fold_no') > pl.col('action_no'))))
    facing_b = (pl.col('last_agg_pidx') == pl.col('b')).fill_null(False)
    hu = pl.col('players_active') == 2
    G = AP.group_by(['hidx', 'a', 'b']).agg(
        pl.when(facing_b).then(pl.col('over_commit')).otherwise(0.0).sum().alias('over_to_partner'),
        pl.when(facing_b).then(pl.col('under_commit')).otherwise(0.0).sum().alias('under_to_partner'),
        pl.when(hu).then(pl.col('under_commit')).otherwise(0.0).sum().alias('under_hu'),
        pl.when(hu).then(pl.col('over_commit')).otherwise(0.0).sum().alias('over_hu'),
        pl.when(~facing_b & (pl.col('players_active') > 2)).then(pl.col('over_commit')).otherwise(0.0).sum().alias('over_vs_third'),
        pl.col('over_commit').sum().alias('over_all'), pl.col('under_commit').sum().alias('under_all'),
        pl.when(pl.col('y') == 0).then(pl.col('pot_bb2')).otherwise(0.0).sum().alias('pot_given_up'))
    RF = [c for c in G.columns if c not in ('hidx', 'a', 'b')]
    sp = S.select('hidx', 'pidx', 'net_chips', 'total_contribution')
    base = (sp.rename({'pidx': 'p', 'net_chips': 'net_p', 'total_contribution': 'c_p'})
            .join(sp.select('hidx', pl.col('pidx').alias('q'), pl.col('net_chips').alias('net_q'), pl.col('total_contribution').alias('c_q')), on='hidx')
            .filter(pl.col('p') < pl.col('q')))
    D = (base.join(G.rename({'a': 'p', 'b': 'q'}).rename({c: 'P_' + c for c in RF}), on=['hidx', 'p', 'q'], how='left')
             .join(G.rename({'a': 'q', 'b': 'p'}).rename({c: 'Q_' + c for c in RF}), on=['hidx', 'p', 'q'], how='left'))
    w = (pl.col('net_p') > pl.col('net_q')) | ((pl.col('net_p') == pl.col('net_q')) & (pl.col('c_p') <= pl.col('c_q')))
    ex = []
    for c in RF:
        pc, qc = pl.col('P_' + c).fill_null(0), pl.col('Q_' + c).fill_null(0)
        ex += [pl.when(w).then(pc).otherwise(qc).cast(pl.Float32).alias('vW_' + c), pl.when(w).then(qc).otherwise(pc).cast(pl.Float32).alias('vL_' + c)]
    D = D.select('hidx', 'p', 'q', *ex)
    D.write_parquet(VD + f'chunk{ch:02d}.parquet')
    log('chunk', ch, D.shape)
log('done')
