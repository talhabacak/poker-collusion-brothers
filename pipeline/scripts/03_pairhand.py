"""Step 3: features for every (pair, shared hand), all 15 pairs per hand, both phases. Written in table chunks."""
import polars as pl, numpy as np, time, os, sys
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; PH = OUT + 'pairhand/'; os.makedirs(PH, exist_ok=True)
NCH = 20
seats = pl.read_parquet(OUT + 'seats_feat.parquet')
hands = pl.read_parquet(OUT + 'hands.parquet').select('hidx', 't_order', 'final_pot', 'board_len', 'players_at_showdown')
acts = pl.read_parquet(OUT + 'actions_scored.parquet')

surp = pl.col('surp'); y = pl.col('y')
f_fold = y == 0; f_pass = y.is_in([1, 2]); f_agg = y >= 3
facing = (pl.col('last_agg_pidx') == pl.col('b')).fill_null(False)
chips = pl.col('amount') / pl.col('big_blind'); potbb = pl.col('pot_before') / pl.col('big_blind')
rel = pl.col('rel')
def s_when(cond, val): return pl.when(cond).then(val).otherwise(0.0).sum()
AGG = [
    pl.len().alias('n_act'), surp.sum().alias('surp_sum'), surp.max().alias('surp_max'),
    s_when(f_fold, surp).alias('surp_fold'), s_when(f_pass, surp).alias('surp_pass'), s_when(f_agg, surp).alias('surp_agg'),
    s_when(facing, surp).alias('surp_facing'), (f_fold & facing).sum().alias('n_fold_facing'),
    pl.when(f_fold & facing).then(rel).otherwise(None).max().alias('rel_fold_facing'),
    pl.when(f_fold & facing).then(pl.col('own_str')).otherwise(None).max().alias('str_fold_facing'),
    pl.when(f_fold).then(pl.col('own_str')).otherwise(None).max().alias('str_fold'),
    s_when(rel < 0, chips).alias('chips_behind'), s_when((rel < 0) & facing, chips).alias('chips_behind_facing'),
    (chips * (-rel)).sum().alias('dump_value'),
    s_when(f_pass & (rel > 0), surp).alias('surp_pass_ahead'), (f_pass & (rel > 0)).sum().alias('n_pass_ahead'),
    s_when(f_pass & (rel > 0), potbb * rel).alias('soft_value'),
    s_when(f_fold, potbb * rel).alias('fold_value'),
    (f_agg & (pl.col('players_active') > 2)).sum().alias('n_agg_multi'), s_when(f_agg & (pl.col('players_active') > 2), surp).alias('surp_agg_multi'),
    s_when(pl.col('players_active') == 2, surp).alias('surp_hu'),
    pl.when(f_agg).then(pl.col('size_z')).otherwise(None).max().alias('size_z_max'),
    pl.when(f_agg).then(pl.col('size_z')).otherwise(None).min().alias('size_z_min'),
    (y == 5).sum().alias('n_allin'), chips.sum().alias('chips_in'),
    s_when(pl.col('st') > 0, surp).alias('surp_post'),
]
ROLE_FEATS = ['n_act', 'surp_sum', 'surp_max', 'surp_fold', 'surp_pass', 'surp_agg', 'surp_facing', 'n_fold_facing', 'rel_fold_facing', 'str_fold_facing', 'str_fold',
              'chips_behind', 'chips_behind_facing', 'dump_value', 'surp_pass_ahead', 'n_pass_ahead', 'soft_value', 'fold_value', 'n_agg_multi', 'surp_agg_multi',
              'surp_hu', 'size_z_max', 'size_z_min', 'n_allin', 'chips_in', 'surp_post']

for ch in range(NCH):
    S = seats.filter(pl.col('tidx') % NCH == ch)
    A = acts.filter(pl.col('tidx') % NCH == ch)
    part = S.select('hidx', pl.col('pidx').alias('b'), pl.col('fold_no').alias('b_fold_no'), pl.col('eq_pf').alias('b_eq'), pl.col('hs1').alias('b_hs1'), pl.col('hs2').alias('b_hs2'), pl.col('hs3').alias('b_hs3'))
    AP = (A.rename({'pidx': 'a'}).join(part, on='hidx').filter(pl.col('b') != pl.col('a'))
          .filter(pl.col('b_fold_no').is_null() | (pl.col('b_fold_no') > pl.col('action_no'))))
    AP = AP.with_columns(
        pl.when(pl.col('st') == 0).then(pl.col('eq_pf')).otherwise(pl.col('hs')).alias('own_str'),
        pl.when(pl.col('st') == 0).then(pl.col('b_eq')).when(pl.col('st') == 1).then(pl.col('b_hs1')).when(pl.col('st') == 2).then(pl.col('b_hs2')).otherwise(pl.col('b_hs3')).alias('b_str'))
    AP = AP.with_columns((pl.col('own_str') - pl.col('b_str')).fill_null(0.0).alias('rel'))
    AG = AP.group_by(['hidx', 'a', 'b']).agg(AGG)
    # third-party folds facing aggression from a pair member while other member active
    T = (A.filter((y == 0) & pl.col('last_agg_pidx').is_not_null()).select('hidx', 'action_no', pl.col('pidx').alias('actor'), pl.col('last_agg_pidx').alias('r'))
         .join(part.select('hidx', 'b', 'b_fold_no'), on='hidx')
         .filter((pl.col('b') != pl.col('actor')) & (pl.col('b') != pl.col('r')) & (pl.col('b_fold_no').is_null() | (pl.col('b_fold_no') > pl.col('action_no'))))
         .with_columns(pl.min_horizontal('r', 'b').alias('p'), pl.max_horizontal('r', 'b').alias('q'))
         .group_by(['hidx', 'p', 'q']).agg(pl.len().alias('n_third_fold_vs_pair')))
    # pair base
    sp = S.select('hidx', 'pidx', 'tidx', 'is_eval', 'big_blind', 'net_chips', 'total_contribution', 'folded', 'went_to_showdown')
    base = (sp.rename({'pidx': 'p', 'net_chips': 'net_p', 'total_contribution': 'c_p', 'folded': 'f_p', 'went_to_showdown': 'sd_p'})
            .join(sp.select('hidx', pl.col('pidx').alias('q'), pl.col('net_chips').alias('net_q'), pl.col('total_contribution').alias('c_q'), pl.col('folded').alias('f_q'), pl.col('went_to_showdown').alias('sd_q')), on='hidx')
            .filter(pl.col('p') < pl.col('q')).join(hands, on='hidx'))
    Pp = AG.rename({'a': 'p', 'b': 'q'}).rename({c: 'P_' + c for c in ROLE_FEATS})
    Pq = AG.rename({'a': 'q', 'b': 'p'}).rename({c: 'Q_' + c for c in ROLE_FEATS})
    D = base.join(Pp, on=['hidx', 'p', 'q'], how='left').join(Pq, on=['hidx', 'p', 'q'], how='left').join(T, on=['hidx', 'p', 'q'], how='left')
    w_is_p = (pl.col('net_p') > pl.col('net_q')) | ((pl.col('net_p') == pl.col('net_q')) & (pl.col('c_p') <= pl.col('c_q')))
    D = D.with_columns(w_is_p.alias('w_is_p'))
    fill = {c: (None if c in ('rel_fold_facing', 'str_fold_facing', 'str_fold', 'size_z_max', 'size_z_min') else 0.0) for c in ROLE_FEATS}
    exprs = []
    for c in ROLE_FEATS:
        pc, qc = pl.col('P_' + c), pl.col('Q_' + c)
        if fill[c] is not None: pc, qc = pc.fill_null(0), qc.fill_null(0)
        exprs += [pl.when(pl.col('w_is_p')).then(pc).otherwise(qc).cast(pl.Float32).alias('W_' + c), pl.when(pl.col('w_is_p')).then(qc).otherwise(pc).cast(pl.Float32).alias('L_' + c)]
    bb = pl.col('big_blind')
    D = D.with_columns(exprs + [
        (pl.col('final_pot') / bb).cast(pl.Float32).alias('pot_bb'),
        ((pl.col('net_p') - pl.col('net_q')).abs() / bb).cast(pl.Float32).alias('flow_bb'),
        ((pl.col('net_p') + pl.col('net_q')) / bb).cast(pl.Float32).alias('pair_net_bb'),
        (pl.max_horizontal('net_p', 'net_q') / bb).cast(pl.Float32).alias('W_net_bb'),
        (pl.min_horizontal('net_p', 'net_q') / bb).cast(pl.Float32).alias('L_net_bb'),
        ((pl.col('c_p') > bb) & (pl.col('c_q') > bb)).cast(pl.Int8).alias('both_vpip'),
        (pl.col('sd_p') & pl.col('sd_q')).cast(pl.Int8).alias('both_sd'),
        (pl.col('f_p') ^ pl.col('f_q')).cast(pl.Int8).alias('one_fold'),
        (~pl.col('f_p') & ~pl.col('f_q')).cast(pl.Int8).alias('neither_fold'),
        ((pl.col('net_p') > 0) & (pl.col('net_q') < 0) | (pl.col('net_q') > 0) & (pl.col('net_p') < 0)).cast(pl.Int8).alias('opp_sign'),
        pl.col('n_third_fold_vs_pair').fill_null(0).cast(pl.Float32),
        (pl.max_horizontal('c_p', 'c_q') / bb).cast(pl.Float32).alias('max_contrib_bb'),
    ]).drop([c for c in D.columns if c.startswith('P_') or c.startswith('Q_')] + ['w_is_p', 'net_p', 'net_q', 'c_p', 'c_q', 'f_p', 'f_q', 'sd_p', 'sd_q', 'final_pot'])
    D = D.with_columns((pl.col('W_surp_sum') + pl.col('L_surp_sum')).alias('tot_surp'), pl.max_horizontal('W_surp_max', 'L_surp_max').alias('tot_surp_max'),
                       (pl.col('W_surp_facing') + pl.col('L_surp_facing')).alias('tot_surp_facing'))
    # within-pair context (same phase): standardized surprisal and local episode density
    D = D.sort(['p', 'q', 'is_eval', 't_order'])
    g = ['p', 'q', 'is_eval']
    D = D.with_columns(pl.len().over(g).cast(pl.Int32).alias('pair_n'),
                       ((pl.col('tot_surp') - pl.col('tot_surp').mean().over(g)) / (pl.col('tot_surp').std().over(g) + 1e-3)).cast(pl.Float32).alias('tot_surp_z'),
                       (pl.col('tot_surp').rank('average').over(g) / pl.len().over(g)).cast(pl.Float32).alias('tot_surp_pct'),
                       pl.col('tot_surp').rolling_mean(21, center=True, min_samples=1).over(g).cast(pl.Float32).alias('roll21_surp'),
                       pl.col('tot_surp').rolling_mean(61, center=True, min_samples=1).over(g).cast(pl.Float32).alias('roll61_surp'),
                       (pl.col('t_order').rank('ordinal').over(g) / pl.len().over(g)).cast(pl.Float32).alias('pair_tpos'))
    D.write_parquet(PH + f'chunk{ch:02d}.parquet')
    log('chunk', ch, D.shape)
log('done')
