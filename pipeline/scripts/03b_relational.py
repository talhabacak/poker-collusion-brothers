"""Step 3b: relational likelihood ratio. For every action of player a taken while player b is still in the hand, compare
a's action distribution in that context *against b* with a's own distribution in the same context against everyone else.
Context = street group x own strength x strength vs b x who a is facing (none/self/b/third) x heads-up.
Hand-level: log p_ab(action|ctx) - log p_a,others(action|ctx), with the current action left out of p_ab (so only repeated
pair-specific behaviour scores). Pair-level: G-like statistic per direction."""
import polars as pl, numpy as np, time, os
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; RD = OUT + 'relational/'; os.makedirs(RD, exist_ok=True)
NCH = 10; ALPHA = 3.0; BETA = 10.0
seats = pl.read_parquet(OUT + 'seats_feat.parquet')
acts = pl.read_parquet(OUT + 'actions_scored.parquet')

pair_level = []
for ch in range(NCH):
    S = seats.filter(pl.col('tidx') % NCH == ch)
    A = acts.filter(pl.col('tidx') % NCH == ch)
    part = S.select('hidx', pl.col('pidx').alias('b'), pl.col('fold_no').alias('b_fold_no'), pl.col('eq_pf').alias('b_eq'),
                    pl.col('hs1').alias('b_hs1'), pl.col('hs2').alias('b_hs2'), pl.col('hs3').alias('b_hs3'))
    AP = (A.rename({'pidx': 'a'}).join(part, on='hidx').filter(pl.col('b') != pl.col('a'))
          .filter(pl.col('b_fold_no').is_null() | (pl.col('b_fold_no') > pl.col('action_no'))))
    pre = pl.col('st') == 0
    AP = AP.with_columns(
        pl.when(pre).then(pl.col('eq_pf')).otherwise(pl.col('hs')).alias('own'),
        pl.when(pre).then(pl.col('b_eq')).when(pl.col('st') == 1).then(pl.col('b_hs1')).when(pl.col('st') == 2).then(pl.col('b_hs2')).otherwise(pl.col('b_hs3')).alias('bstr'))
    AP = AP.with_columns(
        pl.col('st').clip(0, 2).alias('c_st'),
        pl.when(pre).then(pl.when(pl.col('own') < 0.45).then(0).when(pl.col('own') < 0.56).then(1).otherwise(2))
          .otherwise(pl.when(pl.col('own') < 0.17).then(0).when(pl.col('own') < 0.55).then(1).otherwise(2)).fill_null(1).alias('c_own'),
        pl.when(pre).then(pl.when(pl.col('own') - pl.col('bstr') < -0.08).then(0).when(pl.col('own') - pl.col('bstr') > 0.08).then(2).otherwise(1))
          .otherwise(pl.when(pl.col('own') - pl.col('bstr') < -0.1).then(0).when(pl.col('own') - pl.col('bstr') > 0.1).then(2).otherwise(1)).fill_null(1).alias('c_rel'),
        pl.when(pl.col('last_agg_pidx').is_null()).then(0).when(pl.col('last_agg_pidx') == pl.col('a')).then(1)
          .when(pl.col('last_agg_pidx') == pl.col('b')).then(2).otherwise(3).alias('c_face'),
        (pl.col('players_active') == 2).cast(pl.Int32).alias('c_hu'),
        pl.when(pl.col('y') == 0).then(0).when(pl.col('y') <= 2).then(1).otherwise(2).alias('act'))
    AP = AP.with_columns(((((pl.col('c_st') * 3 + pl.col('c_own')) * 3 + pl.col('c_rel')) * 4 + pl.col('c_face')) * 2 + pl.col('c_hu')).alias('ctx'))
    AP = AP.select('hidx', 'tidx', 'is_eval', 'a', 'b', 'ctx', 'act', 'c_face', 'c_hu', 'c_st', 'chips' if 'chips' in AP.columns else (pl.col('amount') / pl.col('big_blind')).alias('chips'))
    g_ab = ['a', 'b', 'is_eval', 'ctx']
    C_ab = AP.group_by(g_ab + ['act']).agg(pl.len().alias('n_ab_act'))
    T_ab = AP.group_by(g_ab).agg(pl.len().alias('n_ab'))
    C_a = AP.group_by(['a', 'is_eval', 'ctx', 'act']).agg(pl.len().alias('n_a_act'))
    T_a = AP.group_by(['a', 'is_eval', 'ctx']).agg(pl.len().alias('n_a'))
    P0 = AP.group_by(['ctx', 'act']).agg(pl.len().alias('n0')).with_columns((pl.col('n0') / pl.col('n0').sum().over('ctx')).alias('p0'))
    # cell table: all (a,b,phase,ctx,act) where the pair has at least one opportunity in ctx; include acts never taken (count 0)
    cells = T_ab.join(pl.DataFrame({'act': [0, 1, 2]}, schema={'act': AP.schema['act']}), how='cross')
    cells = (cells.join(C_ab, on=g_ab + ['act'], how='left').with_columns(pl.col('n_ab_act').fill_null(0))
             .join(C_a, on=['a', 'is_eval', 'ctx', 'act'], how='left').join(T_a, on=['a', 'is_eval', 'ctx'], how='left')
             .join(P0.select('ctx', 'act', 'p0'), on=['ctx', 'act'], how='left')
             .with_columns(pl.col('n_a_act').fill_null(0), pl.col('p0').fill_null(1e-3)))
    cells = cells.with_columns(
        ((pl.col('n_a_act') - pl.col('n_ab_act') + BETA * pl.col('p0')) / (pl.col('n_a') - pl.col('n_ab') + BETA)).alias('p_a'))
    cells = cells.with_columns(((pl.col('n_ab_act') + ALPHA * pl.col('p_a')) / (pl.col('n_ab') + ALPHA)).alias('p_ab'))
    # pair-level statistic per direction: sum over cells n_ab_act * log(p_ab/p_a)  (smoothed, in-sample)
    G = cells.group_by(['a', 'b', 'is_eval']).agg(
        (pl.col('n_ab_act') * (pl.col('p_ab') / pl.col('p_a')).log()).sum().alias('G'),
        pl.col('n_ab').sum().alias('G_opp'))
    pair_level.append(G)
    # action-level leave-one-out LLR
    AP = AP.join(cells.select(g_ab + ['act', 'n_ab_act', 'n_ab', 'p_a']), on=g_ab + ['act'], how='left')
    AP = AP.with_columns(((((pl.col('n_ab_act') - 1).clip(0) + ALPHA * pl.col('p_a')) / ((pl.col('n_ab') - 1).clip(0) + ALPHA)) / pl.col('p_a')).log().cast(pl.Float32).alias('llr'))
    H = AP.group_by(['hidx', 'a', 'b']).agg(
        pl.col('llr').sum().alias('llr_sum'), pl.col('llr').max().alias('llr_max'), pl.col('llr').clip(0).sum().alias('llr_pos'),
        pl.when(pl.col('c_face') == 2).then(pl.col('llr')).otherwise(0.0).sum().alias('llr_face_b'),
        pl.when(pl.col('c_face') == 3).then(pl.col('llr')).otherwise(0.0).sum().alias('llr_face_x'),
        pl.when(pl.col('c_hu') == 1).then(pl.col('llr')).otherwise(0.0).sum().alias('llr_hu'),
        pl.when(pl.col('c_st') > 0).then(pl.col('llr')).otherwise(0.0).sum().alias('llr_post'),
        pl.when(pl.col('act') == 0).then(pl.col('llr')).otherwise(0.0).sum().alias('llr_fold'),
        pl.when(pl.col('act') == 1).then(pl.col('llr')).otherwise(0.0).sum().alias('llr_pass'),
        pl.when(pl.col('act') == 2).then(pl.col('llr')).otherwise(0.0).sum().alias('llr_agg'))
    RF = [c for c in H.columns if c.startswith('llr')]
    # to unordered pair rows with W/L roles (same rule as step 3)
    sp = S.select('hidx', 'pidx', 'net_chips', 'total_contribution')
    base = (sp.rename({'pidx': 'p', 'net_chips': 'net_p', 'total_contribution': 'c_p'})
            .join(sp.select('hidx', pl.col('pidx').alias('q'), pl.col('net_chips').alias('net_q'), pl.col('total_contribution').alias('c_q')), on='hidx')
            .filter(pl.col('p') < pl.col('q')))
    D = (base.join(H.rename({'a': 'p', 'b': 'q'}).rename({c: 'P_' + c for c in RF}), on=['hidx', 'p', 'q'], how='left')
             .join(H.rename({'a': 'q', 'b': 'p'}).rename({c: 'Q_' + c for c in RF}), on=['hidx', 'p', 'q'], how='left'))
    w = (pl.col('net_p') > pl.col('net_q')) | ((pl.col('net_p') == pl.col('net_q')) & (pl.col('c_p') <= pl.col('c_q')))
    ex = []
    for c in RF:
        pc, qc = pl.col('P_' + c).fill_null(0), pl.col('Q_' + c).fill_null(0)
        ex += [pl.when(w).then(pc).otherwise(qc).cast(pl.Float32).alias('W_' + c), pl.when(w).then(qc).otherwise(pc).cast(pl.Float32).alias('L_' + c)]
    ex += [(pl.col('P_llr_sum').fill_null(0) + pl.col('Q_llr_sum').fill_null(0)).cast(pl.Float32).alias('rel_llr_tot'),
           pl.max_horizontal(pl.col('P_llr_max').fill_null(0), pl.col('Q_llr_max').fill_null(0)).cast(pl.Float32).alias('rel_llr_max'),
           (pl.col('P_llr_pos').fill_null(0) + pl.col('Q_llr_pos').fill_null(0)).cast(pl.Float32).alias('rel_llr_pos')]
    D = D.select('hidx', 'p', 'q', *ex)
    D.write_parquet(RD + f'chunk{ch:02d}.parquet')
    log('chunk', ch, 'action-partner rows', AP.height, 'pair-hands', D.height)

G = pl.concat(pair_level)
Gp = G.with_columns(pl.min_horizontal('a', 'b').alias('p'), pl.max_horizontal('a', 'b').alias('q'))
Gp = Gp.group_by(['p', 'q', 'is_eval']).agg(pl.col('G').max().alias('G_max'), pl.col('G').min().alias('G_min'), pl.col('G').sum().alias('G_sum'),
                                             pl.col('G_opp').sum().alias('G_opp'))
Gp = Gp.with_columns((pl.col('G_sum') / pl.col('G_opp')).alias('G_rate'))
Gp.write_parquet(OUT + 'pair_relational.parquet')
log('done', Gp.shape)
