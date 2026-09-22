"""Step 7: information-sharing detector (candidate mechanism for other_coordination).
Train a second policy model that also sees ONE other seated player's hole cards. For an honest player those cards
carry no information about their own action; for a pair that shares information they do. Score per ordered pair =
mean log-likelihood gain of the partner-aware model over the blind model, on that player's actions while the partner
is dealt in. Compared against the same player's gain with every other opponent, so style is cancelled out."""
import polars as pl, numpy as np, lightgbm as lgb, time, os
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; SEED = 42
BASE = ['st', 'relpos', 'hi', 'lo', 'suited', 'pocket', 'gap', 'eq_pf', 'hs', 'hc', 'potodds', 'tocall_bb', 'pot_bb', 'stack_bb', 'tocall_stack', 'spr',
        'players_active', 'players_dealt', 'n_agg_hand', 'n_agg_st', 'own_agg_before', 'n_act_st', 'self_last_agg', 'big_blind', 'action_no',
        'ps_n', 'ps_open_raise', 'ps_open_fold', 'ps_vs_raise_agg', 'ps_vs_raise_fold', 'ps_post_agg', 'ps_post_fold_vs_bet']
EXTRA = ['b_eq', 'b_str', 'b_rel', 'b_folded', 'b_relpos', 'b_active']
A = pl.read_parquet(OUT + 'actions_feat.parquet')
S = pl.read_parquet(OUT + 'seats_feat.parquet')
rng = np.random.default_rng(SEED)
# one row per (action, other seated player b): keep a random subsample for training, everything for scoring is too big
part = S.select('hidx', pl.col('pidx').alias('b'), pl.col('eq_pf').alias('b_eq'), pl.col('hs1').alias('b_hs1'), pl.col('hs2').alias('b_hs2'), pl.col('hs3').alias('b_hs3'),
                pl.col('fold_no').alias('b_fold_no'), pl.col('relpos').alias('b_relpos'))

def build_chunk(ch):
    Ac = A.filter(pl.col('tidx') % NCH == ch)
    AP = Ac.rename({'pidx': 'a'}).join(part, on='hidx').filter(pl.col('a') != pl.col('b'))
    AP = AP.with_columns(
        pl.when(pl.col('st') == 0).then(pl.col('b_eq')).when(pl.col('st') == 1).then(pl.col('b_hs1')).when(pl.col('st') == 2).then(pl.col('b_hs2')).otherwise(pl.col('b_hs3')).alias('b_str'),
        (pl.col('b_fold_no').is_null() | (pl.col('b_fold_no') > pl.col('action_no'))).cast(pl.Int8).alias('b_active'),
        pl.col('b_fold_no').is_not_null().cast(pl.Int8).alias('b_folded'))
    AP = AP.with_columns((pl.when(pl.col('st') == 0).then(pl.col('eq_pf')).otherwise(pl.col('hs')) - pl.col('b_str')).alias('b_rel'))
    return AP.select('hidx', 'tidx', 'is_eval', 'a', 'b', 'y', *BASE, *EXTRA)

NCH = 20
pr = dict(objective='multiclass', num_class=6, learning_rate=0.08, num_leaves=127, min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.7,
          bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16, seed=SEED)
models = {}
for parity, src_ch in [(0, 1), (1, 0)]:   # model used on tables with tidx%2==parity is trained on the other parity
    T = build_chunk(src_ch)
    idx = rng.permutation(T.height)[:3_000_000]
    yt = T['y'].to_numpy()[idx]
    mb = lgb.train(pr, lgb.Dataset(T.select(BASE).to_numpy().astype(np.float32)[idx], yt), 300)
    me = lgb.train(pr, lgb.Dataset(T.select(BASE + EXTRA).to_numpy().astype(np.float32)[idx], yt), 300)
    models[parity] = (mb, me)
    log('trained models for parity', parity, 'rows', len(idx))
    del T

parts = []
for ch in range(NCH):
    AP = build_chunk(ch)
    mb, me = models[ch % 2]
    Xb = AP.select(BASE).to_numpy().astype(np.float32); Xe = AP.select(BASE + EXTRA).to_numpy().astype(np.float32)
    yv = AP['y'].to_numpy(); i = np.arange(len(yv))
    g = (np.log(np.clip(me.predict(Xe)[i, yv], 1e-6, 1)) - np.log(np.clip(mb.predict(Xb)[i, yv], 1e-6, 1))).astype(np.float32)
    AP = AP.select('a', 'b', 'is_eval', 'b_active').with_columns(pl.Series('gain', g))
    parts.append(AP.group_by(['a', 'b', 'is_eval']).agg(pl.col('gain').sum().alias('s_ab'), pl.len().alias('n_ab'),
                                                        pl.when(pl.col('b_active') == 1).then(pl.col('gain')).otherwise(None).mean().alias('g_ab_active')))
    log('scored chunk', ch, 'rows', len(yv), 'mean gain', float(g.mean()))
    del AP, Xb, Xe, g

G = pl.concat(parts).group_by(['a', 'b', 'is_eval']).agg(pl.col('s_ab').sum(), pl.col('n_ab').sum(), pl.col('g_ab_active').mean())
G = G.with_columns((pl.col('s_ab') / pl.col('n_ab')).alias('g_ab'))
Ga = G.group_by(['a', 'is_eval']).agg(pl.col('s_ab').sum().alias('s_a'), pl.col('n_ab').sum().alias('n_a'))
G = G.join(Ga, on=['a', 'is_eval']).with_columns(((pl.col('s_a') - pl.col('s_ab')) / (pl.col('n_a') - pl.col('n_ab'))).alias('g_a_other'))
G = G.with_columns((pl.col('g_ab') - pl.col('g_a_other')).alias('excess'))
Gp = G.with_columns(pl.min_horizontal('a', 'b').alias('p'), pl.max_horizontal('a', 'b').alias('q')).group_by(['p', 'q', 'is_eval']).agg(
    pl.col('excess').max().alias('info_excess_max'), pl.col('excess').min().alias('info_excess_min'), pl.col('excess').mean().alias('info_excess_mean'),
    pl.col('g_ab_active').max().alias('info_active_max'))
Gp.write_parquet(OUT + 'pair_infoshare.parquet')
log('done', Gp.shape)
