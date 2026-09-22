"""Step 15: the opponent-conditioned policy, and the residuals it makes possible.

Step 13 asked "while B was live, did A depart from what the policy expected in those spots" -- but the
policy it used does not know B is there, so its expectation is the same whoever is sitting across. The
departure therefore still contains everything that is merely "this opponent is not like the others",
which is legitimate poker, not collusion.

Here the expectation itself is conditioned on the opponent: one training row per (action, live opponent),
carrying who the actor is answering, where that opponent sits relative to the actor, how many opponents
are still live, and the opponent's own six style rates. What is left after subtracting THAT expectation
is the part of A's behaviour towards B that neither the spot nor B's playing style explains.

This is the piece step 13 approximated. No label is read: the target is the action taken.

Writes oppcond_pair.parquet and oppcond_hand.parquet.
"""
import polars as pl, numpy as np, lightgbm as lgb, time, os
from scipy import stats
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; NCH = int(os.environ.get('OC_CHUNKS', '20')); SEED = 42; ALPHA = 0.5
TRAIN_ROWS = int(os.environ.get('OC_TRAIN', '3000000'))

BASE = ['st', 'relpos', 'hi', 'lo', 'suited', 'pocket', 'gap', 'eq_pf', 'hs', 'hc', 'potodds', 'tocall_bb', 'pot_bb',
        'stack_bb', 'tocall_stack', 'spr', 'players_active', 'players_dealt', 'n_agg_hand', 'n_agg_st',
        'own_agg_before', 'n_act_st', 'self_last_agg', 'big_blind', 'action_no',
        'ps_n', 'ps_open_raise', 'ps_open_fold', 'ps_vs_raise_agg', 'ps_vs_raise_fold', 'ps_post_agg', 'ps_post_fold_vs_bet']
OPP = ['answering', 'opp_relpos_diff', 'n_live', 'o_ps_open_raise', 'o_ps_open_fold', 'o_ps_vs_raise_agg',
       'o_ps_vs_raise_fold', 'o_ps_post_agg', 'o_ps_post_fold_vs_bet']
FEATS = BASE + OPP

A = pl.read_parquet(OUT + 'actions_feat.parquet').select('hidx', 'tidx', 'is_eval', 'pidx', 'y', 'last_agg_pidx', *BASE)
S = pl.read_parquet(OUT + 'seats_feat.parquet').select('hidx', 'pidx', 'tidx', 'fold_no', 'relpos')
PS = pl.read_parquet(OUT + 'actions_feat.parquet').select(
    'pidx', 'is_eval', 'ps_open_raise', 'ps_open_fold', 'ps_vs_raise_agg', 'ps_vs_raise_fold', 'ps_post_agg', 'ps_post_fold_vs_bet').unique(['pidx', 'is_eval'])
PS = PS.rename({c: 'o_' + c for c in PS.columns if c.startswith('ps_')}).rename({'pidx': 'opp'})
log('actions', A.shape, 'styles', PS.shape)

def expand(ch):
    a = A.filter(pl.col('tidx') % NCH == ch)
    s = S.filter(pl.col('tidx') % NCH == ch).select('hidx', pl.col('pidx').alias('opp'), 'fold_no', pl.col('relpos').alias('o_relpos'))
    X = a.join(s, on='hidx').filter(
        (pl.col('opp') != pl.col('pidx')) &
        (pl.col('fold_no').is_null() | (pl.col('fold_no') > pl.col('action_no'))))
    X = X.join(PS, on=['opp', 'is_eval'], how='left')
    return X.with_columns(
        (pl.col('last_agg_pidx') == pl.col('opp')).cast(pl.Int8).alias('answering'),
        (pl.col('o_relpos') - pl.col('relpos')).alias('opp_relpos_diff'),
        pl.len().over(['hidx', 'action_no']).alias('n_live')).with_columns(
        [pl.col(c).fill_null(0.0) for c in OPP])

# ---- train on a sample, two folds by table parity so nothing is scored by a model that saw its table
rng = np.random.default_rng(SEED)
tr = pl.concat([expand(ch) for ch in range(0, NCH, 3)])   # step 3, not 4: with NCH=20 a step of 4 hits only even tidx and leaves one parity fold empty
log('training pool', tr.shape)
idx = rng.permutation(tr.height)[:TRAIN_ROWS]
tr = tr[idx]
Xtr = tr.select(FEATS).to_numpy().astype(np.float32); ytr = tr['y'].to_numpy(); par = tr['tidx'].to_numpy() % 2
PARAMS = dict(objective='multiclass', num_class=6, learning_rate=0.06, num_leaves=127, min_child_samples=100,
              feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=16, seed=SEED)
models = {}
for p in (0, 1):
    m = lgb.train(PARAMS, lgb.Dataset(Xtr[par != p], ytr[par != p]), 350); models[p] = m
    pr = m.predict(Xtr[par == p]); ll = -np.log(np.clip(pr[np.arange((par == p).sum()), ytr[par == p]], 1e-9, 1)).mean()
    log(f'parity {p}: trained on {(par != p).sum()} rows, held-out logloss {ll:.4f}')
del tr, Xtr, ytr

# ---- score everything, accumulate per directed pair and per hand
pair_parts, hand_parts = [], []
for ch in range(NCH):
    X = expand(ch)
    par = X['tidx'].to_numpy() % 2
    P = np.zeros((X.height, 6), np.float32)
    for p in (0, 1):
        sel = par == p
        if sel.sum(): P[sel] = models[1 - p].predict(X.filter(pl.Series(sel)).select(FEATS).to_numpy().astype(np.float32))
    pf = P[:, 0]; pc = P[:, 1] + P[:, 2]; pa = P[:, 3] + P[:, 4] + P[:, 5]
    y = X['y'].to_numpy()
    of = (y == 0).astype(np.float64); oc = np.isin(y, [1, 2]).astype(np.float64); oa = np.isin(y, [3, 4, 5]).astype(np.float64)
    nll = -np.log(np.clip(of * pf + oc * pc + oa * pa, 1e-9, 1))
    ent = -(pf * np.log(np.clip(pf, 1e-12, 1)) + pc * np.log(np.clip(pc, 1e-12, 1)) + pa * np.log(np.clip(pa, 1e-12, 1)))
    X = X.with_columns([pl.Series(k, v) for k, v in dict(
        _of=of, _oc=oc, _oa=oa, _pf=pf.astype(np.float64), _pc=pc.astype(np.float64), _pa=pa.astype(np.float64),
        _nll=nll, _ent=ent, _fac=X['answering'].to_numpy().astype(np.float64)).items()])
    g = X.group_by(['pidx', 'opp', 'is_eval']).agg(
        pl.len().alias('n'), pl.col('_of').sum().alias('o_f'), pl.col('_oc').sum().alias('o_c'), pl.col('_oa').sum().alias('o_a'),
        pl.col('_pf').sum().alias('e_f'), pl.col('_pc').sum().alias('e_c'), pl.col('_pa').sum().alias('e_a'),
        (pl.col('_pf') * (1 - pl.col('_pf'))).sum().alias('v_f'), (pl.col('_pc') * (1 - pl.col('_pc'))).sum().alias('v_c'),
        (pl.col('_pa') * (1 - pl.col('_pa'))).sum().alias('v_a'),
        pl.col('_nll').sum().alias('nll'), pl.col('_ent').sum().alias('ent'), pl.col('_fac').sum().alias('n_fac'),
        (pl.col('_of') * pl.col('_fac')).sum().alias('of_fac'), (pl.col('_oa') * pl.col('_fac')).sum().alias('oa_fac'),
        (pl.col('_pf') * pl.col('_fac')).sum().alias('ef_fac'), (pl.col('_pa') * pl.col('_fac')).sum().alias('ea_fac'),
        (pl.col('_pf') * (1 - pl.col('_pf')) * pl.col('_fac')).sum().alias('vf_fac'),
        (pl.col('_pa') * (1 - pl.col('_pa')) * pl.col('_fac')).sum().alias('va_fac'))
    pair_parts.append(g)
    h = X.group_by(['hidx', 'pidx', 'opp']).agg(
        pl.col('_nll').sum().alias('h_nll'), pl.col('_nll').max().alias('h_nll_max'),
        (pl.col('_nll') - pl.col('_ent')).sum().alias('h_exc'), (pl.col('_nll') - pl.col('_ent')).max().alias('h_exc_max'),
        ((pl.col('_of') - pl.col('_pf')) + (pl.col('_oa') - pl.col('_pa'))).sum().alias('h_dev'),
        ((pl.col('_nll') - pl.col('_ent')) * pl.col('_fac')).sum().alias('h_exc_fac'), pl.len().alias('h_n'))
    hand_parts.append(h)
    log('chunk', ch, 'rows', X.height)

D = pl.concat(pair_parts).group_by(['pidx', 'opp', 'is_eval']).agg([pl.col(c).sum() for c in pair_parts[0].columns if c not in ('pidx', 'opp', 'is_eval')])
n = D['n'].to_numpy().astype(np.float64); nn = np.maximum(n, 1.0)
obs = np.stack([D['o_f'].to_numpy(), D['o_c'].to_numpy(), D['o_a'].to_numpy()], 1).astype(np.float64)
exp = np.stack([D['e_f'].to_numpy(), D['e_c'].to_numpy(), D['e_a'].to_numpy()], 1).astype(np.float64)
den = (n + ALPHA * 3)[:, None]; p_obs = (obs + ALPHA) / den; p_mod = (exp + ALPHA) / den
g = 2.0 * n * (p_obs * np.log(p_obs / p_mod)).sum(1)
tail = -stats.chi2.logsf(np.maximum(g, 0.0), 2) / np.log(10.0)
z = lambda o, e, v: (o - e) / np.sqrt(np.maximum(v, 1.0))
nf = np.maximum(D['n_fac'].to_numpy().astype(np.float64), 1.0)
F = D.select('pidx', 'opp', 'is_eval').with_columns([pl.Series(k, v.astype(np.float32)) for k, v in {
    'oc_g': g, 'oc_g_tail': tail, 'oc_g_per': g / nn,
    'oc_fold_z': z(obs[:, 0], exp[:, 0], D['v_f'].to_numpy()), 'oc_call_z': z(obs[:, 1], exp[:, 1], D['v_c'].to_numpy()),
    'oc_aggr_z': z(obs[:, 2], exp[:, 2], D['v_a'].to_numpy()),
    'oc_fold_dev': (obs[:, 0] - exp[:, 0]) / nn, 'oc_aggr_dev': (obs[:, 2] - exp[:, 2]) / nn,
    'oc_surp': D['nll'].to_numpy() / nn, 'oc_surp_excess': (D['nll'].to_numpy() - D['ent'].to_numpy()) / nn,
    'oc_fold_z_fac': z(D['of_fac'].to_numpy(), D['ef_fac'].to_numpy(), D['vf_fac'].to_numpy()),
    'oc_aggr_z_fac': z(D['oa_fac'].to_numpy(), D['ea_fac'].to_numpy(), D['va_fac'].to_numpy()),
    'oc_n': n,
}.items()])
CF = [c for c in F.columns if c.startswith('oc_')]
F = F.with_columns(pl.min_horizontal('pidx', 'opp').alias('p'), pl.max_horizontal('pidx', 'opp').alias('q'))
PR = F.group_by(['p', 'q', 'is_eval']).agg(
    [pl.col(c).max().alias(c + '_hi') for c in CF] + [pl.col(c).min().alias(c + '_lo') for c in CF] +
    [pl.col('oc_g_tail').sum().alias('oc_g_tail_sum'), pl.col('oc_surp_excess').sum().alias('oc_exc_sum')])
PR.write_parquet(OUT + 'oppcond_pair.parquet'); log('written pair', PR.shape)

HH = pl.concat(hand_parts).with_columns(pl.min_horizontal('pidx', 'opp').alias('p'), pl.max_horizontal('pidx', 'opp').alias('q'))
HC = [c for c in HH.columns if c.startswith('h_')]
HP = HH.group_by(['hidx', 'p', 'q']).agg(
    [pl.col(c).sum().alias('och_' + c + '_s') for c in HC] + [pl.col(c).max().alias('och_' + c + '_m') for c in HC])
HP = HP.with_columns([pl.col(c).cast(pl.Float32) for c in HP.columns if c.startswith('och_')])
HP.write_parquet(OUT + 'oppcond_hand.parquet'); log('written hand', HP.shape)
