"""Step 13: opponent-aware policy residuals, per directed pair.

This is the one idea from the teammate's pipeline that we never had, and it was
their largest late gain (+0.0087 / +0.0100 on the board, stage 150 there).

Our relational LLR (03b) asks: does A act differently towards B than towards the
rest of the table? Its baseline is A's own behaviour. This asks a different
question: while B was still live in the hand, did A's actions depart from what
the population policy expected *in those spots* -- and it normalises the
departure by the model's own variance, so a run of easy spots cannot fake it.

Counting-based LLR and model-based residual fail differently, which is why both
are kept. No label is read here; this is a deterministic transform of the logs.

Writes data/interim/oppaware_pair.parquet (one row per unordered pair per phase,
each directed statistic folded to hi/lo).
"""
import polars as pl, numpy as np, time, os
from scipy import stats
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; NCH = int(os.environ.get('OA_CHUNKS', '20')); ALPHA = 0.5

S = pl.read_parquet(OUT + 'seats_feat.parquet').select('hidx', 'pidx', 'tidx', 'is_eval', 'fold_no')
A = pl.read_parquet(OUT + 'actions_scored.parquet').select(
    'hidx', 'tidx', 'is_eval', 'action_no', 'pidx', 'y', 'p_fold', 'p_pass', 'p_agg', 'last_agg_pidx', 'surp')
log('actions', A.shape, 'seats', S.shape)

aggs = []
for ch in range(NCH):
    a = A.filter(pl.col('tidx') % NCH == ch)
    s = S.filter(pl.col('tidx') % NCH == ch).select('hidx', pl.col('pidx').alias('opp'), 'fold_no')
    # one row per (action, opponent still live when the action was taken)
    X = a.join(s, on='hidx').filter(
        (pl.col('opp') != pl.col('pidx')) &
        (pl.col('fold_no').is_null() | (pl.col('fold_no') > pl.col('action_no'))))
    pf, pc, pa = pl.col('p_fold'), pl.col('p_pass'), pl.col('p_agg')
    of = (pl.col('y') == 0).cast(pl.Float64)
    oc = pl.col('y').is_in([1, 2]).cast(pl.Float64)
    oa = pl.col('y').is_in([3, 4, 5]).cast(pl.Float64)
    fac = (pl.col('last_agg_pidx') == pl.col('opp')).cast(pl.Float64)   # actor is answering THIS opponent
    nll = -(of * pf.clip(1e-9).log() + oc * pc.clip(1e-9).log() + oa * pa.clip(1e-9).log())
    ent = -(pf * pf.clip(1e-12).log() + pc * pc.clip(1e-12).log() + pa * pa.clip(1e-12).log())
    X = X.with_columns(of.alias('of'), oc.alias('oc'), oa.alias('oa_'), fac.alias('fac'),
                       nll.alias('nll'), ent.alias('ent'))
    g = X.group_by(['pidx', 'opp', 'is_eval']).agg(
        pl.len().alias('n'),
        pl.col('of').sum().alias('o_f'), pl.col('oc').sum().alias('o_c'), pl.col('oa_').sum().alias('o_a'),
        pl.col('p_fold').sum().alias('e_f'), pl.col('p_pass').sum().alias('e_c'), pl.col('p_agg').sum().alias('e_a'),
        (pl.col('p_fold') * (1 - pl.col('p_fold'))).sum().alias('v_f'),
        (pl.col('p_pass') * (1 - pl.col('p_pass'))).sum().alias('v_c'),
        (pl.col('p_agg') * (1 - pl.col('p_agg'))).sum().alias('v_a'),
        pl.col('nll').sum().alias('nll'), pl.col('ent').sum().alias('ent'),
        pl.col('fac').sum().alias('n_fac'),
        (pl.col('of') * pl.col('fac')).sum().alias('of_fac'), (pl.col('oa_') * pl.col('fac')).sum().alias('oa_fac'),
        (pl.col('p_fold') * pl.col('fac')).sum().alias('ef_fac'), (pl.col('p_agg') * pl.col('fac')).sum().alias('ea_fac'),
        (pl.col('p_fold') * (1 - pl.col('p_fold')) * pl.col('fac')).sum().alias('vf_fac'),
        (pl.col('p_agg') * (1 - pl.col('p_agg')) * pl.col('fac')).sum().alias('va_fac'))
    aggs.append(g); log('chunk', ch, 'expanded rows', X.height, '-> directed', g.height)

D = pl.concat(aggs).group_by(['pidx', 'opp', 'is_eval']).agg([pl.col(c).sum() for c in aggs[0].columns if c not in ('pidx', 'opp', 'is_eval')])
log('directed pairs', D.shape)

n = D['n'].to_numpy().astype(np.float64); nn = np.maximum(n, 1.0)
obs = np.stack([D['o_f'].to_numpy(), D['o_c'].to_numpy(), D['o_a'].to_numpy()], 1).astype(np.float64)
exp = np.stack([D['e_f'].to_numpy(), D['e_c'].to_numpy(), D['e_a'].to_numpy()], 1).astype(np.float64)
den = (n + ALPHA * 3)[:, None]
p_obs = (obs + ALPHA) / den; p_mod = (exp + ALPHA) / den
g = 2.0 * n * (p_obs * np.log(p_obs / p_mod)).sum(1)
tail = -stats.chi2.logsf(np.maximum(g, 0.0), 2) / np.log(10.0)
z = lambda o, e, v: (o - e) / np.sqrt(np.maximum(v, 1.0))
nf = np.maximum(D['n_fac'].to_numpy().astype(np.float64), 1.0)
F = D.select('pidx', 'opp', 'is_eval').with_columns([pl.Series(k, v.astype(np.float32)) for k, v in {
    'oa_g': g, 'oa_g_tail': tail, 'oa_g_per': g / nn, 'oa_n': n,
    'oa_fold_dev': (obs[:, 0] - exp[:, 0]) / nn, 'oa_call_dev': (obs[:, 1] - exp[:, 1]) / nn,
    'oa_aggr_dev': (obs[:, 2] - exp[:, 2]) / nn,
    'oa_fold_z': z(obs[:, 0], exp[:, 0], D['v_f'].to_numpy()),
    'oa_call_z': z(obs[:, 1], exp[:, 1], D['v_c'].to_numpy()),
    'oa_aggr_z': z(obs[:, 2], exp[:, 2], D['v_a'].to_numpy()),
    'oa_surp': D['nll'].to_numpy() / nn,
    'oa_surp_excess': (D['nll'].to_numpy() - D['ent'].to_numpy()) / nn,
    'oa_n_facing': D['n_fac'].to_numpy(),
    'oa_fold_z_fac': z(D['of_fac'].to_numpy(), D['ef_fac'].to_numpy(), D['vf_fac'].to_numpy()),
    'oa_aggr_z_fac': z(D['oa_fac'].to_numpy(), D['ea_fac'].to_numpy(), D['va_fac'].to_numpy()),
    'oa_aggr_dev_fac': (D['oa_fac'].to_numpy() - D['ea_fac'].to_numpy()) / nf,
}.items()])
FEATS = [c for c in F.columns if c.startswith('oa_')]
# fold the two directions onto the unordered pair, the key the pair model uses
F = F.with_columns(pl.min_horizontal('pidx', 'opp').alias('p'), pl.max_horizontal('pidx', 'opp').alias('q'))
P = F.group_by(['p', 'q', 'is_eval']).agg(
    [pl.col(c).max().alias(c + '_hi') for c in FEATS] + [pl.col(c).min().alias(c + '_lo') for c in FEATS] +
    [pl.col('oa_g_tail').sum().alias('oa_g_tail_sum'), pl.col('oa_g').sum().alias('oa_g_sum')])
P.write_parquet(OUT + 'oppaware_pair.parquet')
log('written pair', P.shape, len(P.columns), 'columns')

# ---- hand level: which hands carry the pair's characteristic deviation?
# log_t[g, a] = log( p_obs(a) / p_model(a) ) for directed group g. A hand scores high when the actions
# taken in it are exactly the ones this pair over-produces relative to the policy. This is the hand-level
# read of the same statistic, and it is what the evidence ranker needs: the pair column says "this pair is
# odd", this says "this hand is where the oddness happened".
if os.environ.get('OA_HAND', '1') == '1':
    log_t = np.log(p_obs / p_mod)                      # (n_directed, 3)
    key = D.select('pidx', 'opp', 'is_eval').with_columns(pl.Series('gidx', np.arange(len(D), dtype=np.int64)))
    TL = pl.DataFrame({'gidx': np.arange(len(D), dtype=np.int64),
                       'tilt_f': log_t[:, 0].astype(np.float32), 'tilt_c': log_t[:, 1].astype(np.float32),
                       'tilt_a': log_t[:, 2].astype(np.float32)})
    key = key.join(TL, on='gidx')
    outs = []
    for ch in range(NCH):
        a = A.filter(pl.col('tidx') % NCH == ch)
        s = S.filter(pl.col('tidx') % NCH == ch).select('hidx', pl.col('pidx').alias('opp'), 'fold_no')
        X = a.join(s, on='hidx').filter(
            (pl.col('opp') != pl.col('pidx')) &
            (pl.col('fold_no').is_null() | (pl.col('fold_no') > pl.col('action_no'))))
        X = X.join(key, on=['pidx', 'opp', 'is_eval'], how='inner')
        of = (pl.col('y') == 0); oc = pl.col('y').is_in([1, 2]); oa_ = pl.col('y').is_in([3, 4, 5])
        tilt = (pl.when(of).then(pl.col('tilt_f')).when(oc).then(pl.col('tilt_c')).otherwise(pl.col('tilt_a')))
        nll = -(pl.when(of).then(pl.col('p_fold')).when(oc).then(pl.col('p_pass')).otherwise(pl.col('p_agg'))).clip(1e-9).log()
        fac = (pl.col('last_agg_pidx') == pl.col('opp'))
        X = X.with_columns(tilt.alias('tilt'), nll.alias('nll3'), fac.alias('fac'))
        g2 = X.group_by(['hidx', 'pidx', 'opp']).agg(
            pl.col('tilt').sum().alias('t_sum'), pl.col('tilt').max().alias('t_max'),
            pl.col('nll3').sum().alias('n_sum'), pl.col('nll3').max().alias('n_max'),
            (pl.col('tilt') * pl.col('fac')).sum().alias('t_fac'), pl.len().alias('t_n'))
        outs.append(g2); log('hand chunk', ch, g2.height)
    H = pl.concat(outs)
    H = H.with_columns(pl.min_horizontal('pidx', 'opp').alias('p'), pl.max_horizontal('pidx', 'opp').alias('q'))
    HP = H.group_by(['hidx', 'p', 'q']).agg(
        pl.col('t_sum').sum().alias('oah_tilt'), pl.col('t_sum').max().alias('oah_tilt_hi'),
        pl.col('t_sum').min().alias('oah_tilt_lo'), pl.col('t_max').max().alias('oah_tilt_amax'),
        pl.col('n_sum').sum().alias('oah_nll'), pl.col('n_max').max().alias('oah_nll_max'),
        pl.col('t_fac').sum().alias('oah_tilt_fac'), pl.col('t_n').sum().alias('oah_n'))
    HP.write_parquet(OUT + 'oppaware_hand.parquet')
    log('written hand', HP.shape)
