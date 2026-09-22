"""Step 2: population action policy p(action | state) + bet-size model, 2-fold by table -> out-of-fold surprisal for every action."""
import polars as pl, numpy as np, lightgbm as lgb, time
from sklearn.metrics import log_loss
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; SEED = 42

FEATS = ['st', 'relpos', 'hi', 'lo', 'suited', 'pocket', 'gap', 'eq_pf', 'hs', 'hc', 'potodds', 'tocall_bb', 'pot_bb', 'stack_bb', 'tocall_stack', 'spr',
         'players_active', 'players_dealt', 'n_agg_hand', 'n_agg_st', 'own_agg_before', 'n_act_st', 'self_last_agg', 'big_blind', 'action_no',
         'ps_n', 'ps_open_raise', 'ps_open_fold', 'ps_vs_raise_agg', 'ps_vs_raise_fold', 'ps_post_agg', 'ps_post_fold_vs_bet']
A = pl.read_parquet(OUT + 'actions_feat.parquet')
A = A.with_columns((pl.col('tidx') % 2).alias('fold'),
                   pl.when(pl.col('agg') == 1).then(((pl.col('amount_to').cast(pl.Float32)) / pl.col('pot_before').clip(1)).log1p()).otherwise(None).alias('size_y'))
X = A.select(FEATS).to_numpy().astype(np.float32); y = A['y'].to_numpy(); fold = A['fold'].to_numpy(); sy = A['size_y'].to_numpy()
log('matrix', X.shape)
rng = np.random.default_rng(SEED)
P = np.zeros((len(y), 6), dtype=np.float32); size_res = np.full(len(y), np.nan, dtype=np.float32)
for f in [0, 1]:
    tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
    tr = rng.permutation(tr); va = tr[:300_000]; trn = tr[300_000:4_300_000]
    m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.08, num_leaves=255, min_child_samples=200, subsample=0.7, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=16, random_state=SEED)
    m.fit(X[trn], y[trn], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    P[te] = m.predict_proba(X[te])
    log(f'fold {f} policy best_iter={m.best_iteration_} oof logloss={log_loss(y[te], P[te], labels=range(6)):.4f}')
    # bet size model
    trs = tr[~np.isnan(sy[tr])][:3_000_000]; tes = te[~np.isnan(sy[te])]
    ms = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.08, num_leaves=127, min_child_samples=200, subsample=0.7, subsample_freq=1, verbose=-1, n_jobs=16, random_state=SEED)
    Xs = np.column_stack([X, y[:, None].astype(np.float32)])
    ms.fit(Xs[trs], sy[trs])
    pred = ms.predict(Xs[tes]); res = sy[tes] - pred
    size_res[tes] = res / (np.std(res) + 1e-6)
    log(f'fold {f} size model resid std={np.std(res):.4f} vs raw std={np.std(sy[tes]):.4f}')
    del m, ms
prior = np.bincount(y, minlength=6) / len(y)
log('prior-only logloss', log_loss(y, np.tile(prior, (len(y), 1))[:2_000_000] if False else np.tile(prior, (200000, 1)), labels=range(6)) if False else '')
surp = -np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1))
out = A.select('hidx', 'tidx', 'is_eval', 'action_no', 'pidx', 'st', 'y', 'amount', 'amount_to', 'pot_before', 'to_call', 'players_active', 'big_blind',
               'last_agg_pidx', 'eq_pf', 'hs', 'stack_before').with_columns(
    pl.Series('surp', surp.astype(np.float32)), pl.Series('p_fold', P[:, 0]), pl.Series('p_pass', P[:, 1] + P[:, 2]), pl.Series('p_agg', P[:, 3] + P[:, 4] + P[:, 5]),
    pl.Series('size_z', size_res).fill_nan(None))
out.write_parquet(OUT + 'actions_scored.parquet')
log('saved; mean surprisal', surp.mean())
