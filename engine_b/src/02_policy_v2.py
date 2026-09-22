"""Step 2 v2: sharper population policy. Adds draw features (flush/straight draws from hole+board per street),
betting line (own aggression per street, opened preflop, last-street aggressor), and street-level opponent context
(callers so far, raisers so far). Same 2-fold-by-table OOF surprisal as step 2. Overwrites actions_scored.parquet
(re-run 02b afterwards; the run script deletes the 02b backup first)."""
import polars as pl, numpy as np, lightgbm as lgb, time, os
from sklearn.metrics import log_loss
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; RAW = 'data/raw/'; SEED = 42
RANKS = '23456789TJQKA'; SUITS = 'shdc'
A = pl.read_parquet(OUT + 'actions_feat.parquet')
# ---------- draw features per (hidx, pidx, street) ----------
seats = pl.read_parquet(RAW + 'seats.parquet', columns=['hand_id', 'player_id', 'hole_card_1', 'hole_card_2'])
hands = pl.read_parquet(RAW + 'hands.parquet', columns=['hand_id', 'board_cards'])
players = pl.read_parquet(OUT + 'players.parquet').select('player_id', 'pidx')
H = pl.read_parquet(OUT + 'hands.parquet').select('hidx', 'hand_id')
seats = seats.join(H, on='hand_id').join(players, on='player_id').join(hands, on='hand_id').drop('hand_id', 'player_id')
def card_arrays(df):
    b = df['board_cards'].fill_null('').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).to_list()
    h1 = df['hole_card_1'].to_list(); h2 = df['hole_card_2'].to_list()
    n = len(h1); rank = np.full((n, 7), -1, np.int8); suit = np.full((n, 7), -1, np.int8)
    for i in range(n):
        cs = [h1[i], h2[i]] + b[i][:5]
        for j, c in enumerate(cs):
            rank[i, j] = RANKS.index(c[0]); suit[i, j] = SUITS.index(c[1])
    return rank, suit
rank, suit = card_arrays(seats)
log('cards parsed', rank.shape)
def draws(ncard):
    """flush draw / straight draw flags using hole + first ncard board cards"""
    r = rank[:, :2 + ncard]; s = suit[:, :2 + ncard]
    valid = r >= 0
    # flush draw: exactly 4 of a suit including >=1 hole card; made flush: >=5
    fd = np.zeros(len(r), np.int8); made_f = np.zeros(len(r), np.int8)
    for su in range(4):
        cnt = ((s == su) & valid).sum(1); hole = ((s[:, :2] == su)).sum(1)
        fd |= ((cnt == 4) & (hole >= 1)).astype(np.int8); made_f |= (cnt >= 5).astype(np.int8)
    # straight draw: a 5-rank window containing 4 distinct ranks incl. a hole card (open-ended or gutshot); made straight: 5
    present = np.zeros((len(r), 14), bool)  # ace also low at index 0
    for j in range(r.shape[1]):
        m = valid[:, j]; present[np.arange(len(r))[m], r[m, j] + 1] = True
    present[:, 0] = present[:, 13]
    holep = np.zeros((len(r), 14), bool)
    for j in range(2): holep[np.arange(len(r)), r[:, j] + 1] = True
    holep[:, 0] = holep[:, 13]
    sd = np.zeros(len(r), np.int8); made_s = np.zeros(len(r), np.int8); oesd = np.zeros(len(r), np.int8)
    for lo in range(0, 10):
        w = present[:, lo:lo + 5]; c = w.sum(1); hw = holep[:, lo:lo + 5].any(1)
        made_s |= (c == 5).astype(np.int8)
        sd |= ((c == 4) & hw).astype(np.int8)
        # open-ended: 4 consecutive present with room on both sides
    for lo in range(1, 10):
        w4 = present[:, lo:lo + 4].all(1)
        oesd |= (w4 & holep[:, lo:lo + 4].any(1) & (lo > 0) & (lo + 4 < 14)).astype(np.int8)
    return fd, made_f, sd, oesd, made_s
frames = []
for st, nc in [(1, 3), (2, 4)]:   # river has no draws
    fd, mf, sd, oe, ms = draws(nc)
    frames.append(seats.select('hidx', 'pidx').with_columns(pl.lit(st, dtype=pl.Int8).alias('st'), pl.Series('flush_draw', fd), pl.Series('str_draw', sd), pl.Series('oesd', oe)))
DR = pl.concat(frames)
A = A.join(DR, on=['hidx', 'pidx', 'st'], how='left').with_columns([pl.col(c).fill_null(0) for c in ['flush_draw', 'str_draw', 'oesd']])
log('draw features joined')
# ---------- betting line ----------
A = A.sort(['hidx', 'action_no'])
A = A.with_columns(
    (pl.col('agg').cum_sum().over(['hidx', 'pidx', 'st']) - pl.col('agg')).alias('own_agg_st_before'),
    pl.col('y').is_in([2]).cast(pl.Int16).alias('is_call'))
A = A.with_columns((pl.col('is_call').cum_sum().over(['hidx', 'st']) - pl.col('is_call')).alias('n_call_st'),
                   (pl.col('is_call').cum_sum().over(['hidx']) - pl.col('is_call')).alias('n_call_hand'))
# did this player make the last aggressive action of the previous street?
last_agg_prev = (A.filter(pl.col('agg') == 1).group_by(['hidx', 'st']).agg(pl.col('pidx').last().alias('prev_agg_pidx'))
                 .with_columns((pl.col('st') + 1).alias('st')))
A = A.join(last_agg_prev, on=['hidx', 'st'], how='left').with_columns((pl.col('prev_agg_pidx') == pl.col('pidx')).fill_null(False).cast(pl.Int8).alias('was_prev_aggressor')).drop('prev_agg_pidx')
opened = A.filter((pl.col('st') == 0) & (pl.col('agg') == 1)).group_by('hidx').agg(pl.col('pidx').first().alias('pf_opener'))
A = A.join(opened, on='hidx', how='left').with_columns((pl.col('pf_opener') == pl.col('pidx')).fill_null(False).cast(pl.Int8).alias('is_pf_opener')).drop('pf_opener')
A = A.with_columns((pl.col('pot_before') / (pl.col('players_dealt') * pl.col('big_blind'))).alias('pot_per_player_bb'))
FEATS = ['st', 'relpos', 'hi', 'lo', 'suited', 'pocket', 'gap', 'eq_pf', 'hs', 'hc', 'potodds', 'tocall_bb', 'pot_bb', 'stack_bb', 'tocall_stack', 'spr',
         'players_active', 'players_dealt', 'n_agg_hand', 'n_agg_st', 'own_agg_before', 'n_act_st', 'self_last_agg', 'big_blind', 'action_no',
         'ps_n', 'ps_open_raise', 'ps_open_fold', 'ps_vs_raise_agg', 'ps_vs_raise_fold', 'ps_post_agg', 'ps_post_fold_vs_bet',
         'flush_draw', 'str_draw', 'oesd', 'own_agg_st_before', 'n_call_st', 'n_call_hand', 'was_prev_aggressor', 'is_pf_opener', 'pot_per_player_bb']
A = A.with_columns((pl.col('tidx') % 2).alias('fold'),
                   pl.when(pl.col('agg') == 1).then(((pl.col('amount_to').cast(pl.Float32)) / pl.col('pot_before').clip(1)).log1p()).otherwise(None).alias('size_y'))
X = A.select(FEATS).to_numpy().astype(np.float32); y = A['y'].to_numpy(); fold = A['fold'].to_numpy(); sy = A['size_y'].to_numpy()
log('matrix', X.shape)
rng = np.random.default_rng(SEED)
P = np.zeros((len(y), 6), dtype=np.float32); size_res = np.full(len(y), np.nan, dtype=np.float32)
for f in [0, 1]:
    tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
    tr = rng.permutation(tr); va = tr[:300_000]; trn = tr[300_000:5_300_000]
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.06, num_leaves=255, min_child_samples=200, subsample=0.7, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=16, random_state=SEED)
    m.fit(X[trn], y[trn], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    P[te] = m.predict_proba(X[te])
    log(f'fold {f} policy v2 best_iter={m.best_iteration_} oof logloss={log_loss(y[te], P[te], labels=range(6)):.4f}')
    trs = tr[~np.isnan(sy[tr])][:3_000_000]; tes = te[~np.isnan(sy[te])]
    ms = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.08, num_leaves=127, min_child_samples=200, subsample=0.7, subsample_freq=1, verbose=-1, n_jobs=16, random_state=SEED)
    Xs = np.column_stack([X, y[:, None].astype(np.float32)])
    ms.fit(Xs[trs], sy[trs]); res = sy[tes] - ms.predict(Xs[tes]); size_res[tes] = res / (np.std(res) + 1e-6)
    del m, ms
surp = -np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1))
out = A.select('hidx', 'tidx', 'is_eval', 'action_no', 'pidx', 'st', 'y', 'amount', 'amount_to', 'pot_before', 'to_call', 'players_active', 'big_blind',
               'last_agg_pidx', 'eq_pf', 'hs', 'stack_before').with_columns(
    pl.Series('surp', surp.astype(np.float32)), pl.Series('p_fold', P[:, 0]), pl.Series('p_pass', P[:, 1] + P[:, 2]), pl.Series('p_agg', P[:, 3] + P[:, 4] + P[:, 5]),
    pl.Series('size_z', size_res).fill_nan(None))
out.write_parquet(OUT + 'actions_scored.parquet')
log('saved; mean surprisal', float(surp.mean()))
