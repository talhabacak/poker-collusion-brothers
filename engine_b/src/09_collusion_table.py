"""Step 9: collusion table (Mazrooei, Archibald & Bowling, AAAI 2013, 'Automating Collusion Detection in Sequential
Games'). For every action taken by player k, measure how much it changed player j's expected value:

    C(j, k) = sum over actions a by k along the hand of  [ V_j(h.a) - V_j(h) ]

V_j(h) is a learned estimate of what player j ends the hand with, given the public state and j's own cards. Colluders
show up as pairs where each one's actions systematically raise the other's expected value. This is different from our
surprisal (which measures how unlikely an action was) and from step 3d (which measured chips committed): it measures
who benefits from whom, without assuming any pattern of collusive behaviour."""
import polars as pl, numpy as np, lightgbm as lgb, time, os
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; CD = OUT + 'collusion/'; os.makedirs(CD, exist_ok=True)
NCH = 20; SEED = 42
seats = pl.read_parquet(OUT + 'seats_feat.parquet')
acts = pl.read_parquet(OUT + 'actions_scored.parquet')
hands = pl.read_parquet(OUT + 'hands.parquet').select('hidx', 'players_dealt')

VF = ['st', 'pot_bb', 'own_contrib_bb', 'own_stack_bb', 'own_str', 'relpos', 'players_active', 'is_live', 'to_act',
      'n_act_so_far', 'amt_to_call_bb', 'players_dealt']

def build_states(ch):
    """one row per (hand, action index t, seated player j): the state just before action t, from j's point of view"""
    S = seats.filter(pl.col('tidx') % NCH == ch)
    A = acts.filter(pl.col('tidx') % NCH == ch).sort(['hidx', 'action_no'])
    # state before each action: pot, who is live, how much each player has already put in
    A = A.with_columns((pl.col('amount').cum_sum().over(['hidx', 'pidx']) - pl.col('amount')).alias('contrib_before'))
    X = A.select('hidx', 'action_no', 'tidx', 'st', 'pot_before', 'players_active', 'to_call', 'big_blind',
                 pl.col('pidx').alias('actor')).join(
        S.select('hidx', 'pidx', 'fold_no', 'eq_pf', 'hs1', 'hs2', 'hs3', 'relpos', 'starting_stack', 'net_chips', 'big_blind').drop('big_blind'), on='hidx')
    X = X.join(A.select('hidx', 'action_no', pl.col('pidx').alias('pidx'), 'contrib_before'), on=['hidx', 'action_no', 'pidx'], how='left')
    # each player's own contribution before this action: last known value, forward filled
    X = X.sort(['hidx', 'pidx', 'action_no']).with_columns(pl.col('contrib_before').forward_fill().over(['hidx', 'pidx']).fill_null(0).alias('own_contrib'))
    X = X.with_columns(
        (pl.col('pot_before') / pl.col('big_blind')).alias('pot_bb'),
        (pl.col('own_contrib') / pl.col('big_blind')).alias('own_contrib_bb'),
        ((pl.col('starting_stack') - pl.col('own_contrib')) / pl.col('big_blind')).alias('own_stack_bb'),
        pl.when(pl.col('st') == 0).then(pl.col('eq_pf')).when(pl.col('st') == 1).then(pl.col('hs1'))
          .when(pl.col('st') == 2).then(pl.col('hs2')).otherwise(pl.col('hs3')).fill_null(0.5).alias('own_str'),
        (pl.col('fold_no').is_null() | (pl.col('fold_no') >= pl.col('action_no'))).cast(pl.Int8).alias('is_live'),
        (pl.col('actor') == pl.col('pidx')).cast(pl.Int8).alias('to_act'),
        (pl.col('to_call') / pl.col('big_blind')).alias('amt_to_call_bb'),
        pl.col('action_no').alias('n_act_so_far'),
        (pl.col('net_chips') / pl.col('big_blind')).alias('target'))
    X = X.join(hands, on='hidx')
    return X.select('hidx', 'action_no', 'pidx', 'actor', 'tidx', 'target', *VF)

# ---- train the value function on two table folds
log('building training states')
TR = pl.concat([build_states(ch) for ch in range(2)])
log('training rows', TR.shape)
Xtr = TR.select(VF).to_numpy().astype(np.float32); ytr = TR['target'].to_numpy()
rng = np.random.default_rng(SEED); idx = rng.permutation(len(ytr))[:6_000_000]
models = {}
for par in [0, 1]:
    sel = idx[(TR['tidx'].to_numpy()[idx] % 2) != par]
    m = lgb.train(dict(objective='l2', learning_rate=0.06, num_leaves=255, min_child_samples=200, feature_fraction=0.9,
                       bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=16, seed=SEED),
                  lgb.Dataset(Xtr[sel], ytr[sel]), 400)
    models[par] = m
    log(f'value function for parity {par}: trained on {len(sel)} rows, rmse={np.sqrt(np.mean((m.predict(Xtr[sel[:200000]])-ytr[sel[:200000]])**2)):.3f} bb')
del TR, Xtr, ytr

# ---- collusion values per (hand, j, k)
for ch in range(NCH):
    X = build_states(ch)
    par = ch % 2
    V = models[1 - par].predict(X.select(VF).to_numpy().astype(np.float32))
    X = X.with_columns(pl.Series('V', V.astype(np.float32))).sort(['hidx', 'pidx', 'action_no'])
    # delta for player j caused by the action at index t (whoever acted)
    X = X.with_columns((pl.col('V').shift(-1).over(['hidx', 'pidx']) - pl.col('V')).alias('dV'))
    X = X.drop_nulls('dV').filter(pl.col('actor') != pl.col('pidx'))     # impact of k's action on j (j != k)
    C = X.group_by(['hidx', pl.col('pidx').alias('j'), pl.col('actor').alias('k')]).agg(
        pl.col('dV').sum().alias('c_jk'), pl.col('dV').clip(0).sum().alias('c_jk_pos'), pl.col('dV').max().alias('c_jk_max'))
    # to unordered pair rows: how much each member's actions helped the other
    D = (C.rename({'j': 'p', 'k': 'q'}).rename({'c_jk': 'help_p_by_q', 'c_jk_pos': 'helppos_p_by_q', 'c_jk_max': 'helpmax_p_by_q'})
         .join(C.rename({'j': 'q', 'k': 'p'}).rename({'c_jk': 'help_q_by_p', 'c_jk_pos': 'helppos_q_by_p', 'c_jk_max': 'helpmax_q_by_p'}),
               on=['hidx', 'p', 'q'], how='full', coalesce=True))
    D = D.filter(pl.col('p') < pl.col('q')).with_columns([pl.col(c).fill_null(0).cast(pl.Float32) for c in
                 ['help_p_by_q', 'helppos_p_by_q', 'helpmax_p_by_q', 'help_q_by_p', 'helppos_q_by_p', 'helpmax_q_by_p']])
    D = D.with_columns((pl.col('help_p_by_q') + pl.col('help_q_by_p')).alias('ct_mutual'),
                       (pl.col('help_p_by_q') - pl.col('help_q_by_p')).abs().alias('ct_asym'),
                       pl.max_horizontal('help_p_by_q', 'help_q_by_p').alias('ct_max'),
                       (pl.col('helppos_p_by_q') + pl.col('helppos_q_by_p')).alias('ct_pos'))
    D.write_parquet(CD + f'chunk{ch:02d}.parquet')
    log('chunk', ch, D.shape)
log('done')
