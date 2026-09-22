"""Step 4: supervised evidence-hand scorer (multiclass: non-evidence / DT / SP / CI), 5-fold by table.
Dev rows get out-of-fold predictions from the fold model that never saw that table; eval rows get the 5-model average."""
import os
import polars as pl, numpy as np, lightgbm as lgb, time, glob, os, json
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; RAW = 'data/raw/'; SEED = 42; NF = 5
FAM = {'directed_transfer': 1, 'soft_play': 2, 'coordinated_isolation': 3}
players = pl.read_parquet(OUT + 'players.parquet').select('player_id', 'pidx')
def to_pq(df):
    df = df.join(players.rename({'player_id': 'player_1', 'pidx': 'i1'}), on='player_1').join(players.rename({'player_id': 'player_2', 'pidx': 'i2'}), on='player_2')
    return df.with_columns(pl.min_horizontal('i1', 'i2').alias('p'), pl.max_horizontal('i1', 'i2').alias('q')).drop('i1', 'i2')
lab = to_pq(pl.read_csv(RAW + 'development_labels.csv'))
hands = pl.read_parquet(OUT + 'hands.parquet').select('hidx', 'hand_id')
ev = pl.read_csv(RAW + 'development_evidence.csv').join(hands, on='hand_id').select('pair_id', 'hidx', 'evidence_rank')

files = sorted(glob.glob(OUT + 'pairhand/chunk*.parquet'))
REL = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'relational/chunk*.parquet'))])
XTR = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'extra/chunk*.parquet'))])       # step 3c
if os.environ.get('USE_CT', '1') == '1':   # step 9 collusion table (AAAI 2013)
    CTB = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'collusion/chunk*.parquet'))]).select(
        'hidx', 'p', 'q', 'ct_mutual', 'ct_asym', 'ct_max', 'ct_pos', 'help_p_by_q', 'help_q_by_p')
    XTR = XTR.join(CTB, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0.0) for c in CTB.columns if c not in ('hidx', 'p', 'q')])
if os.environ.get('USE_VAL', '1') == '1':   # step 3d counterfactual value features
    VAL = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'value/chunk*.parquet'))])
    XTR = XTR.join(VAL, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0) for c in VAL.columns if c not in ('hidx', 'p', 'q')])
REL = REL.join(XTR, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0) for c in XTR.columns if c not in ('hidx', 'p', 'q')])  # step 3b
sample = pl.read_parquet(files[0], n_rows=5)
EXCL = {'hidx', 'p', 'q', 'tidx', 'is_eval', 't_order', 'pair_tpos', 'big_blind'}
FEATS = [c for c in sample.columns if c not in EXCL] + [c for c in REL.columns if c not in EXCL]
log('n feats', len(FEATS))

L = pl.concat([pl.scan_parquet(f).filter(pl.col('is_eval') == 0).join(lab.lazy().select('p', 'q', 'pair_id', 'label', 'behavior_family'), on=['p', 'q']).collect() for f in files]).join(REL, on=['hidx', 'p', 'q'], how='left')
HOLDOUT_FAM = os.environ.get('HOLDOUT_FAM')   # leave-one-family-out: the model gets no head for this family
if HOLDOUT_FAM:
    ev_h = ev.join(lab.select('pair_id', 'behavior_family'), on='pair_id').filter(pl.col('behavior_family') != HOLDOUT_FAM).select('pair_id', 'hidx', 'evidence_rank')
    log('LOFO: hiding', HOLDOUT_FAM, '->', ev.height - ev_h.height, 'evidence hands relabelled as non-evidence')
    ev = ev_h
L = L.join(ev, on=['pair_id', 'hidx'], how='left').with_columns(
    pl.when(pl.col('evidence_rank').is_not_null()).then(pl.col('behavior_family').replace_strict(FAM, default=0)).otherwise(0).alias('target'),
    (pl.col('tidx') % NF).alias('fold'))
log('labeled rows', L.shape, L['target'].value_counts().sort('target').to_dicts())
X = L.select(FEATS).to_numpy().astype(np.float32); yv = L['target'].to_numpy(); fold = L['fold'].to_numpy()
params = dict(objective='multiclass', num_class=4, learning_rate=0.05, num_leaves=31, min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=16, seed=SEED)
NROUND = 400
oof = np.zeros((len(yv), 4), dtype=np.float32); models = []
for f in range(NF):
    tr = fold != f
    m = lgb.train(params, lgb.Dataset(X[tr], yv[tr]), NROUND)
    oof[fold == f] = m.predict(X[fold == f]); models.append(m)
    m.save_model(OUT + f'hand_scorer_f{f}.txt')
log('trained 5 fold models')
imp = sorted(zip(FEATS, models[0].feature_importance('gain')), key=lambda t: -t[1])[:25]
print('top features:', [(n, round(v)) for n, v in imp])

def map5(df, score_col):
    truth = ev.group_by('pair_id').agg(pl.col('hidx').alias('truth'))
    r = (df.filter(pl.col('label') == 1).sort(score_col, descending=True).group_by('pair_id', maintain_order=True).agg(pl.col('hidx').head(5).alias('pred'))
         .join(truth, on='pair_id'))
    aps = []
    for pred, tru in zip(r['pred'].to_list(), r['truth'].to_list()):
        t = set(tru); hits = 0; s = 0.0
        for i, h in enumerate(pred):
            if h in t: hits += 1; s += hits / (i + 1)
        aps.append(s / min(len(t), 5))
    return float(np.mean(aps))
L = L.with_columns(pl.Series('s_ev', 1 - oof[:, 0]), pl.Series('s_dt', oof[:, 1]), pl.Series('s_sp', oof[:, 2]), pl.Series('s_ci', oof[:, 3]))
L = L.with_columns(pl.col('behavior_family').replace_strict({'directed_transfer': 's_dt', 'soft_play': 's_sp', 'coordinated_isolation': 's_ci'}, default='s_ev').alias('_fc'))
L = L.with_columns(pl.when(pl.col('_fc') == 's_dt').then(pl.col('s_dt')).when(pl.col('_fc') == 's_sp').then(pl.col('s_sp')).when(pl.col('_fc') == 's_ci').then(pl.col('s_ci')).otherwise(pl.col('s_ev')).alias('s_truefam'))
res = {'map5_s_ev': map5(L, 's_ev'), 'map5_truefam_oracle': map5(L, 's_truefam'), 'map5_raw_surprisal': map5(L, 'tot_surp')}
log('Evidence MAP@5 (OOF, public positives):', res)
json.dump(res, open(OUT + 'hand_scorer_cv.json', 'w'), indent=1)

# score every pair-hand row
os.makedirs(OUT + 'handscores', exist_ok=True)
KEEP = ['hidx', 'p', 'q', 'tidx', 'is_eval', 't_order', 'tot_surp', 'tot_surp_max', 'pot_bb', 'flow_bb', 'both_sd', 'one_fold', 'neither_fold', 'board_len',
        'n_third_fold_vs_pair', 'both_vpip', 'opp_sign', 'W_n_agg_multi', 'L_n_agg_multi', 'W_surp_pass_ahead', 'L_surp_pass_ahead', 'L_dump_value', 'W_dump_value',
        'L_fold_value', 'W_fold_value', 'pair_n', 'n3_fold_to_pair', 'third_dead_bb', 'n_pair_agg_multiway', 'n_pair_reraise_over_third', 'rel_llr_tot', 'rel_llr_pos', 'rel_llr_max', 'L_llr_face_b', 'W_llr_face_x', 'L_llr_pass', 'W_llr_pass']
for i, fpath in enumerate(files):
    D = pl.read_parquet(fpath).join(REL, on=['hidx', 'p', 'q'], how='left')
    Xd = D.select(FEATS).to_numpy().astype(np.float32); isev = D['is_eval'].to_numpy(); fo = (D['tidx'].to_numpy() % NF)
    P = np.zeros((len(D), 4), dtype=np.float32)
    dev = np.where(isev == 0)[0]
    for f in range(NF):
        idx = dev[fo[dev] == f]
        if len(idx): P[idx] = models[f].predict(Xd[idx])
    evi = np.where(isev == 1)[0]
    if len(evi): P[evi] = np.mean([m.predict(Xd[evi]) for m in models], axis=0)
    D.select(KEEP).with_columns(pl.Series('s_ev', 1 - P[:, 0]), pl.Series('s_dt', P[:, 1]), pl.Series('s_sp', P[:, 2]), pl.Series('s_ci', P[:, 3])).write_parquet(OUT + f'handscores/chunk{i:02d}.parquet')
    log('scored', fpath, len(D))
log('done')
