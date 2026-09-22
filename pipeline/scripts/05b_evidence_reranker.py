"""Step 5b: stage-2 evidence reranker (lambdarank, within-pair chronology + stage-1 scores).
Reads submission.csv (risk + behavior from step 5), replaces evidence columns, writes submission_v2.csv."""
import os
import polars as pl, numpy as np, glob, lightgbm as lgb, time, json

def _assert_unique_keys(df, name):   # a stale extra file in a globbed artefact dir duplicates rows and inflates MAP@5
    if df is not None and df.select('hidx', 'p', 'q').n_unique() != df.height:
        raise SystemExit(f'{name}: {df.height - df.select("hidx", "p", "q").n_unique()} duplicate (hidx,p,q) rows - stale file in the artefact directory?')

t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; RAW = 'data/raw/'; NF = 5; SEED = 42; NCAND = int(__import__('os').environ.get('NCAND', 50))
players = pl.read_parquet(OUT + 'players.parquet').select('player_id', 'pidx')
def to_pq(df):
    df = df.join(players.rename({'player_id': 'player_1', 'pidx': 'i1'}), on='player_1').join(players.rename({'player_id': 'player_2', 'pidx': 'i2'}), on='player_2')
    return df.with_columns(pl.min_horizontal('i1', 'i2').alias('p'), pl.max_horizontal('i1', 'i2').alias('q')).drop('i1', 'i2')
lab = to_pq(pl.read_csv(RAW + 'development_labels.csv').filter(pl.col('label') == 1))
evp = to_pq(pl.read_csv(RAW + 'evaluation_pairs.csv'))
hands = pl.read_parquet(OUT + 'hands.parquet').select('hidx', 'hand_id')
ev = pl.read_csv(RAW + 'development_evidence.csv').join(hands, on='hand_id').select('pair_id', 'hidx', 'evidence_rank')

S1B = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'stage1b/*.parquet'))])
ACTJ = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'actmodel/*.parquet'))]) if os.environ.get('USE_ACT', '1') == '1' else None
_assert_unique_keys(ACTJ, 'ACTJ')
ACT_COLS = [c for c in (ACTJ.columns if ACTJ is not None else []) if c not in ('hidx', 'p', 'q')]
VALJ = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'value/chunk*.parquet'))]) if os.environ.get('USE_VAL', '1') == '1' else None
VAL_COLS = [c for c in (VALJ.columns if VALJ is not None else []) if c not in ('hidx', 'p', 'q')]
TOKJ = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'tokmodel/*.parquet'))]) if os.environ.get('USE_TOK', '0') == '1' else None
# step 13 hand level: how much does THIS hand carry the pair's characteristic policy deviation
OAHJ = pl.read_parquet(OUT + 'oppaware_hand.parquet') if os.environ.get('USE_OAH', '0') == '1' else None
OAH_COLS = [c for c in (OAHJ.columns if OAHJ is not None else []) if c not in ('hidx', 'p', 'q')]
# step 14: bet-sizing signature of value transfer, per hand and pair
SZJ = pl.read_parquet(OUT + 'sizing_hand.parquet') if os.environ.get('USE_SZ', '0') == '1' else None
SZ_COLS = [c for c in (SZJ.columns if SZJ is not None else []) if c not in ('hidx', 'p', 'q')]
# step 15 hand level: same reading, but the expectation knows which opponent is sitting there
OCHJ = pl.read_parquet(OUT + 'oppcond_hand.parquet') if os.environ.get('USE_OCH', '0') == '1' else None
OCH_COLS = [c for c in (OCHJ.columns if OCHJ is not None else []) if c not in ('hidx', 'p', 'q')]
TOK_COLS = [c for c in (TOKJ.columns if TOKJ is not None else []) if c not in ('hidx', 'p', 'q')]
_FAMS3 = ['directed_transfer', 'soft_play', 'coordinated_isolation']
_t5 = lambda c: pl.col(c).sort(descending=True).head(5).mean()
FAMRULE = (pl.concat([pl.scan_parquet(f).select('p', 'q', 'is_eval', 's_dt', 's_sp', 's_ci').collect() for f in sorted(glob.glob(OUT + 'handscores/chunk*.parquet'))])
           .group_by(['p', 'q', 'is_eval']).agg(_t5('s_dt').alias('a_dt'), _t5('s_sp').alias('a_sp'), _t5('s_ci').alias('a_ci')))
FAMRULE = FAMRULE.with_columns(pl.when((pl.col('a_dt') >= pl.col('a_sp')) & (pl.col('a_dt') >= pl.col('a_ci'))).then(pl.lit(_FAMS3[0]))
                               .when(pl.col('a_sp') >= pl.col('a_ci')).then(pl.lit(_FAMS3[1])).otherwise(pl.lit(_FAMS3[2])).alias('rule_fam')).select('p', 'q', 'is_eval', 'rule_fam')
def stage2_features(H):
    H = H.join(S1B, on=['hidx', 'p', 'q'], how='left').with_columns(pl.col('s1b').fill_null(-99))
    if ACTJ is not None: H = H.join(ACTJ, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0) for c in ACT_COLS])
    if VALJ is not None: H = H.join(VALJ, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0) for c in VAL_COLS])
    if TOKJ is not None: H = H.join(TOKJ, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0) for c in TOK_COLS])
    if OAHJ is not None:
        H = H.join(OAHJ, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0) for c in OAH_COLS])
    if SZJ is not None:
        H = H.join(SZJ, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0) for c in SZ_COLS])
    if OCHJ is not None:
        H = H.join(OCHJ, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0) for c in OCH_COLS])
    g = ['pair_id']
    H = H.sort(g + ['t_order'])
    H = H.with_columns(pl.col('s1b').rank('ordinal', descending=True).over(g).alias('r1b'),
                       (pl.col('s1b') - pl.col('s1b').max().over(g)).alias('s1b_rel_max'),
                       (pl.col('s1b') - pl.col('s1b').sort(descending=True).slice(4, 1).first().over(g)).alias('s1b_minus_5th'))
    H = H.with_columns(pl.min_horizontal(pl.col('s_ev').rank('ordinal', descending=True).over(g), pl.col('r1b')).rank('ordinal').over(g).alias('r'),
                       pl.int_range(pl.len()).over(g).alias('ti'), pl.len().over(g).alias('n'),
                       pl.max_horizontal('s_dt', 's_sp', 's_ci').alias('s_fam_max'))
    H = H.with_columns(((pl.col('tot_surp') - pl.col('tot_surp').mean().over(g)) / (pl.col('tot_surp').std().over(g) + 1e-3)).alias('surp_z_pair'),
                       ((pl.col('rel_llr_pos') - pl.col('rel_llr_pos').mean().over(g)) / (pl.col('rel_llr_pos').std().over(g) + 1e-3)).alias('rel_pos_z_pair'),
                       pl.col('tot_surp').rank('ordinal', descending=True).over(g).alias('surp_rank_pair'))
    H = H.with_columns((pl.col('ti') / pl.col('n')).alias('tpos'), (pl.col('r') <= 15).alias('cand'),
                       (pl.col('s_ev') / pl.col('s_ev').max().over(g)).alias('s_rel_max'),
                       (pl.col('s_ev') - pl.col('s_ev').sort(descending=True).slice(4, 1).first().over(g)).alias('s_minus_5th'))
    exprs = []
    for t in [0.1, 0.2, 0.3]:
        f = (pl.col('s_ev') > t).cast(pl.Int32)
        exprs += [(f.cum_sum().over(g) - f).alias(f'prior_gt{t}'), (f.sum().over(g) - f.cum_sum().over(g)).alias(f'after_gt{t}')]
    H = H.with_columns(exprs + [(pl.col('s_ev').cum_sum().over(g) - pl.col('s_ev')).alias('prior_s_sum'),
                                (pl.col('s_ev').sum().over(g) - pl.col('s_ev').cum_sum().over(g)).alias('after_s_sum')])
    c = H.filter('cand').with_columns(pl.col('ti').rank('ordinal').over(g).alias('cand_chrono'))
    H = H.join(c.select(*g, 'hidx', 'cand_chrono'), on=g + ['hidx'], how='left').with_columns(pl.col('cand_chrono').fill_null(99))
    H = H.with_columns((pl.col('prior_s_sum') / (pl.col('prior_s_sum') + pl.col('after_s_sum') + 1e-6)).alias('mass_before'))
    return H.filter(pl.col('r') <= NCAND).sort(['pair_id', 'r'])

FEATS = ['s_ev', 's_dt', 's_sp', 's_ci', 's_fam_max', 'r', 's_rel_max', 's_minus_5th', 'tot_surp', 'tot_surp_max', 'pot_bb', 'flow_bb', 'both_sd', 'one_fold', 'neither_fold',
         'board_len', 'n_third_fold_vs_pair', 'both_vpip', 'opp_sign', 'W_n_agg_multi', 'L_n_agg_multi', 'W_surp_pass_ahead', 'L_surp_pass_ahead', 'L_dump_value', 'W_dump_value',
         'L_fold_value', 'W_fold_value', 'tpos', 'cand_chrono', 'prior_gt0.1', 'prior_gt0.2', 'prior_gt0.3', 'after_gt0.1', 'after_gt0.2', 'after_gt0.3', 'prior_s_sum',
         'after_s_sum', 'mass_before', 'rel_llr_tot', 'rel_llr_pos', 'rel_llr_max', 'L_llr_face_b', 'W_llr_face_x', 'L_llr_pass', 'W_llr_pass',
         'surp_z_pair', 'rel_pos_z_pair', 'surp_rank_pair', 's1b', 'r1b', 's1b_rel_max', 's1b_minus_5th',
         'n3_fold_to_pair', 'third_dead_bb', 'n_pair_agg_multiway', 'n_pair_reraise_over_third'] + ACT_COLS + TOK_COLS + VAL_COLS + OAH_COLS + SZ_COLS + OCH_COLS
import os
if os.environ.get('RERANK_NO_REL') == '1':
    FEATS = [f for f in FEATS if not (f.startswith('rel_') or 'llr' in f or f in ('surp_z_pair', 'rel_pos_z_pair', 'surp_rank_pair'))]
if os.environ.get('RERANK_NO_S1B') == '1':
    FEATS = [f for f in FEATS if not f.startswith('s1b') and f != 'r1b']
OUTFILE = os.environ.get('RERANK_OUT', 'submission_v3.csv')
PARAMS = dict(objective='lambdarank', eval_at=[5], lambdarank_truncation_level=int(os.environ.get('LR_TRUNC','10')), learning_rate=0.03, num_leaves=15, min_child_samples=30, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=16, seed=SEED)

files = sorted(glob.glob(OUT + 'handscores/chunk*.parquet'))
D = pl.concat([pl.scan_parquet(f).filter(pl.col('is_eval') == 0).join(lab.lazy().select('p', 'q', 'pair_id'), on=['p', 'q']).collect() for f in files])
D = stage2_features(D).join(ev, on=['pair_id', 'hidx'], how='left').with_columns(pl.col('evidence_rank').is_not_null().cast(pl.Int32).alias('y'), (pl.col('tidx') % NF).alias('fold'))
# GRADE: the organiser ranks each pair's five evidence hands by strength, and our recall follows that
# ranking closely (rank 1: 0.85 ... rank 5: 0.46). All five score the same under MAP@5, so capacity spent
# on the easy ones is wasted. 'hard' relabels weak evidence as the highest lambdarank grade; 'easy' is the
# opposite direction, as a control.
GRADE = os.environ.get('GRADE', 'off')
if GRADE != 'off':
    r = pl.col('evidence_rank')
    lv = (r if GRADE == 'hard' else (6 - r))          # hard: rank5 -> 5, rank1 -> 1
    D = D.with_columns(pl.when(r.is_null()).then(0).otherwise(lv).cast(pl.Int32).alias('y'))
    PARAMS_EXTRA = dict(label_gain=[0, 1, 2, 3, 4, 5])
else:
    PARAMS_EXTRA = {}
truth = ev.group_by('pair_id').agg(pl.col('hidx').alias('truth'))
def map5(df, col):
    r = df.sort([col, 'hidx'], descending=[True, False]).group_by('pair_id', maintain_order=True).agg(pl.col('hidx').head(5).alias('pred')).join(truth, on='pair_id')
    out = []
    for pred, tru in zip(r['pred'].to_list(), r['truth'].to_list()):
        t = set(tru); h = 0; s = 0.0
        for i, x in enumerate(pred):
            if x in t: h += 1; s += h / (i + 1)
        out.append(s / min(len(t), 5))
    return float(np.mean(out))
PER_FAM = os.environ.get('RERANK_PER_FAMILY') == '1'
FAMS3 = ['directed_transfer', 'soft_play', 'coordinated_isolation']
if PER_FAM:
    D = D.join(lab.select('p', 'q', 'behavior_family'), on=['p', 'q'], how='left') \
         .join(FAMRULE.filter(pl.col('is_eval') == 0).drop('is_eval'), on=['p', 'q'], how='left').with_columns(pl.col('rule_fam').fill_null(_FAMS3[0]))
models = []; oof = np.zeros(len(D)); fo = D['fold'].to_numpy()
for f in range(NF):
    tr = D.filter(pl.col('fold') != f)
    if PER_FAM:
        fam_models = {}
        for fam in FAMS3:
            trf = tr.filter(pl.col('behavior_family') == fam)
            grpf = trf.group_by('pair_id', maintain_order=True).len()['len'].to_numpy()
            fam_models[fam] = lgb.train(PARAMS, lgb.Dataset(trf.select(FEATS).to_numpy().astype(np.float32), trf['y'].to_numpy(), group=grpf), 400)
        te = D.filter(pl.col('fold') == f)
        famv = te['rule_fam'].to_numpy(); Xte = te.select(FEATS).to_numpy().astype(np.float32); pr = np.zeros(len(te))
        for fam in FAMS3:
            sel = famv == fam
            if sel.sum(): pr[sel] = fam_models[fam].predict(Xte[sel])
        oof[fo == f] = pr; models.append(fam_models)
        continue
    grp = tr.group_by('pair_id', maintain_order=True).len()['len'].to_numpy()
    seeds = [int(x) for x in os.environ.get('RERANK_SEEDS', str(SEED)).split(',')]
    ms = [lgb.train({**PARAMS, **PARAMS_EXTRA, 'seed': sd, 'bagging_seed': sd + 1, 'feature_fraction_seed': sd + 2},
                    lgb.Dataset(tr.select(FEATS).to_numpy().astype(np.float32), tr['y'].to_numpy(), group=grp), 400) for sd in seeds]
    Xte = D.filter(pl.col('fold') == f).select(FEATS).to_numpy().astype(np.float32)
    oof[fo == f] = np.mean([pl.Series(m.predict(Xte)).rank().to_numpy() for m in ms], axis=0); models += ms
cv = {'per_family': PER_FAM, 'stage1_map5': map5(D, 's_ev'), 'stage1b_map5': map5(D, 's1b'), 'reranker_map5_oof': map5(D.with_columns(pl.Series('s2', oof)), 's2')}
# per-rank recall: are the organiser's weaker evidence hands (rank 4-5) the ones we miss?
_D = D.with_columns(pl.Series('s2', oof))
_top5 = _D.sort(['pair_id', 's2'], descending=[False, True]).group_by('pair_id', maintain_order=True).head(5).select('pair_id', 'hidx').with_columns(pl.lit(1).alias('hit'))
_r = ev.join(_top5, on=['pair_id', 'hidx'], how='left').with_columns(pl.col('hit').fill_null(0))
cv['recall_by_rank'] = {int(k): round(v, 4) for k, v in
                        zip(*_r.group_by('evidence_rank').agg(pl.col('hit').mean()).sort('evidence_rank').to_dict(as_series=False).values())}
if os.environ.get('RERANK_DUMP'):   # save the out-of-fold ranking so several configurations can be blended
    _D.select('pair_id', 'hidx', 's2', 'y').write_parquet(OUT + 'rerank_oof_' + os.environ['RERANK_DUMP'] + '.parquet')
log('CV', cv); json.dump(cv, open(OUT + 'reranker_cv.json', 'w'), indent=1)

E = pl.concat([pl.scan_parquet(f).filter(pl.col('is_eval') == 1).join(evp.lazy().select('p', 'q', 'pair_id'), on=['p', 'q']).collect() for f in files])
E = stage2_features(E).join(FAMRULE.filter(pl.col('is_eval') == 1).drop('is_eval'), on=['p', 'q'], how='left').with_columns(pl.col('rule_fam').fill_null(_FAMS3[0]))
XE = E.select(FEATS).to_numpy().astype(np.float32)
if PER_FAM:
    famv = E['rule_fam'].to_numpy(); sc2 = np.zeros(len(E))
    for fam in _FAMS3:
        sel = famv == fam
        if sel.sum(): sc2[sel] = np.mean([fm[fam].predict(XE[sel]) for fm in models], axis=0)
    E = E.with_columns(pl.Series('s2', sc2))
else:
    E = E.with_columns(pl.Series('s2', np.mean([pl.Series(m.predict(XE)).rank().to_numpy() for m in models], axis=0)))
log('eval candidates', E.shape, 'pairs', E['pair_id'].n_unique())
top5 = (E.sort(['pair_id', 's2', 'hidx'], descending=[False, True, False]).group_by('pair_id', maintain_order=True).agg(pl.col('hidx').head(5))
        .explode('hidx').join(hands, on='hidx', how='left').group_by('pair_id', maintain_order=True).agg(pl.col('hand_id')))
sub = pl.read_csv(os.environ.get('RERANK_IN', 'submission.csv')).select('pair_id', 'risk_score', 'predicted_behavior').join(top5, on='pair_id', how='left')
sub = sub.with_columns([pl.col('hand_id').list.get(i, null_on_oob=True).fill_null('NO_EVIDENCE').alias(f'evidence_hand_{i+1}') for i in range(5)]).drop('hand_id')
sub.write_csv(OUTFILE)
old = pl.read_csv(os.environ.get('RERANK_IN', 'submission.csv'))
same1 = (old['evidence_hand_1'] == sub['evidence_hand_1']).mean()
log('written', OUTFILE, sub.shape, '| evidence_hand_1 unchanged share:', round(same1, 3))
