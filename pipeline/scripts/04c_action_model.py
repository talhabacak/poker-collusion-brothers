"""Step 4c: action-level collusion model. Learns 'the behavior-specific visible action' directly.
Rows: every action of a pair member in a pair-hand, with partner context. Label: hand is a planted evidence hand
for that pair. One model per family (true family for training, rule family for scoring), 5-fold by table.
Outputs per (hidx,p,q): act_max, act_sum, act_max_W/L-free (per member), n_act_gt05, street of the max.
Dev rows are scored out-of-fold. Eval rows are restricted to each pair's top-NCAND hands by stage-1 s_ev."""
import polars as pl, numpy as np, lightgbm as lgb, glob, os, time, json
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; RAW = 'data/raw/'; NF = 5; SEED = 42; NCAND = int(os.environ.get('ACT_NCAND', 60)); NCH = 10
AD = OUT + 'actmodel/'; os.makedirs(AD, exist_ok=True)
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
players = pl.read_parquet(OUT + 'players.parquet').select('player_id', 'pidx')
def to_pq(df):
    df = df.join(players.rename({'player_id': 'player_1', 'pidx': 'i1'}), on='player_1').join(players.rename({'player_id': 'player_2', 'pidx': 'i2'}), on='player_2')
    return df.with_columns(pl.min_horizontal('i1', 'i2').alias('p'), pl.max_horizontal('i1', 'i2').alias('q')).drop('i1', 'i2')
lab = to_pq(pl.read_csv(RAW + 'development_labels.csv')); pos = lab.filter(pl.col('label') == 1)
evp = to_pq(pl.read_csv(RAW + 'evaluation_pairs.csv'))
hands = pl.read_parquet(OUT + 'hands.parquet').select('hidx', 'hand_id', 'button_seat') if 'button_seat' in pl.read_parquet(OUT + 'hands.parquet', n_rows=1).columns else pl.read_parquet(OUT + 'hands.parquet').select('hidx', 'hand_id')
ev = pl.read_csv(RAW + 'development_evidence.csv').join(hands.select('hidx', 'hand_id'), on='hand_id').select('pair_id', 'hidx')
AF = pl.read_parquet(OUT + 'actions_feat.parquet')
AS = pl.read_parquet(OUT + 'actions_scored.parquet').select('hidx', 'action_no', 'surp', 'p_fold', 'p_pass', 'p_agg', 'size_z', 'last_agg_pidx')
AF = AF.join(AS, on=['hidx', 'action_no']).sort(['hidx', 'action_no'])
AF = AF.with_columns((pl.col('amount').cum_sum().over(['hidx', 'pidx']) - pl.col('amount')).alias('inv_before'))
S = pl.read_parquet(OUT + 'seats_feat.parquet')
# family rule for every pair (same rule as 04b/05b) and stage-1 rank for candidate selection
_t5 = lambda c: pl.col(c).sort(descending=True).head(5).mean()
HSC = pl.concat([pl.scan_parquet(f).select('hidx', 'p', 'q', 'is_eval', 's_ev', 's_dt', 's_sp', 's_ci').collect() for f in sorted(glob.glob(OUT + 'handscores/chunk*.parquet'))])
FAMRULE = HSC.group_by(['p', 'q', 'is_eval']).agg(_t5('s_dt').alias('a_dt'), _t5('s_sp').alias('a_sp'), _t5('s_ci').alias('a_ci'))
FAMRULE = FAMRULE.with_columns(pl.when((pl.col('a_dt') >= pl.col('a_sp')) & (pl.col('a_dt') >= pl.col('a_ci'))).then(pl.lit(FAMS[0]))
                               .when(pl.col('a_sp') >= pl.col('a_ci')).then(pl.lit(FAMS[1])).otherwise(pl.lit(FAMS[2])).alias('rule_fam')).select('p', 'q', 'is_eval', 'rule_fam')
HSC = HSC.with_columns(pl.col('s_ev').rank('ordinal', descending=True).over(['p', 'q', 'is_eval']).alias('r1'))
CAND_EVAL = HSC.filter((pl.col('is_eval') == 1) & (pl.col('r1') <= NCAND)).select('hidx', 'p', 'q').join(evp.select('p', 'q'), on=['p', 'q'])
CAND_DEV = HSC.filter(pl.col('is_eval') == 0).select('hidx', 'p', 'q').join(lab.select('p', 'q'), on=['p', 'q'])
CAND_DEV_UNLAB = HSC.filter((pl.col('is_eval') == 0) & (pl.col('r1') <= NCAND)).select('hidx', 'p', 'q').join(lab.select('p', 'q'), on=['p', 'q'], how='anti')
log('candidates dev', CAND_DEV.height, 'eval', CAND_EVAL.height)

ACT_F = ['st', 'relpos', 'hi', 'lo', 'suited', 'pocket', 'gap', 'eq_pf', 'hs', 'hc', 'potodds', 'tocall_bb', 'pot_bb', 'stack_bb', 'tocall_stack', 'spr',
         'players_active', 'n_agg_hand', 'n_agg_st', 'own_agg_before', 'n_act_st', 'self_last_agg', 'action_no', 'y', 'surp', 'p_fold', 'p_pass', 'p_agg', 'size_z',
         'amt_bb', 'amt_pot', 'inv_bb', 'b_active', 'b_folded_before', 'b_str', 'rel', 'facing', 'b_relpos', 'b_contrib_bb', 'a_net_bb', 'b_net_bb', 'n_third_active',
         'ps_open_raise', 'ps_vs_raise_fold', 'ps_post_agg', 'ps_post_fold_vs_bet'] + (['pair_tpos', 'pair_hand_idx'] if os.environ.get('ACT_CHRONO', '0') == '1' else [])

def build(cand):
    """action rows for pair members in candidate pair-hands"""
    sp = S.select('hidx', 'pidx', 'fold_no', 'eq_pf', 'hs1', 'hs2', 'hs3', 'relpos', 'total_contribution', 'net_chips')
    rows = []
    for me, other in [('p', 'q'), ('q', 'p')]:
        X = cand.join(AF.rename({'pidx': me}), on=['hidx', me])
        X = X.join(sp.rename({'pidx': other, 'fold_no': 'b_fold_no', 'eq_pf': 'b_eq', 'hs1': 'b_hs1', 'hs2': 'b_hs2', 'hs3': 'b_hs3', 'relpos': 'b_relpos',
                              'total_contribution': 'b_contrib', 'net_chips': 'b_net'}), on=['hidx', other])
        X = X.join(sp.select('hidx', pl.col('pidx').alias(me), pl.col('net_chips').alias('a_net')), on=['hidx', me])
        X = X.with_columns(pl.col(me).alias('a'), pl.col(other).alias('b'))
        rows.append(X.drop('p', 'q'))
    X = pl.concat(rows)
    X = X.with_columns(pl.min_horizontal('a', 'b').alias('p'), pl.max_horizontal('a', 'b').alias('q'))
    pre = pl.col('st') == 0
    X = X.with_columns(
        (pl.col('amount') / pl.col('big_blind')).alias('amt_bb'), (pl.col('amount') / pl.col('pot_before').clip(1)).alias('amt_pot'),
        (pl.col('inv_before') / pl.col('big_blind')).alias('inv_bb'),
        (pl.col('b_fold_no').is_null() | (pl.col('b_fold_no') > pl.col('action_no'))).cast(pl.Int8).alias('b_active'),
        (pl.col('b_fold_no').is_not_null() & (pl.col('b_fold_no') < pl.col('action_no'))).cast(pl.Int8).alias('b_folded_before'),
        pl.when(pre).then(pl.col('b_eq')).when(pl.col('st') == 1).then(pl.col('b_hs1')).when(pl.col('st') == 2).then(pl.col('b_hs2')).otherwise(pl.col('b_hs3')).alias('b_str'),
        pl.when(pl.col('last_agg_pidx').is_null()).then(0).when(pl.col('last_agg_pidx') == pl.col('a')).then(1).when(pl.col('last_agg_pidx') == pl.col('b')).then(2).otherwise(3).alias('facing'),
        (pl.col('b_contrib') / pl.col('big_blind')).alias('b_contrib_bb'), (pl.col('a_net') / pl.col('big_blind')).alias('a_net_bb'), (pl.col('b_net') / pl.col('big_blind')).alias('b_net_bb'))
    X = X.with_columns((pl.when(pre).then(pl.col('eq_pf')).otherwise(pl.col('hs')) - pl.col('b_str')).alias('rel'),
                       (pl.col('players_active') - 1 - pl.col('b_active')).alias('n_third_active'))
    if os.environ.get('ACT_CHRONO', '0') == '1':
        ordr = (X.select('hidx', 'p', 'q').unique().join(pl.read_parquet(OUT + 'hands.parquet').select('hidx', 't_order'), on='hidx')
                 .sort(['p', 'q', 't_order'])
                 .with_columns(pl.int_range(pl.len()).over(['p', 'q']).alias('pair_hand_idx'), pl.len().over(['p', 'q']).alias('_n')))
        ordr = ordr.with_columns((pl.col('pair_hand_idx') / pl.col('_n')).alias('pair_tpos')).select('hidx', 'p', 'q', 'pair_hand_idx', 'pair_tpos')
        X = X.join(ordr, on=['hidx', 'p', 'q'], how='left')
    return X.select('hidx', 'p', 'q', 'a', 'tidx', *ACT_F)

def aggregate(X, col):
    # aggregates plus a description of the single highest-scoring action: after the multiple-instance refinement the
    # model points at one action per hand, so what that action IS carries information for the reranker
    am = pl.col(col).arg_max()
    return X.group_by(['hidx', 'p', 'q']).agg(pl.col(col).max().alias('act_max'), pl.col(col).sum().alias('act_sum'), pl.col(col).mean().alias('act_mean'),
                                              (pl.col(col) > 0.5).sum().alias('act_n05'), pl.col(col).filter(pl.col('st') == 0).max().fill_null(0).alias('act_max_pre'),
                                              pl.col(col).filter(pl.col('st') > 0).max().fill_null(0).alias('act_max_post'),
                                              pl.col(col).filter(pl.col('b_active') == 1).max().fill_null(0).alias('act_max_bactive'),
                                              pl.col(col).top_k(2).min().alias('act_2nd'),
                                              pl.col('st').get(am).alias('top_st'), pl.col('y').get(am).alias('top_y'), pl.col('facing').get(am).alias('top_facing'),
                                              pl.col('surp').get(am).alias('top_surp'), pl.col('rel').get(am).alias('top_rel'),
                                              pl.col('b_active').get(am).alias('top_b_active'), pl.col('amt_bb').get(am).alias('top_amt_bb'),
                                              (pl.col('a_net_bb').get(am) > pl.col('b_net_bb').get(am)).cast(pl.Int8).alias('top_actor_won'),
                                              pl.col('players_active').get(am).alias('top_players_active'))

# ---- training rows: all labelled dev pairs' hands
D = build(CAND_DEV).join(lab.select('p', 'q', 'pair_id', 'label', 'behavior_family'), on=['p', 'q'])
D = D.join(ev.with_columns(pl.lit(1).alias('y_ev')), on=['pair_id', 'hidx'], how='left').with_columns(pl.col('y_ev').fill_null(0), (pl.col('tidx') % NF).alias('fold'))
D = D.join(FAMRULE.filter(pl.col('is_eval') == 0).drop('is_eval'), on=['p', 'q'], how='left').with_columns(pl.col('rule_fam').fill_null(FAMS[0]))
log('train action rows', D.height, 'positives', int(D['y_ev'].sum()))
PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_child_samples=50, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, scale_pos_weight=3.0, verbose=-1, num_threads=16, seed=SEED)
NR = 500
Xd = D.select(ACT_F).to_numpy().astype(np.float32); yd = D['y_ev'].to_numpy(); fo = D['fold'].to_numpy()
tf = D['behavior_family'].to_numpy(); rf = D['rule_fam'].to_numpy(); ispos = D['label'].to_numpy() == 1
MIL_ROUNDS = int(os.environ.get('MIL_ROUNDS', '1'))   # >1: keep only the best-scoring action per evidence hand as
MIL_KEEP = int(os.environ.get('MIL_KEEP', '2'))        # positive (multiple-instance refinement); the rest are dropped
hand_key = (D['hidx'].to_numpy().astype(np.int64) << 20) + D['p'].to_numpy().astype(np.int64) % (1 << 20)
oof = np.zeros(len(D)); models = {}; keep = np.ones(len(D), bool)
# HOLDOUT_FOLD: a strict test of MIL. That fold takes no part in any round (neither training nor label
# refinement); it is scored once at the end by a model trained on the other folds' final labels.
HOLD = int(os.environ.get('HOLDOUT_FOLD', '-1'))
inner = [f for f in range(NF) if f != HOLD]
for rnd in range(MIL_ROUNDS):
    oof = np.zeros(len(D)); models = {}
    for f in inner:
        for fam in FAMS:
            # train on positive pairs of this family (evidence vs their other hands) + all labelled-negative pairs' hands as extra negatives
            tr = (fo != f) & (fo != HOLD) & ((ispos & (tf == fam)) | (~ispos)) & keep
            m = lgb.train(PARAMS, lgb.Dataset(Xd[tr], yd[tr]), NR); models[(f, fam)] = m
            te = (fo == f) & (rf == fam)
            if te.sum(): oof[te] = m.predict(Xd[te])
        log('round', rnd, 'fold', f, 'done')
    if rnd + 1 < MIL_ROUNDS:
        # within each evidence hand keep the MIL_KEEP highest out-of-fold scoring actions as positives, drop the rest
        keep = np.ones(len(D), bool)
        pos_rows = np.where(yd == 1)[0]
        order = np.lexsort((-oof[pos_rows], hand_key[pos_rows]))
        pr = pos_rows[order]; hk = hand_key[pr]
        rank_in_hand = np.zeros(len(pr), int)
        start = 0
        for i in range(1, len(pr) + 1):
            if i == len(pr) or hk[i] != hk[start]:
                rank_in_hand[start:i] = np.arange(i - start); start = i
        keep[pr[rank_in_hand >= MIL_KEEP]] = False
        log(f'MIL round {rnd}: kept {int(keep[yd == 1].sum())} of {len(pos_rows)} positive action rows')
if HOLD >= 0:
    for fam in FAMS:
        tr = (fo != HOLD) & ((ispos & (tf == fam)) | (~ispos)) & keep
        m = lgb.train(PARAMS, lgb.Dataset(Xd[tr], yd[tr]), NR); models[(HOLD, fam)] = m
        te = (fo == HOLD) & (rf == fam)
        if te.sum(): oof[te] = m.predict(Xd[te])
    log('holdout fold', HOLD, 'scored by a model that never touched it')
D = D.with_columns(pl.Series('act_p', oof))
Hd = aggregate(D, 'act_p')
# hand-level evaluation inside positive pairs
Hp = Hd.join(pos.select('p', 'q', 'pair_id'), on=['p', 'q']).join(ev.with_columns(pl.lit(1).alias('is_ev')), on=['pair_id', 'hidx'], how='left').with_columns(pl.col('is_ev').fill_null(0))
truth = ev.group_by('pair_id').agg(pl.col('hidx').alias('truth'))
def map5(df, col):
    r = df.sort(['pair_id', col], descending=[False, True]).group_by('pair_id', maintain_order=True).agg(pl.col('hidx').head(5).alias('pred')).join(truth, on='pair_id')
    out = []
    for pred, tru in zip(r['pred'].to_list(), r['truth'].to_list()):
        t = set(tru); h = 0; s = 0.0
        for i, x in enumerate(pred):
            if x in t: h += 1; s += h / (i + 1)
        out.append(s / min(len(t), 5))
    return float(np.mean(out))
from sklearn.metrics import roc_auc_score
cv = {c: map5(Hp, c) for c in ['act_max', 'act_sum', 'act_max_bactive', 'act_2nd']}
if HOLD >= 0:
    Ht = Hp.join(D.select('hidx', 'tidx').unique('hidx'), on='hidx')
    Hh = Ht.filter(pl.col('tidx') % NF == HOLD)
    cv['holdout_act_max'] = map5(Hh, 'act_max'); cv['holdout_pairs'] = Hh['pair_id'].n_unique()
    cv['inner_act_max'] = map5(Ht.filter(pl.col('tidx') % NF != HOLD), 'act_max')
cv['auc_act_max'] = roc_auc_score(Hp['is_ev'].to_numpy(), Hp['act_max'].to_numpy())
log('CV (positive pairs, OOF)', cv); json.dump(cv, open(OUT + 'actmodel_cv.json', 'w'), indent=1)
imp = sorted(zip(ACT_F, models[(0, 'coordinated_isolation')].feature_importance('gain')), key=lambda t: -t[1])[:15]
print('top features (iso model):', [(a, round(b)) for a, b in imp])
# A CV run must not clobber the artifact set -- but it must also not leave a file inside AD, which the
# downstream steps glob wholesale: a second copy of the dev rows silently duplicates every candidate hand
# and inflates MAP@5 (each true evidence hand can then be "found" twice in a top-5).
Hd.write_parquet((OUT + 'actmodel_cvonly_dev.parquet') if os.environ.get('CV_ONLY') == '1' else (AD + 'dev.parquet'))
if os.environ.get('CV_ONLY') == '1':
    log('CV_ONLY: skipping unlabelled-dev and evaluation scoring'); raise SystemExit(0)
# ---- remaining candidates (unlabelled dev pairs: out-of-fold by table; eval pairs: 5-fold mean), chunked by table
tmap = S.select('hidx', 'tidx').unique()
for phase, cand, tag in [(0, CAND_DEV_UNLAB, 'devunlab'), (1, CAND_EVAL, 'eval')]:
    for ch in range(NCH):
        C = cand.join(tmap, on='hidx').filter(pl.col('tidx') % NCH == ch).drop('tidx')
        if C.height == 0: continue
        FSRC = FAMRULE.filter(pl.col('is_eval') == phase).drop('is_eval')
        X = build(C).join(FSRC, on=['p', 'q'], how='left').with_columns(pl.col('rule_fam').fill_null(FAMS[0]))
        Xe = X.select(ACT_F).to_numpy().astype(np.float32); rfe = X['rule_fam'].to_numpy(); pr = np.zeros(len(X))
        foldv = (X['tidx'].to_numpy() % NF) if phase == 0 else None
        for fam in FAMS:
            sel = rfe == fam
            if not sel.sum(): continue
            if phase == 0:
                for f2 in range(NF):
                    s2 = sel & (foldv == f2)
                    if s2.sum(): pr[s2] = models[(f2, fam)].predict(Xe[s2])
            else:
                pr[sel] = np.mean([models[(f, fam)].predict(Xe[sel]) for f in range(NF)], axis=0)
        aggregate(X.with_columns(pl.Series('act_p', pr)), 'act_p').write_parquet(AD + f'{tag}{ch:02d}.parquet')
        log(tag, 'chunk', ch, 'rows', len(X))
log('done')
