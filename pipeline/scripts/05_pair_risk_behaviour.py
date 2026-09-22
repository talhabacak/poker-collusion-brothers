"""Step 5: pair-level aggregation, risk model, behavior model, local metrics, submission."""
import polars as pl, numpy as np, lightgbm as lgb, time, glob, json, os
from sklearn.metrics import average_precision_score, roc_auc_score

def _assert_unique_keys(df, name):   # a stale extra file in a globbed artefact dir duplicates rows and inflates MAP@5
    if df is not None and df.select('hidx', 'p', 'q').n_unique() != df.height:
        raise SystemExit(f'{name}: {df.height - df.select("hidx", "p", "q").n_unique()} duplicate (hidx,p,q) rows - stale file in the artefact directory?')
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; RAW = 'data/raw/'; SEED = 42; NF = 5
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
players = pl.read_parquet(OUT + 'players.parquet').select('player_id', 'pidx')
def to_pq(df):
    df = df.join(players.rename({'player_id': 'player_1', 'pidx': 'i1'}), on='player_1').join(players.rename({'player_id': 'player_2', 'pidx': 'i2'}), on='player_2')
    return df.with_columns(pl.min_horizontal('i1', 'i2').alias('p'), pl.max_horizontal('i1', 'i2').alias('q')).drop('i1', 'i2')
lab = to_pq(pl.read_csv(RAW + 'development_labels.csv'))
evp = to_pq(pl.read_csv(RAW + 'evaluation_pairs.csv'))
hands = pl.read_parquet(OUT + 'hands.parquet').select('hidx', 'hand_id')
ev = pl.read_csv(RAW + 'development_evidence.csv').join(hands, on='hand_id').select('pair_id', 'hidx')
pos_players = pl.concat([lab.filter(pl.col('label') == 1).select(pl.col('p').alias('x')), lab.filter(pl.col('label') == 1).select(pl.col('q').alias('x'))]).unique()['x']

HS = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'handscores/chunk*.parquet'))])
if os.environ.get('USE_CT', '1') == '1':
    CTP = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'collusion/chunk*.parquet'))]).select('hidx','p','q','ct_mutual','ct_asym','ct_max','ct_pos')
    HS = HS.join(CTP, on=['hidx','p','q'], how='left').with_columns([pl.col(c).fill_null(0.0) for c in ['ct_mutual','ct_asym','ct_max','ct_pos']])
if os.environ.get('USE_VAL', '1') == '1':
    VALP = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'value/chunk*.parquet'))]).select('hidx', 'p', 'q', 'vL_under_to_partner', 'vW_over_to_partner', 'vL_over_to_partner', 'vL_under_hu', 'vW_over_vs_third', 'vL_pot_given_up')
    HS = HS.join(VALP, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0.0) for c in ['vL_under_to_partner', 'vW_over_to_partner', 'vL_over_to_partner', 'vL_under_hu', 'vW_over_vs_third', 'vL_pot_given_up']])
if os.environ.get('USE_ACT_PAIR', '1') == '1':
    # step 4c action-level scores. Coverage differs by pair type (labelled dev pairs are scored on every hand,
    # the rest only on their top-60 candidates), so aggregates are restricted to the top-60 for every pair.
    ACTP = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'actmodel/*.parquet'))]).select('hidx', 'p', 'q', 'act_max', 'act_sum')
    _assert_unique_keys(ACTP, 'ACTP')
    HS = HS.join(ACTP, on=['hidx', 'p', 'q'], how='left')
    HS = HS.with_columns(pl.col('s_ev').rank('ordinal', descending=True).over(['p', 'q', 'is_eval']).alias('_r60'))
    HS = HS.with_columns(pl.when(pl.col('_r60') <= 60).then(pl.col('act_max')).otherwise(None).fill_null(0.0).alias('act_max'),
                         pl.when(pl.col('_r60') <= 60).then(pl.col('act_sum')).otherwise(None).fill_null(0.0).alias('act_sum')).drop('_r60')
if os.environ.get('USE_S1B') == '1':   # tested and reverted: no real gain once dev scores are out-of-fold
    S1B = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'stage1b/*.parquet'))])
    HS = HS.join(S1B, on=['hidx', 'p', 'q'], how='left').with_columns(pl.col('s1b').fill_null(-99.0))
log('hand scores', HS.shape)
import os


TRUNC = float(os.environ.get('TRUNC_DEV', '0'))  # keep only the last TRUNC share of each dev pair's hands, to match the
if TRUNC > 0:                                     # evaluation period (40% of hands) when training the pair model
    HS = HS.sort(['p', 'q', 'is_eval', 't_order']).with_columns(
        (pl.int_range(pl.len()).over(['p', 'q', 'is_eval']) / pl.len().over(['p', 'q', 'is_eval'])).alias('_pos'))
    HS = HS.filter((pl.col('is_eval') == 1) | (pl.col('_pos') >= 1 - TRUNC)).drop('_pos')
    log('dev truncated to last', TRUNC, '->', HS.filter(pl.col('is_eval') == 0).height, 'dev pair-hands')

VAL_AGGS = []; CT_AGGS = []
def pair_features(H):
    global VAL_AGGS
    _t = lambda c, k: pl.col(c).sort(descending=True).head(k).mean()
    global CT_AGGS
    CT_AGGS = ([_t('ct_asym', 5).alias('ct_asym_top5'), _t('ct_pos', 5).alias('ct_pos_top5'), pl.col('ct_pos').mean().alias('ct_pos_mean'),
                _t('ct_max', 5).alias('ct_max_top5'), pl.col('ct_mutual').mean().alias('ct_mut_mean'), _t('ct_mutual', 5).alias('ct_mut_top5')]
               if 'ct_asym' in H.columns else [])
    VAL_AGGS = ([_t('vL_under_to_partner', 5).alias('val_under_top5'), _t('vW_over_to_partner', 5).alias('val_over_top5'),
                 _t('vL_over_to_partner', 5).alias('val_Lover_top5'), _t('vL_under_hu', 5).alias('val_underhu_top5'),
                 _t('vW_over_vs_third', 5).alias('val_third_top5'), _t('vL_pot_given_up', 5).alias('val_potgiven_top5')]
                if 'vL_under_to_partner' in H.columns else [])
    g = ['p', 'q', 'is_eval']
    H = H.sort(g + ['t_order']).with_columns(pl.col('s_ev').rolling_mean(15, center=True, min_samples=1).over(g).alias('roll15_ev'),
                                             pl.col('s_ev').rank('ordinal', descending=True).over(g).alias('ev_rank'))
    top = lambda c, k: pl.col(c).sort(descending=True).head(k).mean()
    tk = pl.col('ev_rank') <= 5
    F = H.group_by(g).agg(
        pl.len().alias('n'), pl.col('tidx').first(),
        pl.col('s_ev').max().alias('ev_top1'), top('s_ev', 3).alias('ev_top3'), top('s_ev', 5).alias('ev_top5'), top('s_ev', 10).alias('ev_top10'),
        pl.col('s_ev').mean().alias('ev_mean'), (pl.col('s_ev') > 0.2).sum().alias('ev_n02'), (pl.col('s_ev') > 0.5).sum().alias('ev_n05'),
        pl.col('roll15_ev').max().alias('ev_roll15_max'),
        top('s_dt', 5).alias('dt_top5'), top('s_sp', 5).alias('sp_top5'), top('s_ci', 5).alias('ci_top5'),
        pl.col('act_max').max().alias('actp_top1'), top('act_max', 5).alias('actp_top5'), top('act_max', 10).alias('actp_top10'),
        (pl.col('act_max') > 0.5).sum().alias('actp_n05'), top('act_sum', 5).alias('actsum_top5'),
        *VAL_AGGS, *CT_AGGS,
        top('rel_llr_pos', 5).alias('rel_pos_top5'), pl.col('rel_llr_tot').mean().alias('rel_tot_mean'), top('rel_llr_max', 5).alias('rel_max_top5'),
        pl.col('s_dt').sum().alias('dt_sum'), pl.col('s_sp').sum().alias('sp_sum'), pl.col('s_ci').sum().alias('ci_sum'),
        top('tot_surp', 5).alias('surp_top5'), pl.col('tot_surp').mean().alias('surp_mean'), top('tot_surp_max', 5).alias('surpmax_top5'),
        *[pl.col(c).filter(tk).mean().alias('t5_' + c) for c in ['pot_bb', 'flow_bb', 'both_sd', 'one_fold', 'neither_fold', 'board_len', 'n_third_fold_vs_pair', 'both_vpip',
                                                                 'opp_sign', 'W_n_agg_multi', 'L_n_agg_multi', 'W_surp_pass_ahead', 'L_surp_pass_ahead', 'L_dump_value', 'L_fold_value',
                                                                 'n3_fold_to_pair', 'third_dead_bb', 'n_pair_agg_multiway']],
    )
    F = F.with_columns((pl.col('ev_n02') / pl.col('n')).alias('ev_rate02'), pl.col('n').log().alias('log_n'),
                       )
    F = F.join(pl.read_parquet(OUT + 'pair_relational.parquet'), on=['p', 'q', 'is_eval'], how='left')  # step 3b pair-level G
    if os.environ.get('USE_OA', '0') == '1':   # step 13: opponent-aware policy residuals (their largest late gain)
        O = pl.read_parquet(OUT + 'oppaware_pair.parquet')
        keep = os.environ.get('OA_COLS', '')
        if keep: O = O.select('p', 'q', 'is_eval', *keep.split(','))
        F = F.join(O, on=['p', 'q', 'is_eval'], how='left').with_columns(
            [pl.col(c).fill_null(0.0) for c in O.columns if c.startswith('oa_')])
    if os.environ.get('USE_OC', '0') == '1':   # step 15: opponent-CONDITIONED residuals
        Q = pl.read_parquet(OUT + 'oppcond_pair.parquet')
        F = F.join(Q, on=['p', 'q', 'is_eval'], how='left').with_columns(
            [pl.col(c).fill_null(0.0) for c in Q.columns if c.startswith('oc_')])
    if os.environ.get('USE_CTR', '0') == '1':   # step 9b: does the partner help this player more than the player's other opponents do
        R = pl.read_parquet(OUT + 'ct_relational.parquet')
        keep = os.environ.get('CTR_COLS', '')        # all twelve overfit 1,860 labelled pairs; a two-feature subset is the alternative
        if keep: R = R.select('p', 'q', 'is_eval', *keep.split(','))
        F = F.join(R, on=['p', 'q', 'is_eval'], how='left').with_columns(
            [pl.col(c).fill_null(0.0) for c in R.columns if c.startswith('ctr_')])
    # player-level normalization: how much does this pair stand out among each player's other pairs in the same phase
    for key in ['ev_top5', 'surp_top5', 'G_rate']:
        long = pl.concat([F.select('p', 'q', 'is_eval', pl.col('p').alias('x'), key), F.select('p', 'q', 'is_eval', pl.col('q').alias('x'), key)])
        long = long.with_columns(pl.col(key).rank('ordinal', descending=True).over(['x', 'is_eval']).alias('_r'),
                                 pl.col(key).sort(descending=True).first().over(['x', 'is_eval']).alias('_b1'),
                                 pl.col(key).sort(descending=True).slice(1, 1).first().over(['x', 'is_eval']).alias('_b2'),
                                 pl.col(key).median().over(['x', 'is_eval']).alias('_med'))
        long = long.with_columns(pl.when(pl.col('_r') == 1).then(pl.col(key) - pl.col('_b2')).otherwise(pl.col(key) - pl.col('_b1')).alias('_margin'),
                                 (pl.col(key) - pl.col('_med')).alias('_above_med'))
        agg = long.group_by(['p', 'q', 'is_eval']).agg(pl.col('_margin').min().alias(f'{key}_margin_min'), pl.col('_margin').max().alias(f'{key}_margin_max'),
                                                       pl.col('_r').max().alias(f'{key}_rank_worst'), pl.col('_above_med').min().alias(f'{key}_above_med_min'))
        F = F.join(agg, on=['p', 'q', 'is_eval'])
    # table-level context: how this pair ranks inside its own pool (off by default: LB-neutral, kept for experiments)
    for key in ([] if os.environ.get('TABLE_FEATS', '0') != '1' else ['ev_top5', 'surp_top5']):
        F = F.with_columns(pl.col(key).rank('ordinal', descending=True).over(['tidx', 'is_eval']).alias(f'{key}_trank'),
                           (pl.col(key) - pl.col(key).median().over(['tidx', 'is_eval'])).alias(f'{key}_tmargin'))
    return F, H

F, HS = pair_features(HS)
PAIR_FEATS = [c for c in F.columns if c not in ('p', 'q', 'is_eval', 'tidx', 'n', 'ev_n02', 'ev_n05', 'actp_n05')]
log('pair features', F.shape, len(PAIR_FEATS))

dev = F.filter(pl.col('is_eval') == 0).join(lab.select('p', 'q', 'pair_id', 'label', 'behavior_family'), on=['p', 'q'], how='left')
dev = dev.with_columns((pl.col('tidx') % NF).alias('fold'), pl.col('label').is_not_null().alias('labeled'),
                       (pl.col('p').is_in(pos_players.implode()) | pl.col('q').is_in(pos_players.implode())).alias('has_pos_player'))
# eval-like unlabeled dev pool: not labeled and no public-positive player (mirrors how evaluation_pairs was built)
U = dev.filter(~pl.col('labeled') & ~pl.col('has_pos_player'))
# 'colluder + innocent opponent' pairs: near-certain negatives (93% of public positives have a single partner) and
# exactly the confusion that matters in evaluation, where a hidden colluder's other opponents are scored too
n_pos_players = pl.col('p').is_in(pos_players.implode()).cast(pl.Int8) + pl.col('q').is_in(pos_players.implode()).cast(pl.Int8)
MIX = dev.filter(~pl.col('labeled') & (n_pos_players == 1))
MIX_W = float(os.environ.get('MIXED_NEG_W', '0.2'))
log('mixed colluder+innocent pairs', MIX.height, 'weight', MIX_W)
Lb = dev.filter(pl.col('labeled'))
log('labeled', Lb.shape, 'unlabeled eval-like dev pairs', U.shape)

def fit_predict(train, test_sets, target, weight=None, obj='binary', num_class=None, rounds=300, seed=SEED):
    params = dict(objective=obj, learning_rate=0.03, num_leaves=15, min_child_samples=20, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=5.0, verbose=-1, num_threads=16, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    if num_class: params['num_class'] = num_class
    ds = lgb.Dataset(train.select(PAIR_FEATS).to_numpy().astype(np.float32), target, weight=weight)
    m = lgb.train(params, ds, rounds)
    return m, [m.predict(t.select(PAIR_FEATS).to_numpy().astype(np.float32)) for t in test_sets]

def run_cv(u_weight, u_frac=0.3):
    rng = np.random.default_rng(SEED)
    oof_L = np.zeros(len(Lb)); oof_U = np.zeros(len(U))
    fL = Lb['fold'].to_numpy(); fU = U['fold'].to_numpy()
    usel = rng.random(len(U)) < u_frac
    for f in range(NF):
        trL = Lb.filter(pl.col('fold') != f)
        parts = [trL]; yt = [trL['label'].to_numpy()]; wt = [np.ones(len(trL))]
        if MIX_W > 0:
            trM = MIX.filter(pl.col('fold') != f); parts.append(trM); yt.append(np.zeros(len(trM))); wt.append(np.full(len(trM), MIX_W))
        if u_weight > 0:
            trU = U.filter(pl.Series((fU != f) & usel)); parts.append(trU); yt.append(np.zeros(len(trU))); wt.append(np.full(len(trU), u_weight))
        tr = pl.concat(parts)
        m, (pL, pU) = fit_predict(tr, [Lb.filter(pl.col('fold') == f), U.filter(pl.col('fold') == f)], np.concatenate(yt), np.concatenate(wt))
        oof_L[fL == f] = pL; oof_U[fU == f] = pU
    return oof_L, oof_U

yL = Lb['label'].to_numpy()
results = {}
for uw in [0.0, 0.05, 0.2]:
    oL, oU = run_cv(uw)
    ap_lab = average_precision_score(yL, oL)
    y_all = np.concatenate([yL, np.zeros(len(oU))]); s_all = np.concatenate([oL, oU])
    ap_pu = average_precision_score(y_all, s_all)
    results[uw] = (ap_lab, ap_pu, oL, oU)
    log(f'u_weight={uw}: Pair AP labeled-only={ap_lab:.4f}  AUC={roc_auc_score(yL, oL):.4f} | Pair AP vs labeled+{len(oU)} unlabeled-as-neg={ap_pu:.4f}')
best_uw = float(os.environ['U_WEIGHT']) if os.environ.get('U_WEIGHT') else max(results, key=lambda k: results[k][1] + results[k][0])
_, _, oL, oU = results[best_uw]
log('chosen u_weight', best_uw)
# baselines for reference
for c in ['ev_top5', 'surp_top5', 'ev_top5_margin_min']:
    s = Lb[c].to_numpy(); y_all = np.concatenate([yL, np.zeros(len(U))]); s_all = np.concatenate([s, U[c].to_numpy()])
    log(f'single feature {c}: AP labeled={average_precision_score(yL, s):.4f}, AP PU={average_precision_score(y_all, s_all):.4f}')

# behavior model on public positives (3 classes), OOF
P = Lb.filter(pl.col('label') == 1)
fam_idx = P['behavior_family'].replace_strict({f: i for i, f in enumerate(FAMS)}).to_numpy()
oofF = np.zeros((len(P), 3)); fP = P['fold'].to_numpy()
famL = np.zeros((len(Lb), 3)); famU = np.zeros((len(U), 3))
for f in range(NF):
    m, (pp, pl_, pu) = fit_predict(P.filter(pl.col('fold') != f), [P.filter(pl.col('fold') == f), Lb.filter(pl.col('fold') == f), U.filter(pl.col('fold') == f)],
                                  fam_idx[fP != f], obj='multiclass', num_class=3, rounds=200)
    oofF[fP == f] = pp; famL[Lb['fold'].to_numpy() == f] = pl_; famU[U['fold'].to_numpy() == f] = pu
acc = (oofF.argmax(1) == fam_idx).mean()
log(f'behavior accuracy on public positives (OOF): {acc:.4f}')
print(pl.DataFrame({'true': [FAMS[i] for i in fam_idx], 'pred': [FAMS[i] for i in oofF.argmax(1)]}).group_by(['true', 'pred']).len().sort(['true', 'pred']))

if os.environ.get('BEH_LOFO') == '1':
    # Leave-one-family-out for the behaviour head: is an UNSEEN family recognisable by the head being unsure?
    # For each family X: train on the other two (2-class), OOF-predict those two (in-family), predict X (unseen).
    # The 'unsure' score is the top-class probability; report unseen-family recall vs in-family false flags per threshold.
    rows = []
    for k_out, fam_out in enumerate(FAMS):
        Pin = P.filter(pl.col('behavior_family') != fam_out); Pout = P.filter(pl.col('behavior_family') == fam_out)
        others = [f for f in FAMS if f != fam_out]
        yin = Pin['behavior_family'].replace_strict({f: i for i, f in enumerate(others)}).to_numpy()
        fin = Pin['fold'].to_numpy(); oof_in = np.zeros((len(Pin), 2)); p_out = np.zeros((len(Pout), 2))
        for f in range(NF):
            m, (a, b) = fit_predict(Pin.filter(pl.col('fold') != f), [Pin.filter(pl.col('fold') == f), Pout], yin[fin != f], obj='multiclass', num_class=2, rounds=200)
            oof_in[fin == f] = a; p_out += b / NF
        rows.append((fam_out, oof_in.max(1), p_out.max(1)))
    print('\nBEH_LOFO: unseen-family recall vs known-family false-flag rate, by top-class-probability threshold')
    print(f'{"thr":>6s} ' + ' '.join(f'{fo[:12]:>12s}' for fo, _, _ in rows) + f' {"unseen recall":>14s} {"known flagged":>14s}')
    for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
        rec = [float((po < thr).mean()) for _, _, po in rows]; fp = [float((pi < thr).mean()) for _, pi, _ in rows]
        print(f'{thr:6.2f} ' + ' '.join(f'{r:12.3f}' for r in rec) + f' {np.mean(rec):14.3f} {np.mean(fp):14.3f}')
    print('(2-class head: probabilities are more confident than the 3-class head, so read thresholds relatively)\n')
def behavior_map(fam_true, risk, fam_pred):
    aps = []
    for k, fam in enumerate(FAMS):
        yk = (fam_true == fam).astype(int); sk = np.where(fam_pred == k, risk, 0.0)
        aps.append(average_precision_score(yk, sk) if yk.sum() else 0.0)
    return float(np.mean(aps)), aps
ft_L = Lb['behavior_family'].to_numpy()
pl.DataFrame({'p': Lb['p'], 'q': Lb['q'], 'pred_fam': [FAMS[i] for i in famL.argmax(1)]}).write_parquet(OUT + 'pair_behavior_oof.parquet')
bm_lab, bm_lab_each = behavior_map(ft_L, oL, famL.argmax(1))
bm_pu, bm_pu_each = behavior_map(np.concatenate([ft_L, np.array(['none'] * len(U))]), np.concatenate([oL, oU]), np.concatenate([famL.argmax(1), famU.argmax(1)]))
log(f'Behavior MAP labeled-only={bm_lab:.4f} {np.round(bm_lab_each, 3)} | PU={bm_pu:.4f} {np.round(bm_pu_each, 3)}')

# evidence MAP@5 (OOF scorer) using s_ev vs predicted-family prob
truth = ev.group_by('pair_id').agg(pl.col('hidx').alias('truth'))
Hl = HS.filter(pl.col('is_eval') == 0).join(Lb.filter(pl.col('label') == 1).select('p', 'q', 'pair_id'), on=['p', 'q'])
predfam = pl.DataFrame({'pair_id': P['pair_id'], 'pf': [['s_dt', 's_sp', 's_ci'][i] for i in oofF.argmax(1)]})
Hl = Hl.join(predfam, on='pair_id').with_columns(pl.when(pl.col('pf') == 's_dt').then(pl.col('s_dt')).when(pl.col('pf') == 's_sp').then(pl.col('s_sp')).otherwise(pl.col('s_ci')).alias('s_pf'))
Hl = Hl.with_columns((pl.col('s_ev') + pl.col('s_pf')).alias('s_mix'))
def map5(col):
    r = Hl.sort(col, descending=True).group_by('pair_id', maintain_order=True).agg(pl.col('hidx').head(5).alias('pred')).join(truth, on='pair_id')
    out = []
    for pred, tru in zip(r['pred'].to_list(), r['truth'].to_list()):
        t = set(tru); h = 0; s = 0.0
        for i, x in enumerate(pred):
            if x in t: h += 1; s += h / (i + 1)
        out.append(s / min(len(t), 5))
    return float(np.mean(out))
emap = {c: map5(c) for c in ['s_ev', 's_pf', 's_mix', 'tot_surp']}
log('Evidence MAP@5:', emap)
ev_col = max(emap, key=emap.get)

summary = {'pair_ap_labeled': results[best_uw][0], 'pair_ap_pu': results[best_uw][1], 'behavior_map_labeled': bm_lab, 'behavior_map_pu': bm_pu,
           'evidence_map5': emap[ev_col], 'evidence_rank_col': ev_col, 'u_weight': best_uw, 'behavior_acc': acc,
           'mean3_labeled': (results[best_uw][0] + bm_lab + emap[ev_col]) / 3, 'mean3_pu': (results[best_uw][1] + bm_pu + emap[ev_col]) / 3}
log('SUMMARY', json.dumps(summary, indent=1))
json.dump(summary, open(OUT + 'cv_summary.json', 'w'), indent=1)

# ---------------- robustness: pair features from a 67% random subsample of dev hands (eval has fewer shared hands) ----------------
rng = np.random.default_rng(SEED)
HSsub = HS.filter(pl.col('is_eval') == 0).filter(pl.Series(rng.random(HS.filter(pl.col('is_eval') == 0).height) < 0.67)).drop('roll15_ev', 'ev_rank')
Fs, _ = pair_features(HSsub)
Ls = Lb.select('p', 'q', 'label', 'fold').join(Fs.drop('tidx'), on=['p', 'q'])
oofs = np.zeros(len(Ls))
for f in range(NF):
    tr = Lb.filter(pl.col('fold') != f); m, _ = fit_predict(tr, [], tr['label'].to_numpy())
    idx = Ls['fold'].to_numpy() == f
    oofs[idx] = m.predict(Ls.filter(pl.Series(idx)).select(PAIR_FEATS).to_numpy().astype(np.float32))
log(f'robustness: model trained on full-window features, scored on 67% hand subsample: AP={average_precision_score(Ls["label"].to_numpy(), oofs):.4f}')

# ---------------- optional PU self-training: treat confident unlabelled pairs as positives ----------------
import os
PU_SELF = os.environ.get('PU_SELFTRAIN', '0') == '1'
SEEDS = [int(x) for x in os.environ.get('SEEDS', '42').split(',')]
SUBFILE = os.environ.get('SUB_OUT', 'submission.csv')
if PU_SELF:
    rng2 = np.random.default_rng(SEED)
    oU_cur = oU.copy()
    for it in range(2):
        hi = oU_cur > 0.9; lo = oU_cur < 0.01
        log(f'self-training round {it}: pseudo-positives {hi.sum()}, confident negatives {lo.sum()}')
        newL = np.zeros(len(U)); f_all = U['fold'].to_numpy()
        for f in range(NF):
            trU = U.filter(pl.Series((f_all != f) & (hi | lo)))
            wU = np.where(oU_cur[(f_all != f) & (hi | lo)] > 0.9, 1.0, 0.3)
            yU = (oU_cur[(f_all != f) & (hi | lo)] > 0.9).astype(float)
            trL = Lb.filter(pl.col('fold') != f)
            tr = pl.concat([trL, trU])
            m, (pL, pU2) = fit_predict(tr, [Lb.filter(pl.col('fold') == f), U.filter(pl.col('fold') == f)],
                                       np.concatenate([trL['label'].to_numpy(), yU]), np.concatenate([np.ones(len(trL)), wU]))
            oL[Lb['fold'].to_numpy() == f] = pL; newL[f_all == f] = pU2
        oU_cur = newL
        log(f'   after round {it}: labeled-only AP={average_precision_score(yL, oL):.4f}, pseudo-positives now {(oU_cur > 0.9).sum()}')
    oU = oU_cur

if os.environ.get('DUMP_DEV') == '1':   # out-of-fold risk for every dev pair, to calibrate the unsupervised blend offline
    pl.concat([pl.DataFrame({'p': Lb['p'], 'q': Lb['q'], 'label': yL.astype(np.int32), 'risk': oL}),
               pl.DataFrame({'p': U['p'], 'q': U['q'], 'label': np.zeros(len(U), np.int32), 'risk': oU})]).write_parquet(OUT + 'pair_risk_dev_oof.parquet')
    log('dev OOF risk dumped')
# ---------------- final fit + submission ----------------
E = F.filter(pl.col('is_eval') == 1)
E = evp.select('pair_id', 'p', 'q').join(E, on=['p', 'q'], how='left')
log('eval pairs matched', E.filter(pl.col('n').is_not_null()).height, '/', E.height)
risk = np.zeros(len(E)); famE = np.zeros((len(E), 3))
rngu = np.random.default_rng(SEED); usel = rngu.random(len(U)) < 0.3
rank_acc = np.zeros(len(E))
for f in range(NF):
    trL = Lb.filter(pl.col('fold') != f); parts = [trL]; yt = [trL['label'].to_numpy()]; wt = [np.ones(len(trL))]
    if MIX_W > 0:
        trM = MIX.filter(pl.col('fold') != f); parts.append(trM); yt.append(np.zeros(len(trM))); wt.append(np.full(len(trM), MIX_W))
    if best_uw > 0:
        trU = U.filter(pl.Series((U['fold'].to_numpy() != f) & usel)); parts.append(trU); yt.append(np.zeros(len(trU))); wt.append(np.full(len(trU), best_uw))
    if PU_SELF:
        hi = oU > 0.9; lo = oU < 0.01; sel = (U['fold'].to_numpy() != f) & (hi | lo)
        trU2 = U.filter(pl.Series(sel)); parts.append(trU2)
        yt.append((oU[sel] > 0.9).astype(float)); wt.append(np.where(oU[sel] > 0.9, 1.0, 0.3))
    tr_all = pl.concat(parts); y_all_tr = np.concatenate(yt); w_all_tr = np.concatenate(wt)
    for sd in SEEDS:

        m, (pe,) = fit_predict(tr_all, [E], y_all_tr, w_all_tr, seed=sd)
        risk += pe / (NF * len(SEEDS)); rank_acc += pl.Series(pe).rank().to_numpy() / (NF * len(SEEDS) * len(E))
    m, (pf,) = fit_predict(P.filter(pl.col('fold') != f), [E], fam_idx[fP != f], obj='multiclass', num_class=3, rounds=200); famE += pf / NF
# deterministic tie-break from poker activity (top-5 surprisal), keeps order otherwise
tb = E['surp_top5'].fill_null(0).to_numpy(); tb = (tb - tb.min()) / (tb.max() - tb.min() + 1e-9)
risk = np.clip(risk * (1 - 1e-6) + tb * 1e-7, 0, 1)
pl.DataFrame({'pair_id': E['pair_id'], 'p_dt': famE[:, 0], 'p_sp': famE[:, 1], 'p_ci': famE[:, 2], 'risk': risk}).write_parquet(OUT + 'eval_behavior_probs.parquet')
E = E.with_columns(pl.Series('risk_score', risk), pl.Series('predicted_behavior', [FAMS[i] for i in famE.argmax(1)]), pl.Series('_pf', [['s_dt', 's_sp', 's_ci'][i] for i in famE.argmax(1)]))
He = HS.filter(pl.col('is_eval') == 1).join(E.select('p', 'q', 'pair_id', '_pf'), on=['p', 'q'])
He = He.with_columns(pl.when(pl.col('_pf') == 's_dt').then(pl.col('s_dt')).when(pl.col('_pf') == 's_sp').then(pl.col('s_sp')).otherwise(pl.col('s_ci')).alias('s_pf'))
He = He.with_columns((pl.col('s_ev') + pl.col('s_pf')).alias('s_mix'))
top5 = (He.sort([ev_col, 'hidx'], descending=[True, False]).group_by('pair_id', maintain_order=True).agg(pl.col('hidx').head(5))
        .explode('hidx').join(hands, on='hidx').group_by('pair_id', maintain_order=True).agg(pl.col('hand_id')))
sub = E.select('pair_id', 'risk_score', 'predicted_behavior').join(top5, on='pair_id', how='left')
sub = sub.with_columns([pl.col('hand_id').list.get(i, null_on_oob=True).fill_null('NO_EVIDENCE').alias(f'evidence_hand_{i+1}') for i in range(5)]).drop('hand_id')
tmpl = pl.read_csv(RAW + 'sample_submission.csv').select('pair_id')
sub = tmpl.join(sub, on='pair_id', how='left')
assert sub.height == tmpl.height and sub.null_count().sum_horizontal()[0] == 0
sub.write_csv(SUBFILE)
log('submission written', SUBFILE, sub.shape)
print(sub['predicted_behavior'].value_counts())
print(sub.select(pl.col('risk_score').describe()) if hasattr(pl.Expr, 'describe') else sub['risk_score'].describe())
# pairs above various risk thresholds
for t in [0.5, 0.8, 0.9]:
    print('risk >', t, ':', (sub['risk_score'] > t).sum())
