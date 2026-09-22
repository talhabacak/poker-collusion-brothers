"""Step 4b: within-pair evidence ranker. Trained only inside public positive pairs with features normalised
inside each pair (z-score and rank percentile), so the model learns 'which hand of THIS pair is the planted one'
instead of 'which pair looks coordinated'. Writes s1b for every pair-hand of labelled dev pairs and all eval pairs."""
import polars as pl, numpy as np, glob, lightgbm as lgb, time, os, json

def _assert_unique_keys(df, name):   # a stale extra file in a globbed artefact dir duplicates rows and inflates MAP@5
    if df is not None and df.select('hidx', 'p', 'q').n_unique() != df.height:
        raise SystemExit(f'{name}: {df.height - df.select("hidx", "p", "q").n_unique()} duplicate (hidx,p,q) rows - stale file in the artefact directory?')

t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; RAW = 'data/raw/'; NF = 5; SEED = 42
players = pl.read_parquet(OUT + 'players.parquet').select('player_id', 'pidx')
def to_pq(df):
    df = df.join(players.rename({'player_id': 'player_1', 'pidx': 'i1'}), on='player_1').join(players.rename({'player_id': 'player_2', 'pidx': 'i2'}), on='player_2')
    return df.with_columns(pl.min_horizontal('i1', 'i2').alias('p'), pl.max_horizontal('i1', 'i2').alias('q')).drop('i1', 'i2')
lab = to_pq(pl.read_csv(RAW + 'development_labels.csv'))
pos = lab.filter(pl.col('label') == 1)
evp = to_pq(pl.read_csv(RAW + 'evaluation_pairs.csv'))
hands = pl.read_parquet(OUT + 'hands.parquet').select('hidx', 'hand_id')
ev = pl.read_csv(RAW + 'development_evidence.csv').join(hands, on='hand_id').select('pair_id', 'hidx', 'evidence_rank')
PH = sorted(glob.glob(OUT + 'pairhand/chunk*.parquet')); RE = sorted(glob.glob(OUT + 'relational/chunk*.parquet'))
REL = pl.concat([pl.read_parquet(f) for f in RE])
XTR = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'extra/chunk*.parquet'))])       # step 3c
if os.environ.get('USE_CT', '1') == '1':   # step 9 collusion table (AAAI 2013)
    CTB = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'collusion/chunk*.parquet'))]).select(
        'hidx', 'p', 'q', 'ct_mutual', 'ct_asym', 'ct_max', 'ct_pos', 'help_p_by_q', 'help_q_by_p')
    XTR = XTR.join(CTB, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0.0) for c in CTB.columns if c not in ('hidx', 'p', 'q')])
if os.environ.get('USE_VAL', '1') == '1':   # step 3d counterfactual value features
    VAL = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'value/chunk*.parquet'))])
    XTR = XTR.join(VAL, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0) for c in VAL.columns if c not in ('hidx', 'p', 'q')])
REL = REL.join(XTR, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0) for c in XTR.columns if c not in ('hidx', 'p', 'q')])
if os.environ.get('USE_ACT', '1') == '1':   # step 4c action-level aggregates (dev OOF; eval top candidates only; missing -> 0)
    ACT = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(OUT + 'actmodel/*.parquet'))])
    _assert_unique_keys(ACT, 'ACT')
    REL = REL.join(ACT, on=['hidx', 'p', 'q'], how='left').with_columns([pl.col(c).fill_null(0) for c in ACT.columns if c not in ('hidx', 'p', 'q')])
EXCL = {'hidx', 'p', 'q', 'tidx', 'is_eval', 't_order', 'pair_tpos', 'big_blind', 'pair_id', 'evidence_rank', 'y', 'fold', 'key', 'behavior_family', 'pred_fam', 'rule_fam', 's1b'}

def build(df):
    """add within-pair z-scores and rank percentiles; df must contain one phase of one pair set"""
    g = ['p', 'q']
    raw = [c for c in df.columns if c not in EXCL and df.schema[c] in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt32)]
    df = df.with_columns([((pl.col(c) - pl.col(c).mean().over(g)) / (pl.col(c).std().over(g) + 1e-3)).cast(pl.Float32).alias('z_' + c) for c in raw]
                         + [(pl.col(c).rank('average').over(g) / pl.len().over(g)).cast(pl.Float32).alias('pc_' + c) for c in raw])
    return df, raw + ['z_' + c for c in raw] + ['pc_' + c for c in raw]

SC = [pl.scan_parquet(f).select('hidx', 'p', 'q', 'is_eval', 's_ev', 's_dt', 's_sp', 's_ci') for f in sorted(glob.glob(OUT + 'handscores/chunk*.parquet'))]
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
_top = lambda c: pl.col(c).sort(descending=True).head(5).mean()
FAMRULE = (pl.concat([s_.collect() for s_ in SC]).group_by(['p', 'q', 'is_eval']).agg(_top('s_dt').alias('a_dt'), _top('s_sp').alias('a_sp'), _top('s_ci').alias('a_ci')))
FAMRULE = FAMRULE.with_columns(pl.when((pl.col('a_dt') >= pl.col('a_sp')) & (pl.col('a_dt') >= pl.col('a_ci'))).then(pl.lit(FAMS[0]))
                               .when(pl.col('a_sp') >= pl.col('a_ci')).then(pl.lit(FAMS[1])).otherwise(pl.lit(FAMS[2])).alias('rule_fam')).select('p', 'q', 'is_eval', 'rule_fam')
log('family rule built', FAMRULE.height)
D = pl.concat([pl.scan_parquet(f).filter(pl.col('is_eval') == 0).join(pos.lazy().select('p', 'q', 'pair_id'), on=['p', 'q']).collect() for f in PH]).join(REL, on=['hidx', 'p', 'q'], how='left')
D = D.join(pl.concat([s.filter(pl.col('is_eval') == 0).collect() for s in SC]).drop('is_eval'), on=['hidx', 'p', 'q'], how='left')
D = D.join(ev, on=['pair_id', 'hidx'], how='left').with_columns(pl.col('evidence_rank').is_not_null().cast(pl.Int32).alias('y'), (pl.col('tidx') % NF).alias('fold'))
FAMILY_SOURCE = os.environ.get('FAMILY_SOURCE', 'rule')   # 'rule' = top-5 argmax of stage-1 family scores, 'model' = behaviour model
if FAMILY_SOURCE == 'model' and os.path.exists(OUT + 'pair_behavior_oof.parquet'):
    FAMSRC_DEV = pl.read_parquet(OUT + 'pair_behavior_oof.parquet').select('p', 'q', pl.col('pred_fam').alias('rule_fam'))
else:
    FAMSRC_DEV = FAMRULE.filter(pl.col('is_eval') == 0).drop('is_eval')
D = D.join(pos.select('p', 'q', 'behavior_family'), on=['p', 'q']).join(FAMSRC_DEV, on=['p', 'q'], how='left') \
     .with_columns(pl.col('rule_fam').fill_null(FAMS[0]).alias('pred_fam')).drop('rule_fam')
log('family rule agrees with true family on positives:', round(float((D.unique(subset=['p', 'q']).select(pl.col('pred_fam') == pl.col('behavior_family')).to_series().mean())), 4))
D, FEATS = build(D)
D = D.sort(['p', 'q', 't_order'])
log('train rows', D.height, 'feats', len(FEATS))
PARAMS = dict(objective='lambdarank', eval_at=[5], lambdarank_truncation_level=15, learning_rate=0.03, num_leaves=31, min_child_samples=30,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=16, seed=SEED)
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
oof = np.zeros(len(D)); fo = D['fold'].to_numpy(); models = {}
pf = D['pred_fam'].to_numpy()
for f in range(NF):
    for fam in FAMS:
        tr = D.filter((pl.col('fold') != f) & (pl.col('behavior_family') == fam)).sort(['p', 'q', 't_order'])
        grp = tr.group_by(['p', 'q'], maintain_order=True).len()['len'].to_numpy()
        m = lgb.train(PARAMS, lgb.Dataset(tr.select(FEATS).to_numpy().astype(np.float32), tr['y'].to_numpy(), group=grp), 400)
        models[(f, fam)] = m; m.save_model(OUT + f'stage1b_{fam}_f{f}.txt')
        sel = (fo == f) & (pf == fam)
        if sel.sum():
            oof[sel] = m.predict(D.filter(pl.Series(sel)).select(FEATS).to_numpy().astype(np.float32))
    log('fold', f, 'trained 3 family rankers')
cv = {'s_ev_map5': map5(D, 's_ev'), 's1b_map5': map5(D.with_columns(pl.Series('s1b', oof)), 's1b')}
log('CV', cv); json.dump(cv, open(OUT + 'stage1b_cv.json', 'w'), indent=1)

os.makedirs(OUT + 'stage1b', exist_ok=True)
D.select('hidx', 'p', 'q').with_columns(pl.Series('s1b', oof.astype(np.float32))).write_parquet(OUT + 'stage1b_devpos_oof.parquet')
SCORE_ALL = pl.concat([s.collect() for s in SC])
PHASES = [(0, 'dev')] if os.environ.get('DEV_ONLY') == '1' else [(1, 'eval'), (0, 'dev')]
for phase, tag in PHASES:
    for i, f in enumerate(PH):
        E = pl.read_parquet(f).filter(pl.col('is_eval') == phase).join(REL, on=['hidx', 'p', 'q'], how='left')
        if phase == 1: E = E.join(evp.select('p', 'q'), on=['p', 'q'])
        E = E.join(SCORE_ALL.filter(pl.col('is_eval') == phase).drop('is_eval'), on=['hidx', 'p', 'q'], how='left')
        FSRC = FAMRULE.filter(pl.col('is_eval') == phase).drop('is_eval')
        if FAMILY_SOURCE == 'model' and phase == 1 and os.path.exists(os.environ.get('EVAL_BEHAVIOR', '')):
            FSRC = (pl.read_csv(os.environ['EVAL_BEHAVIOR']).select('pair_id', pl.col('predicted_behavior').alias('rule_fam'))
                    .join(evp.select('pair_id', 'p', 'q'), on='pair_id').select('p', 'q', 'rule_fam'))
        elif FAMILY_SOURCE == 'model' and phase == 0 and os.path.exists(OUT + 'pair_behavior_oof.parquet'):
            FSRC = pl.read_parquet(OUT + 'pair_behavior_oof.parquet').select('p', 'q', pl.col('pred_fam').alias('rule_fam'))
        E = E.join(FSRC, on=['p', 'q'], how='left').with_columns(pl.col('rule_fam').fill_null(FAMS[0]))
        E, _ = build(E)
        X = E.select(FEATS).to_numpy().astype(np.float32)
        pfe = E['rule_fam'].to_numpy(); sc = np.zeros(len(E))
        # development rows must be scored out-of-fold (the model that never saw that pool); evaluation has no labels, use the 5-model mean
        foldv = (E['tidx'].to_numpy() % NF) if phase == 0 else None
        for fam in FAMS:
            sel = pfe == fam
            if not sel.sum(): continue
            if phase == 0:
                for f2 in range(NF):
                    s2 = sel & (foldv == f2)
                    if s2.sum(): sc[s2] = models[(f2, fam)].predict(X[s2])
            else:
                sc[sel] = np.mean([models[(f2, fam)].predict(X[sel]) for f2 in range(NF)], axis=0)
        E.select('hidx', 'p', 'q').with_columns(pl.Series('s1b', sc.astype(np.float32))).write_parquet(OUT + f'stage1b/{tag}{i:02d}.parquet')
        log('scored', tag, 'chunk', i, E.height)
log('done')
