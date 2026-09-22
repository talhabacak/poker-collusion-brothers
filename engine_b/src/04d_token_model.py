"""Step 4d: action-sequence token model (isolation-focused). Each action in a pair-hand becomes a token
(street, role A/B/X, action, size bucket); unigram+bigram+trigram counts are hashed into a sparse matrix.
Two hand-level classifiers: tok_ev (evidence vs other hands, all families) and tok_iso (isolation evidence vs
everything else). Dev rows OOF by table; eval rows = each pair's top-NCAND hands by stage-1."""
import polars as pl, numpy as np, lightgbm as lgb, glob, os, time, json
import scipy.sparse as sp
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'; RAW = 'data/raw/'; NF = 5; SEED = 42; NB = 2 ** 14; NCAND = int(os.environ.get('TOK_NCAND', 30))
TD = OUT + 'tokmodel/'; os.makedirs(TD, exist_ok=True)
players = pl.read_parquet(OUT + 'players.parquet').select('player_id', 'pidx')
def to_pq(df):
    df = df.join(players.rename({'player_id': 'player_1', 'pidx': 'i1'}), on='player_1').join(players.rename({'player_id': 'player_2', 'pidx': 'i2'}), on='player_2')
    return df.with_columns(pl.min_horizontal('i1', 'i2').alias('p'), pl.max_horizontal('i1', 'i2').alias('q')).drop('i1', 'i2')
lab = to_pq(pl.read_csv(RAW + 'development_labels.csv')); evp = to_pq(pl.read_csv(RAW + 'evaluation_pairs.csv'))
hands = pl.read_parquet(OUT + 'hands.parquet').select('hidx', 'hand_id')
ev = pl.read_csv(RAW + 'development_evidence.csv').join(hands, on='hand_id').select('pair_id', 'hidx')
A = pl.read_parquet(OUT + 'actions_scored.parquet').select('hidx', 'action_no', 'pidx', 'st', 'y', 'amount', 'pot_before', 'players_active')
A = A.with_columns(pl.when(pl.col('amount') == 0).then(0).when(pl.col('amount') < 0.5 * pl.col('pot_before')).then(1).when(pl.col('amount') < 1.2 * pl.col('pot_before')).then(2).otherwise(3).alias('sz'),
                   pl.col('players_active').clip(2, 4).alias('pa'))
HSC = pl.concat([pl.scan_parquet(f).select('hidx', 'p', 'q', 'is_eval', 's_ev').collect() for f in sorted(glob.glob(OUT + 'handscores/chunk*.parquet'))])
HSC = HSC.with_columns(pl.col('s_ev').rank('ordinal', descending=True).over(['p', 'q', 'is_eval']).alias('r1'))
CAND_DEV = HSC.filter(pl.col('is_eval') == 0).select('hidx', 'p', 'q').join(lab.select('p', 'q'), on=['p', 'q'])
CAND_EVAL = HSC.filter((pl.col('is_eval') == 1) & (pl.col('r1') <= NCAND)).select('hidx', 'p', 'q').join(evp.select('p', 'q'), on=['p', 'q'])
log('candidates dev', CAND_DEV.height, 'eval', CAND_EVAL.height)

def tokens(cand):
    X = cand.join(A, on='hidx').with_columns(
        pl.when(pl.col('pidx') == pl.col('p')).then(0).when(pl.col('pidx') == pl.col('q')).then(1).otherwise(2).alias('role'))
    X = X.with_columns((((pl.col('st') * 3 + pl.col('role')) * 6 + pl.col('y')) * 4 + pl.col('sz')).cast(pl.Int64).alias('t1')).sort(['hidx', 'p', 'q', 'action_no'])
    g = ['hidx', 'p', 'q']
    X = X.with_columns(pl.col('t1').shift(1).over(g).alias('t0'), pl.col('t1').shift(2).over(g).alias('tm'))
    X = X.with_columns(
        (pl.col('t1') % NB).alias('u'),
        ((pl.col('t0') * 1000 + pl.col('t1')) * 31 % NB).alias('b'),
        ((pl.col('tm') * 1000003 + pl.col('t0') * 1000 + pl.col('t1')) * 17 % NB).alias('tr'),
        (((pl.col('role') * 3 + pl.col('st')) * 6 + pl.col('y')) * 4 + pl.col('pa') + 5000).alias('ctxtok'))
    keys = X.select(g).unique(maintain_order=True).with_row_index('row')
    X = X.join(keys, on=g)
    cols = [X.select('row', pl.col('u').alias('c')), X.select('row', pl.col('b').alias('c')).filter(pl.col('c').is_not_null()),
            X.select('row', pl.col('tr').alias('c')).filter(pl.col('c').is_not_null()), X.select('row', (pl.col('ctxtok') % NB).alias('c'))]
    E = pl.concat(cols).group_by(['row', 'c']).len()
    M = sp.csr_matrix((E['len'].to_numpy().astype(np.float32), (E['row'].to_numpy(), E['c'].to_numpy())), shape=(keys.height, NB))
    return keys, M

keys, M = tokens(CAND_DEV)
D = keys.join(lab.select('p', 'q', 'pair_id', 'label', 'behavior_family'), on=['p', 'q']).join(ev.with_columns(pl.lit(1).alias('y_ev')), on=['pair_id', 'hidx'], how='left').with_columns(pl.col('y_ev').fill_null(0))
tid = pl.read_parquet(OUT + 'hands.parquet').select('hidx', 'tidx')
D = D.join(tid, on='hidx').with_columns((pl.col('tidx') % NF).alias('fold')).sort('row')
y_ev = D['y_ev'].to_numpy(); y_iso = ((D['y_ev'] == 1) & (D['behavior_family'] == 'coordinated_isolation')).to_numpy().astype(int); fo = D['fold'].to_numpy()
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_child_samples=30, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
           scale_pos_weight=5.0, verbose=-1, num_threads=16, seed=SEED)
oof_ev = np.zeros(len(D)); oof_iso = np.zeros(len(D)); models = []
for f in range(NF):
    tr = fo != f
    m1 = lgb.train(PAR, lgb.Dataset(M[tr], y_ev[tr]), 400); m2 = lgb.train(PAR, lgb.Dataset(M[tr], y_iso[tr]), 400)
    oof_ev[fo == f] = m1.predict(M[fo == f]); oof_iso[fo == f] = m2.predict(M[fo == f]); models.append((m1, m2))
    log('fold', f)
from sklearn.metrics import roc_auc_score
pos = D['label'].to_numpy() == 1
cv = {'auc_tok_ev_in_pos_pairs': roc_auc_score(y_ev[pos], oof_ev[pos]), 'auc_tok_iso_in_pos_pairs': roc_auc_score(y_iso[pos], oof_iso[pos])}
truth = ev.group_by('pair_id').agg(pl.col('hidx').alias('truth'))
Dp = D.filter(pl.col('label') == 1).with_columns(pl.Series('tok_ev', oof_ev[pos]), pl.Series('tok_iso', oof_iso[pos]))
def map5(df, col):
    r = df.sort(['pair_id', col], descending=[False, True]).group_by('pair_id', maintain_order=True).agg(pl.col('hidx').head(5).alias('pred')).join(truth, on='pair_id')
    out = []
    for pred, tru in zip(r['pred'].to_list(), r['truth'].to_list()):
        t = set(tru); h = 0; s = 0.0
        for i, x in enumerate(pred):
            if x in t: h += 1; s += h / (i + 1)
        out.append(s / min(len(t), 5))
    return float(np.mean(out))
cv['map5_tok_ev'] = map5(Dp, 'tok_ev'); cv['map5_tok_ev_iso_pairs'] = map5(Dp.filter(pl.col('behavior_family') == 'coordinated_isolation'), 'tok_iso')
log('CV', cv); json.dump(cv, open(OUT + 'tokmodel_cv.json', 'w'), indent=1)
D.select('hidx', 'p', 'q').with_columns(pl.Series('tok_ev', oof_ev.astype(np.float32)), pl.Series('tok_iso', oof_iso.astype(np.float32))).write_parquet(TD + 'dev.parquet')
keys_e, Me = tokens(CAND_EVAL)
pe = np.mean([m1.predict(Me) for m1, _ in models], axis=0); pi = np.mean([m2.predict(Me) for _, m2 in models], axis=0)
keys_e.sort('row').select('hidx', 'p', 'q').with_columns(pl.Series('tok_ev', pe.astype(np.float32)), pl.Series('tok_iso', pi.astype(np.float32))).write_parquet(TD + 'eval.parquet')
log('done', keys_e.height)
