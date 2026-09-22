"""Step 9b: relational normalisation of the collusion table.
Raw C(j,k) says how much k's actions helped j. But some players simply receive more help
(loose tables, passive opponents). What matters is whether the *partner* helps j more than
j's other opponents do. This computes that margin per pair, per phase.
Output: data/interim/ct_relational.parquet with one row per (p, q, is_eval).
"""
import polars as pl, numpy as np, glob, time
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
OUT = 'data/interim/'
hands = pl.read_parquet(OUT + 'hands.parquet').select('hidx', 'is_eval')

D = pl.concat([pl.read_parquet(f).select('hidx', 'p', 'q', 'help_p_by_q', 'help_q_by_p', 'helppos_p_by_q', 'helppos_q_by_p')
               for f in sorted(glob.glob(OUT + 'collusion/chunk*.parquet'))]).join(hands, on='hidx')
log('ct rows', D.shape)

# directional long form: j received `help` from k in this hand
L = pl.concat([
    D.select(pl.col('p').alias('j'), pl.col('q').alias('k'), 'is_eval',
             pl.col('help_p_by_q').alias('h'), pl.col('helppos_p_by_q').alias('hp')),
    D.select(pl.col('q').alias('j'), pl.col('p').alias('k'), 'is_eval',
             pl.col('help_q_by_p').alias('h'), pl.col('helppos_q_by_p').alias('hp')),
])
del D
# per opponent: how much help j gets from k on average, and in k's best hands
_t5 = lambda c: pl.col(c).sort(descending=True).head(5).mean()
A = L.group_by(['j', 'k', 'is_eval']).agg(pl.len().alias('n_jk'), pl.col('h').mean().alias('h_mean'),
                                          _t5('h').alias('h_top5'), pl.col('hp').mean().alias('hp_mean'))
del L
log('opponent aggregates', A.shape)

# relational margin: k against j's other opponents in the same phase
g = ['j', 'is_eval']
# no minimum-hands filter: any threshold would leave the dropped pairs at a constant fill value
# and recreate the "scored vs unscored" artefact (rule 6 in EXPERIMENTS.md)
for c in ['h_mean', 'h_top5', 'hp_mean']:
    A = A.with_columns(((pl.col(c) - pl.col(c).mean().over(g)) / (pl.col(c).std().over(g) + 1e-6)).alias(c + '_z'),
                       pl.col(c).rank('ordinal', descending=True).over(g).alias(c + '_rk'))
log('relational z-scores done')

# back to unordered pairs: p's view of q and q's view of p
COLS = [c + s for c in ['h_mean', 'h_top5', 'hp_mean'] for s in ['_z', '_rk']]
P = A.select(pl.col('j').alias('p'), pl.col('k').alias('q'), 'is_eval', *[pl.col(c).alias('a_' + c) for c in COLS])
Q = A.select(pl.col('k').alias('p'), pl.col('j').alias('q'), 'is_eval', *[pl.col(c).alias('b_' + c) for c in COLS])
F = P.join(Q, on=['p', 'q', 'is_eval'], how='full', coalesce=True).filter(pl.col('p') < pl.col('q'))
F = F.with_columns([pl.col(c).fill_null(0.0) for c in F.columns if c.startswith(('a_', 'b_'))])
# symmetric summaries: the pair is suspicious if help stands out in either direction (max) or both (min)
for c in COLS:
    F = F.with_columns(pl.max_horizontal('a_' + c, 'b_' + c).alias('ctr_max_' + c),
                       pl.min_horizontal('a_' + c, 'b_' + c).alias('ctr_min_' + c))
F = F.select('p', 'q', 'is_eval', *[c for c in F.columns if c.startswith('ctr_')])
F.write_parquet(OUT + 'ct_relational.parquet')
log('written', F.shape, F.columns)
