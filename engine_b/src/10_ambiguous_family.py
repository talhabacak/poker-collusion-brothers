"""Step 10: fourth-family relabelling by family AMBIGUITY (replaces the novelty rule of step 8, which
the 19 Sep sweep showed contributes exactly zero).

Reasoning: the behaviour head is a 3-class model. A pair from the undisclosed fourth family cannot be
fitted by any of the three, so the head should be *unsure* about it. The step-8 rule instead looked for
high risk + LOW evidence score, and relabelled 432 pairs with no effect on the leaderboard at all -- i.e.
it was selecting pairs that are not true positives.

The asymmetry that makes this worth a submission: where the head is already unsure, its current label is
probably wrong anyway, so relabelling costs little; if those pairs really are the fourth family, it pays.
"""
import polars as pl, sys, os
OUT = 'data/interim/'; RAW = 'data/raw/'
IN_SUB, OUT_SUB = sys.argv[1], sys.argv[2]
THR = float(os.environ.get('AMB_THR', '0.60'))    # relabel when the top family probability is below this
TOPN = int(os.environ.get('AMB_TOPN', '1500'))    # only inside this risk band (positives live in the top ~1500)
E = pl.read_parquet(OUT + 'eval_behavior_probs.parquet')
E = E.with_columns(pl.max_horizontal('p_dt', 'p_sp', 'p_ci').alias('conf'),
                   pl.col('risk').rank(descending=True).alias('rr'))
E = E.with_columns(((pl.col('rr') <= TOPN) & (pl.col('conf') < THR)).alias('amb'))
n = int(E['amb'].sum())
print(f'top-{TOPN} icinde aile kesinligi < {THR}: {n} cift yeniden etiketleniyor')
print(f'  bu cesitlerin risk araligi: {E.filter("amb")["risk"].min():.3f} - {E.filter("amb")["risk"].max():.3f}')
sub = pl.read_csv(IN_SUB)
out = sub.join(E.select('pair_id', 'amb'), on='pair_id', how='left').with_columns(pl.col('amb').fill_null(False))
out = out.with_columns(pl.when('amb').then(pl.lit('other_coordination')).otherwise(pl.col('predicted_behavior')).alias('predicted_behavior')).drop('amb')
out.select(sub.columns).write_csv(OUT_SUB)
print('written', OUT_SUB)
