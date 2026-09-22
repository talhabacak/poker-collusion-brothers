"""Step 11 (high-variance): blend the learned pair risk with the strongest unsupervised pair signal.

The risk model is trained on 372 positive pairs. The evaluation set has ~1000 positives we never see, and
Pair AP is 70% of the metric -- so an overfit to those 372 costs more than anywhere else. The relational
G rate (how much each member's behaviour towards the partner departs from their behaviour towards everyone
else) separates labelled pairs at AUC 0.987 with no training at all, so it cannot overfit.

There is no offline metric that can judge this (rule 9): the PU metric treats the hidden positives as
negatives, which is exactly the failure mode we are trying to fix. It is a leaderboard-only question.
"""
import polars as pl, sys, os
OUT = 'data/interim/'; RAW = 'data/raw/'
IN_SUB, OUT_SUB = sys.argv[1], sys.argv[2]
W = float(os.environ.get('BLEND_W', '0.20'))      # weight on the unsupervised signal
players = pl.read_parquet(OUT + 'players.parquet').select('player_id', 'pidx')
evp = pl.read_csv(RAW + 'evaluation_pairs.csv')
evp = evp.join(players.rename({'player_id': 'player_1', 'pidx': 'i1'}), on='player_1').join(
             players.rename({'player_id': 'player_2', 'pidx': 'i2'}), on='player_2')
evp = evp.with_columns(pl.min_horizontal('i1', 'i2').alias('p'), pl.max_horizontal('i1', 'i2').alias('q'))
G = pl.read_parquet(OUT + 'pair_relational.parquet').filter(pl.col('is_eval') == 1).select('p', 'q', 'G_rate', 'G_max')
sub = pl.read_csv(IN_SUB)
D = sub.join(evp.select('pair_id', 'p', 'q'), on='pair_id').join(G, on=['p', 'q'], how='left')
D = D.with_columns([pl.col(c).fill_null(pl.col(c).min()) for c in ['G_rate', 'G_max']])
n = D.height
D = D.with_columns((pl.col('risk_score').rank() / n).alias('r_model'),
                   ((pl.col('G_rate').rank() + pl.col('G_max').rank()) / (2 * n)).alias('r_unsup'))
D = D.with_columns(((1 - W) * pl.col('r_model') + W * pl.col('r_unsup')).alias('new_risk'))
ch = D.select((pl.col('new_risk').rank(descending=True) <= 1500).alias('a'),
              (pl.col('risk_score').rank(descending=True) <= 1500).alias('b'))
print(f'blend w={W}: ilk 1500 cift icinde degisim {int((ch["a"] != ch["b"]).sum())} cift')
out = sub.join(D.select('pair_id', 'new_risk'), on='pair_id', how='left')
out = out.with_columns(pl.col('new_risk').fill_null(pl.col('risk_score')).alias('risk_score')).drop('new_risk')
out.select(sub.columns).write_csv(OUT_SUB)
print('written', OUT_SUB)
