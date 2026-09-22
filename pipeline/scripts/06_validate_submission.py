"""Validate submission.csv against competition rules using raw files only."""
import polars as pl
R = 'data/raw/'
sub = pl.read_csv(__import__('sys').argv[1] if len(__import__('sys').argv)>1 else 'submission.csv'); tmpl = pl.read_csv(R + 'sample_submission.csv'); evp = pl.read_csv(R + 'evaluation_pairs.csv')
EV = [f'evidence_hand_{i}' for i in range(1, 6)]
assert sub.columns == tmpl.columns, sub.columns
assert sub.height == tmpl.height and set(sub['pair_id']) == set(tmpl['pair_id'])
assert sub.null_count().sum_horizontal()[0] == 0
assert sub['risk_score'].is_between(0, 1).all()
assert set(sub['predicted_behavior'].unique()) <= {'none', 'directed_transfer', 'soft_play', 'coordinated_isolation', 'other_coordination'}
long = sub.select('pair_id', *EV).unpivot(index='pair_id', value_name='hand_id').filter(pl.col('hand_id') != 'NO_EVIDENCE')
assert long.height == 0 or long.group_by(['pair_id', 'hand_id']).len()['len'].max() == 1, 'duplicate hand in a row'
seats = pl.read_parquet(R + 'seats.parquet', columns=['hand_id', 'player_id']); hands = pl.read_parquet(R + 'hands.parquet', columns=['hand_id', 'phase'])
chk = long.join(evp, on='pair_id').join(hands, on='hand_id', how='left')
chk = chk.join(seats.rename({'player_id': 'player_1'}).with_columns(pl.lit(1).alias('s1')), on=['hand_id', 'player_1'], how='left') \
         .join(seats.rename({'player_id': 'player_2'}).with_columns(pl.lit(1).alias('s2')), on=['hand_id', 'player_2'], how='left')
print('evidence entries:', chk.height, '| eval phase:', (chk['phase'] == 'evaluation').mean(), '| both players seated:', (chk['s1'].is_not_null() & chk['s2'].is_not_null()).mean())
print('unique risk scores:', sub['risk_score'].n_unique(), '/', sub.height)
print('OK')
