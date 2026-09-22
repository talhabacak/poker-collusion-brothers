"""Step 12: blend two teams' submissions (ours = A, teammate's = B) into one.
Risk       : weighted rank average of the two risk columns.
Evidence   : reciprocal-rank fusion of the two top-5 lists (hands on both lists first, then by fused rank).
Behaviour  : A's label unless B agrees on a different family with A predicting 'other_coordination' off, etc.
             (default: keep A; BEH_MODE=agree keeps A where they agree and A otherwise -- i.e. A -- so the
             only alternative is BEH_MODE=b which takes B's label; both exposed for measurement)
Usage: W_B=0.5 EV_MODE=rrf BEH_MODE=a python src/12_team_blend.py A.csv B.csv out.csv
"""
import polars as pl, sys, os
A, B, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W_B = float(os.environ.get('W_B', '0.5'))          # weight of B's risk rank
EV_MODE = os.environ.get('EV_MODE', 'rrf')         # rrf | a | b
BEH_MODE = os.environ.get('BEH_MODE', 'a')         # a | b
EV = [f'evidence_hand_{i}' for i in range(1, 6)]
a = pl.read_csv(A); b = pl.read_csv(B)
assert a.height == b.height and set(a['pair_id']) == set(b['pair_id']), 'pair sets differ'
b = b.rename({c: c + '_b' for c in b.columns if c != 'pair_id'})
d = a.join(b, on='pair_id')
n = d.height
d = d.with_columns(((1 - W_B) * pl.col('risk_score').rank() / n + W_B * pl.col('risk_score_b').rank() / n).alias('risk_new'))
# evidence fusion
def fuse(row):
    la = [row[c] for c in EV]; lb = [row[c + '_b'] for c in EV]
    if EV_MODE == 'a': return la
    if EV_MODE == 'b': return lb
    score = {}
    for i, h in enumerate(la): score[h] = score.get(h, 0) + 1.0 / (i + 1)
    for i, h in enumerate(lb): score[h] = score.get(h, 0) + 1.0 / (i + 1)
    # hands on both lists get both contributions and naturally rise; ties broken by A's order
    order = sorted(score, key=lambda h: (-score[h], la.index(h) if h in la else 99))
    return order[:5]
fused = [fuse(r) for r in d.select(EV + [c + '_b' for c in EV]).iter_rows(named=True)]
for i, c in enumerate(EV): d = d.with_columns(pl.Series(c, [f[i] for f in fused]))
if BEH_MODE == 'b': d = d.with_columns(pl.col('predicted_behavior_b').alias('predicted_behavior'))
d = d.with_columns(pl.col('risk_new').alias('risk_score'))
ov = sum(len(set([r[c] for c in EV]) & set([r[c + '_b'] for c in EV])) for r in d.select(EV + [c + '_b' for c in EV]).iter_rows(named=True)) / (5 * n)
top = d.select((pl.col('risk_score').rank(descending=True) <= 1500).alias('x'), (pl.col('risk_score_b').rank(descending=True) <= 1500).alias('y'))
print(f'evidence overlap between A and B (all pairs): {ov:.3f} | top-1500 overlap A vs B: {int((top["x"] & top["y"]).sum())}/1500 | behaviour agreement: {(d["predicted_behavior"] == d["predicted_behavior_b"]).mean():.3f}')
d.select(a.columns).write_csv(OUT); print('written', OUT)
