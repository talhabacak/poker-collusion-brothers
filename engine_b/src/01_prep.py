"""Step 1: encode ids, hand strength (preflop equity, postflop made-hand strength), action context features, player style stats.

WINDOWS PATCH (stage 440): the only change from origin/tarik:src/01_prep.py is structural --
the two multiprocessing worker functions and the card tables stay at module level and the
rest of the script moves into main() under an `if __name__ == "__main__"` guard, because
Windows spawns (rather than forks) pool workers and would otherwise re-execute the whole
script in every child. No feature, seed, filter or numeric constant is touched.
"""
import polars as pl, numpy as np, time, os
from multiprocessing import Pool
from treys import Card, Evaluator

RAW = 'data/raw/'; OUT = 'data/interim/'
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

RANKS = '23456789TJQKA'; SUITS = 'shdc'
CARDS = [r + s for r in RANKS for s in SUITS]
CIDX = {c: i for i, c in enumerate(CARDS)}
TREYS = np.array([Card.new(c) for c in CARDS], dtype=np.int64)


# ---------------- preflop equity vs one random hand (169 classes, MC) ----------------
def mc_equity(args):
    hi, lo, suited, n, seed = args
    rng = np.random.default_rng(seed); ev = Evaluator()
    c1 = CIDX[RANKS[hi] + 's']; c2 = CIDX[RANKS[lo] + ('s' if suited else 'h')]
    deck = np.array([c for c in range(52) if c not in (c1, c2)])
    win = 0.0
    for _ in range(n):
        d = rng.choice(deck, 7, replace=False)
        b = [int(TREYS[x]) for x in d[:5]]
        a = ev.evaluate(b, [int(TREYS[c1]), int(TREYS[c2])]); o = ev.evaluate(b, [int(TREYS[d[5]]), int(TREYS[d[6]])])
        win += 1.0 if a < o else (0.5 if a == o else 0.0)
    return hi, lo, suited, win / n


# postflop made-hand strength for every seat on every reached street
def eval_chunk(args):
    hole, board = args
    ev = Evaluator(); out = np.empty((len(hole), 2), dtype=np.float32)
    for i in range(len(hole)):
        b = [int(TREYS[x]) for x in board[i] if x >= 0]
        r = ev.evaluate(b, [int(TREYS[hole[i, 0]]), int(TREYS[hole[i, 1]])])
        out[i, 0] = 1.0 - (r - 1) / 7461.0; out[i, 1] = ev.get_rank_class(r)
    return out


def main():
    # ---------------- hands / players ----------------
    hands = pl.read_parquet(RAW + 'hands.parquet')
    tables = hands.select('table_id').unique().sort('table_id').with_row_index('tidx')
    hands = (hands.join(tables, on='table_id').sort(['tidx', 'started_at', 'hand_id']).with_row_index('hidx')
             .with_columns(pl.col('hidx').cast(pl.Int32), pl.col('tidx').cast(pl.Int16),
                           (pl.col('phase') == 'evaluation').cast(pl.Int8).alias('is_eval'),
                           pl.col('board_cards').fill_null('').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).alias('board_list')))
    hands = hands.with_columns(pl.col('board_list').list.len().cast(pl.Int8).alias('board_len'),
                               pl.col('hidx').rank('ordinal').over('tidx').cast(pl.Int32).alias('t_order'))
    players = pl.read_parquet(RAW + 'players.parquet').sort('player_id').with_row_index('pidx').with_columns(pl.col('pidx').cast(pl.Int32))
    players.write_parquet(OUT + 'players.parquet')
    log('hands', hands.shape)

    jobs = []
    for hi in range(13):
        for lo in range(hi + 1):
            for suited in ([0] if hi == lo else [0, 1]):
                jobs.append((hi, lo, suited, 4000, hi * 100 + lo * 2 + suited))
    with Pool(16) as p:
        eq = p.map(mc_equity, jobs)
    eq_pf = pl.DataFrame(eq, schema=['hi', 'lo', 'suited', 'eq_pf'], orient='row').with_columns(pl.col('hi', 'lo', 'suited').cast(pl.Int8), pl.col('eq_pf').cast(pl.Float32))
    log('preflop equity table', eq_pf.shape)

    # ---------------- seats ----------------
    seats = (pl.read_parquet(RAW + 'seats.parquet')
             .join(hands.select('hand_id', 'hidx', 'tidx', 'is_eval', 'button_seat', 'big_blind', 'board_list'), on='hand_id')
             .join(players.select('player_id', 'pidx'), on='player_id')
             .with_columns(pl.col('hole_card_1').replace_strict(CIDX, return_dtype=pl.Int8).alias('c1'),
                           pl.col('hole_card_2').replace_strict(CIDX, return_dtype=pl.Int8).alias('c2')))
    seats = seats.with_columns((pl.col('c1') // 4).alias('r1'), (pl.col('c2') // 4).alias('r2'))
    seats = seats.with_columns(pl.max_horizontal('r1', 'r2').alias('hi'), pl.min_horizontal('r1', 'r2').alias('lo'),
                               ((pl.col('c1') % 4 == pl.col('c2') % 4) & (pl.col('r1') != pl.col('r2'))).cast(pl.Int8).alias('suited'),
                               ((pl.col('seat_no') - pl.col('button_seat')) % 6).cast(pl.Int8).alias('relpos'))
    seats = seats.join(eq_pf, on=['hi', 'lo', 'suited'], how='left')

    bl = seats.select('board_list').to_series().to_list()
    hole = seats.select('c1', 'c2').to_numpy().astype(np.int8)
    board = np.full((len(bl), 5), -1, dtype=np.int8)
    for i, b in enumerate(bl):
        for j, c in enumerate(b): board[i, j] = CIDX[c]
    blen = (board >= 0).sum(1)
    for s, ncard in [(1, 3), (2, 4), (3, 5)]:
        idx = np.where(blen >= ncard)[0]
        bb = board[idx].copy(); bb[:, ncard:] = -1
        chunks = np.array_split(np.arange(len(idx)), 64)
        with Pool(16) as p:
            res = p.map(eval_chunk, [(hole[idx[c]], bb[c]) for c in chunks])
        res = np.vstack(res)
        hs = np.full(len(bl), np.nan, dtype=np.float32); hc = np.full(len(bl), -1, dtype=np.int8)
        hs[idx] = res[:, 0]; hc[idx] = res[:, 1]
        seats = seats.with_columns(pl.Series(f'hs{s}', hs).fill_nan(None), pl.Series(f'hc{s}', hc))
        log('street', s, 'evaluated', len(idx))
    del bl, board, hole

    # ---------------- actions ----------------
    smap = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    amap = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4, 'all_in': 5}
    A = (pl.read_parquet(RAW + 'actions.parquet')
         .join(hands.select('hand_id', 'hidx', 'tidx', 'is_eval', 'big_blind', 'players_dealt'), on='hand_id')
         .join(players.select('player_id', 'pidx'), on='player_id')
         .with_columns(pl.col('street').replace_strict(smap, return_dtype=pl.Int8).alias('st'),
                       pl.col('action').replace_strict(amap, return_dtype=pl.Int8).alias('y'))
         .drop('hand_id', 'player_id', 'street', 'action')
         .sort(['hidx', 'action_no']))
    A = A.with_columns(pl.col('y').is_in([3, 4, 5]).cast(pl.Int16).alias('agg'))
    A = A.with_columns(
        (pl.col('agg').cum_sum().over('hidx') - pl.col('agg')).alias('n_agg_hand'),
        (pl.col('agg').cum_sum().over(['hidx', 'st']) - pl.col('agg')).alias('n_agg_st'),
        (pl.col('agg').cum_sum().over(['hidx', 'pidx']) - pl.col('agg')).alias('own_agg_before'),
        pl.int_range(pl.len()).over(['hidx', 'st']).cast(pl.Int16).alias('n_act_st'),
        pl.when(pl.col('agg') == 1).then(pl.col('pidx')).otherwise(None).alias('_la'))
    A = A.with_columns(pl.col('_la').forward_fill().shift(1).over('hidx').alias('last_agg_pidx')).drop('_la')
    A = A.with_columns((pl.col('last_agg_pidx') == pl.col('pidx')).fill_null(False).cast(pl.Int8).alias('self_last_agg'),
                       (pl.col('to_call') / pl.col('big_blind')).cast(pl.Float32).alias('tocall_bb'),
                       (pl.col('pot_before') / pl.col('big_blind')).cast(pl.Float32).alias('pot_bb'),
                       (pl.col('stack_before') / pl.col('big_blind')).cast(pl.Float32).alias('stack_bb'),
                       (pl.col('to_call') / (pl.col('pot_before') + pl.col('to_call')).clip(1)).cast(pl.Float32).alias('potodds'),
                       (pl.col('to_call') / (pl.col('stack_before').clip(1))).cast(pl.Float32).alias('tocall_stack'),
                       (pl.col('stack_before') / pl.col('pot_before').clip(1)).cast(pl.Float32).alias('spr'))
    A = A.join(seats.select('hidx', 'pidx', 'relpos', 'hi', 'lo', 'suited', 'eq_pf', 'hs1', 'hs2', 'hs3', 'hc1', 'hc2', 'hc3', 'starting_stack'), on=['hidx', 'pidx'])
    A = A.with_columns(pl.when(pl.col('st') == 1).then(pl.col('hs1')).when(pl.col('st') == 2).then(pl.col('hs2')).when(pl.col('st') == 3).then(pl.col('hs3')).otherwise(None).alias('hs'),
                       pl.when(pl.col('st') == 1).then(pl.col('hc1')).when(pl.col('st') == 2).then(pl.col('hc2')).when(pl.col('st') == 3).then(pl.col('hc3')).otherwise(-1).alias('hc'),
                       (pl.col('hi') == pl.col('lo')).cast(pl.Int8).alias('pocket'),
                       (pl.col('hi') - pl.col('lo')).alias('gap')).drop('hs1', 'hs2', 'hs3', 'hc1', 'hc2', 'hc3')
    log('actions', A.shape)

    # player style stats per (player, phase)
    ps = A.group_by(['pidx', 'is_eval']).agg(
        pl.len().alias('ps_n'),
        ((pl.col('st') == 0) & (pl.col('n_agg_st') == 0)).sum().alias('_o'),
        (((pl.col('st') == 0) & (pl.col('n_agg_st') == 0)) & (pl.col('agg') == 1)).sum().alias('_or'),
        (((pl.col('st') == 0) & (pl.col('n_agg_st') == 0)) & (pl.col('y') == 0)).sum().alias('_of'),
        ((pl.col('st') == 0) & (pl.col('n_agg_st') > 0)).sum().alias('_v'),
        (((pl.col('st') == 0) & (pl.col('n_agg_st') > 0)) & (pl.col('agg') == 1)).sum().alias('_vr'),
        (((pl.col('st') == 0) & (pl.col('n_agg_st') > 0)) & (pl.col('y') == 0)).sum().alias('_vf'),
        (pl.col('st') > 0).sum().alias('_p'),
        ((pl.col('st') > 0) & (pl.col('agg') == 1)).sum().alias('_pa'),
        ((pl.col('st') > 0) & (pl.col('to_call') > 0)).sum().alias('_pb'),
        (((pl.col('st') > 0) & (pl.col('to_call') > 0)) & (pl.col('y') == 0)).sum().alias('_pbf'))
    ps = ps.with_columns(((pl.col('_or') + 1) / (pl.col('_o') + 5)).alias('ps_open_raise'), ((pl.col('_of') + 3) / (pl.col('_o') + 5)).alias('ps_open_fold'),
                         ((pl.col('_vr') + 1) / (pl.col('_v') + 10)).alias('ps_vs_raise_agg'), ((pl.col('_vf') + 6) / (pl.col('_v') + 10)).alias('ps_vs_raise_fold'),
                         ((pl.col('_pa') + 3) / (pl.col('_p') + 10)).alias('ps_post_agg'), ((pl.col('_pbf') + 4) / (pl.col('_pb') + 10)).alias('ps_post_fold_vs_bet'))
    ps = ps.select('pidx', 'is_eval', 'ps_n', *[pl.col(c).cast(pl.Float32) for c in ['ps_open_raise', 'ps_open_fold', 'ps_vs_raise_agg', 'ps_vs_raise_fold', 'ps_post_agg', 'ps_post_fold_vs_bet']])
    A = A.join(ps, on=['pidx', 'is_eval'], how='left')

    # fold action_no per seat
    folds = A.filter(pl.col('y') == 0).select('hidx', 'pidx', pl.col('action_no').cast(pl.Int16).alias('fold_no'))
    seats = seats.join(folds, on=['hidx', 'pidx'], how='left')

    A.write_parquet(OUT + 'actions_feat.parquet')
    seats.select('hidx', 'pidx', 'tidx', 'is_eval', 'seat_no', 'relpos', 'starting_stack', 'total_contribution', 'net_chips', 'folded', 'went_to_showdown', 'won_share',
                 'eq_pf', 'hs1', 'hs2', 'hs3', 'fold_no', 'big_blind').write_parquet(OUT + 'seats_feat.parquet')
    hands.select('hidx', 'hand_id', 'tidx', 'table_id', 'is_eval', 't_order', 'started_at', 'big_blind', 'final_pot', 'board_len', 'players_dealt', 'players_at_showdown').write_parquet(OUT + 'hands.parquet')
    log('done')


if __name__ == '__main__':
    main()
