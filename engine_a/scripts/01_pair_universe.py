"""Stage 1: co-seating universe per phase, and identification of the eval filter rule."""
import sys, time
sys.path.insert(0, "src")
import polars as pl
from pokercol import config as C

t0 = time.time()
seats = pl.scan_parquet(C.SEATS).select("hand_id", "player_id")
hands = pl.scan_parquet(C.HANDS).select("hand_id", "phase", "table_id")
s = seats.join(hands, on="hand_id", how="inner")

a = s.select("hand_id", "phase", "table_id", pl.col("player_id").alias("p1"))
b = s.select("hand_id", pl.col("player_id").alias("p2"))
uni = (
    a.join(b, on="hand_id", how="inner")
    .filter(pl.col("p1") < pl.col("p2"))
    .group_by("phase", "table_id", "p1", "p2")
    .agg(pl.len().alias("shared_hands"))
)
uni = uni.collect(engine="streaming")
print(f"pair universe rows: {uni.height:,}  ({time.time()-t0:.0f}s)")
print(uni.group_by("phase").agg(pl.len().alias("pairs"), pl.col("shared_hands").min().alias("min_sh"),
                                pl.col("shared_hands").median().alias("med_sh")))
uni.write_parquet(C.CACHE / "pair_universe.parquet")

ep = pl.read_csv(C.EVAL_PAIRS).with_columns(
    pl.min_horizontal("player_1", "player_2").alias("p1"),
    pl.max_horizontal("player_1", "player_2").alias("p2"),
)
ev_uni = uni.filter(pl.col("phase") == "evaluation")
j = ep.join(ev_uni, on=["p1", "p2"], how="left")
print("\n=== eval_pairs vs computed evaluation co-seating ===")
print("eval rows:", ep.height, "| matched:", j.filter(pl.col("shared_hands").is_not_null()).height)
print("shared_hands mismatch:", j.filter(pl.col("shared_hands") != pl.col("shared_hands_right")).height)
excl = ev_uni.join(ep.select("p1", "p2"), on=["p1", "p2"], how="anti")
print("co-seated eval pairs NOT in eval_pairs:", excl.height)
print("  their shared_hands max:", excl["shared_hands"].max(), "| included min:", ep["shared_hands"].min())

lab = pl.read_csv(C.DEV_LABELS).with_columns(
    pl.min_horizontal("player_1", "player_2").alias("p1"),
    pl.max_horizontal("player_1", "player_2").alias("p2"),
)
dev_uni = uni.filter(pl.col("phase") == "development")
lj = lab.join(dev_uni, on=["p1", "p2"], how="left")
print("\n=== labeled pairs dev shared_hands ===")
print(lj.group_by("label").agg(pl.len(), pl.col("shared_hands").min().alias("min"),
                               pl.col("shared_hands").median().alias("med"),
                               pl.col("shared_hands").max().alias("max"),
                               pl.col("shared_hands").is_null().sum().alias("nulls")))
print("\ndev pairs with shared_hands >= 38:", dev_uni.filter(pl.col("shared_hands") >= 38).height)
print(f"total {time.time()-t0:.0f}s")
