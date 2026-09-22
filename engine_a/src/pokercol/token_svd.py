"""Action-sequence tokens compressed to a few dimensions for the evidence rankers.

`pokercol.tokens` turns a (pair, hand) into sparse n-gram counts. Truncated
SVD of the log counts, fitted on the positive pairs' hands, gives every hand a
short dense code of its action sequence that the lambdarank models can read.
The vocabulary and the SVD are fitted once on the training hands and applied
unchanged to evaluation hands, in chunks.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from . import tokens as T

MIN_ROWS = 15
CHUNK = 300_000


class TokenSVD:
    def __init__(self, dims: int, acts: pl.DataFrame):
        self.dims = dims
        self.acts = acts
        self.vocab: list[str] | None = None
        self.svd: TruncatedSVD | None = None

    def _counts(self, rows: pl.DataFrame) -> pl.DataFrame:
        parts = []
        for start in range(0, rows.height, CHUNK):
            part = rows.slice(start, CHUNK).with_row_index("cid", offset=start).with_columns(pl.col("cid").cast(pl.Int64))
            parts.append(T.candidate_counts(part, self.acts))
        return pl.concat(parts)

    def fit_transform(self, rows: pl.DataFrame) -> np.ndarray:
        counts = self._counts(rows)
        self.vocab = T.vocabulary(counts, MIN_ROWS)
        x = T.to_csr(counts, self.vocab, rows.height).log1p().tocsr()
        self.svd = TruncatedSVD(n_components=self.dims, random_state=0)
        return self.svd.fit_transform(x)

    def transform(self, rows: pl.DataFrame) -> np.ndarray:
        assert self.vocab is not None and self.svd is not None
        x = T.to_csr(self._counts(rows), self.vocab, rows.height).log1p().tocsr()
        return self.svd.transform(x)

    def columns(self) -> list[str]:
        return [f"tsvd_{i}" for i in range(self.dims)]

    def attach(self, df: pl.DataFrame, z: np.ndarray) -> pl.DataFrame:
        cols = self.columns()
        df = df.with_columns([pl.Series(c, z[:, i]) for i, c in enumerate(cols)])
        return df.with_columns([(pl.col(c).rank("average") / pl.len()).over("pair_id").alias(f"pct_{c}") for c in cols[:8]])

    def feature_names(self) -> list[str]:
        cols = self.columns()
        return cols + [f"pct_{c}" for c in cols[:8]]
