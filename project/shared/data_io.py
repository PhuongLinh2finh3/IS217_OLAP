import os
import numpy as np
import pandas as pd
from shared.config import *

def load_splits(categorical=False):
    # Doc tu file tinh, khong dung run_id
    parts = {
        'train': pd.read_parquet(PATH_TRAIN),
        'valid': pd.read_parquet(PATH_VALID),
        'test': pd.read_parquet(PATH_TEST)
    }
    out = {n: (p[FEATURES].copy(), p[TARGET].copy()) for n, p in parts.items()}
    
    if categorical:
        for col in CATEGORICAL_COLS:
            cats = np.sort(pd.concat([p[col] for p in parts.values()]).unique())
            for n in out:
                out[n][0][col] = pd.Categorical(out[n][0][col], categories=cats)
    return out
