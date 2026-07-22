"""No-leakage splitting (protocol section 7).

Split at UNIT granularity: no unit's rows appear in both train and test. For real
data with multiple time-window rows per unit, key every window by its unit_id and
this function keeps each unit whole.
"""
import numpy as np
from .arms.base import Dataset


def unit_level_split(unit_id, seed, test_frac=0.3):
    rng = np.random.default_rng(seed)
    uids = np.unique(np.asarray(unit_id))
    rng.shuffle(uids)
    ncut = int(round((1.0 - test_frac) * len(uids)))
    return set(uids[:ncut].tolist()), set(uids[ncut:].tolist())


def subset(ds: Dataset, unit_set) -> Dataset:
    mask = np.isin(ds.unit_id, np.array(list(unit_set)))
    return Dataset(unit_id=ds.unit_id[mask], Ttil=ds.Ttil[mask],
                   event=ds.event[mask], x_obs=ds.x_obs[mask])
