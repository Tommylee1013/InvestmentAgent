import pandas as pd

def month_end_business_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    grp = pd.Series(1, index=index).groupby(index.to_period('M'))
    last_idx = grp.apply(lambda s: s.index.max())
    return pd.DatetimeIndex(last_idx.values)

def nearest_on_or_before(idx: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp | None:
    """idx에서 t 이전(포함) 가장 가까운 날짜 반환; 없으면 None."""
    pos = idx.searchsorted(t, side="right") - 1
    if pos < 0:
        return None
    return idx[pos]