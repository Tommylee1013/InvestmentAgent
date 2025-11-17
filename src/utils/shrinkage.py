import numpy as np
import pandas as pd

def mp_denoise_constant_corr(corr: pd.DataFrame, T: int) -> pd.DataFrame:
    """Marčenko–Pastur constant-eigenvalue clipping on correlation."""
    n = corr.shape[0]
    if n <= 1:
        return corr.copy()
    q = max(T / float(n), 1.0000001)  # 안정성
    lambda_plus = (1.0 + 1.0/np.sqrt(q))**2

    vals, vecs = np.linalg.eigh(corr.values)
    mask = vals <= lambda_plus
    if mask.sum() > 0:
        vals[mask] = vals[mask].mean()
    C = vecs @ np.diag(vals) @ vecs.T
    C = (C + C.T) * 0.5
    C = pd.DataFrame(C, index=corr.index, columns=corr.columns)
    np.fill_diagonal(C.values, 1.0)
    return C

def detone_market_mode(corr: pd.DataFrame) -> pd.DataFrame:
    """Remove top eigenmode (market), then restore unit diagonal."""
    vals, vecs = np.linalg.eigh(corr.values)
    i_max = np.argmax(vals)
    v1 = vecs[:, [i_max]]
    l1 = vals[i_max]
    C = corr.values - l1 * (v1 @ v1.T)
    C = (C + C.T) * 0.5
    C = pd.DataFrame(C, index=corr.index, columns=corr.columns)
    np.fill_diagonal(C.values, 1.0)
    return C