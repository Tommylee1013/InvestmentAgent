import numpy as np
import pandas as pd

from scipy.optimize import minimize

def nearest_psd(cov: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    cov = (cov + cov.T) * 0.5
    vals, vecs = np.linalg.eigh(cov.values)
    vals = np.clip(vals, eps, None)
    out = vecs @ np.diag(vals) @ vecs.T
    return pd.DataFrame(out, index=cov.index, columns=cov.columns)

def cov_to_corr(cov: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    std = np.sqrt(np.diag(cov.values)).clip(min=eps)
    inv_std = np.diag(1.0 / std)
    corr = inv_std @ cov.values @ inv_std
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)

def corr_to_cov(corr: pd.DataFrame, vol: pd.Series) -> pd.DataFrame:
    D = np.diag(vol.values)
    cov = D @ corr.values @ D
    return pd.DataFrame(cov, index=corr.index, columns=corr.columns)

def corr_distance(corr: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.sqrt(0.5*(1.0 - corr.clip(-1,1))),
                        index=corr.index, columns=corr.columns)

def minvar_longonly_from_cov(cov: pd.DataFrame) -> pd.Series:
    cov = nearest_psd(cov)
    n = len(cov)
    if n == 1:
        return pd.Series([1.0], index=cov.index)
    C = cov.values
    def obj(w): return w @ C @ w
    cons = [{'type':'eq','fun':lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, 1.0)] * n
    x0 = np.ones(n)/n
    res = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons)
    if not res.success:
        raise ValueError(f"MinVar failed: {res.message}")
    return pd.Series(res.x, index=cov.index)

def ivp_weights_from_cov(cov: pd.DataFrame, eps: float = 1e-12) -> pd.Series:
    var = pd.Series(np.diag(cov.values), index=cov.index).clip(lower=eps)
    inv = 1.0 / var
    w = inv / inv.sum()
    return w