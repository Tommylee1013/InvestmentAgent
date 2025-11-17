import pandas as pd

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from src.utils.util import *
from src.utils.shrinkage import *
from src.allocation.black_litterman import *

def cluster_portfolio_variance(cov: pd.DataFrame, names: list[str], within: str='ivp') -> float:
    sub = cov.loc[names, names]
    w = ivp_weights_from_cov(sub) if within=='ivp' else minvar_longonly_from_cov(sub)
    return float(w.values @ sub.values @ w.values)

def nco_weights(cov: pd.DataFrame, linkage_method: str='ward', within: str='ivp') -> pd.Series:
    if cov.shape[0] == 1:
        return pd.Series([1.0], index=cov.index)
    corr = cov_to_corr(cov)
    dist = corr_distance(corr)
    Z = linkage(squareform(dist.values, checks=False), method=linkage_method)
    order = leaves_list(Z)
    ordered = list(cov.index[order])

    w = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            k = len(cl)//2
            L, R = cl[:k], cl[k:]
            vL = cluster_portfolio_variance(cov, L, within)
            vR = cluster_portfolio_variance(cov, R, within)
            aL = 1.0 - vL/(vL+vR)
            aR = 1.0 - aL
            w[L] *= aL
            w[R] *= aR
            nxt += [L, R]
        clusters = nxt
    w = w.clip(lower=0)
    w /= w.sum()
    return w.reindex(cov.index)

def rolling_posterior_nco(
    returns: pd.DataFrame,         # 일간 수익률 (index=영업일, columns=자산)
    market_weight: pd.DataFrame,   # 일간(or 영업일) 타깃 비중 (각 행 합≈1)
    lookback_days: int = 252,
    linkage_method: str = 'ward',
    within: str = 'ivp',
    # BL hyper
    delta: float = 3.0,
    tau: float = 0.02,
    confidence: float = 0.5,
    # denoise/detone
    denoise: bool = True,
    detone: bool = True,
    fillna_zero: bool = False
) -> pd.DataFrame:

    dates = returns.index
    mes = month_end_business_days(dates)
    mw_idx = market_weight.index

    weights_at_me = {}

    for i in range(1, len(mes)):  # 직전 월말 필요
        t = mes[i]
        t_prev = mes[i-1]

        # market_weight에서 't_prev와 같거나 직전'에 해당하는 가장 가까운 날짜를 사용
        mw_t_prev = nearest_on_or_before(mw_idx, t_prev)
        if mw_t_prev is None:
            continue

        pos = dates.get_loc(t)
        if isinstance(pos, slice):
            pos = pos.stop - 1
        start = pos - lookback_days
        if start < 0:
            continue

        window = returns.iloc[start:pos]  # t 전일까지(look-ahead 방지)
        valid = window.dropna(axis=1, how='any')
        if valid.shape[1] == 0:
            continue

        # 표본 Σ
        Sigma_raw = nearest_psd(valid.cov())

        # 디노이즈/디톤 (상관 기준)
        if denoise or detone:
            vols = pd.Series(np.sqrt(np.diag(Sigma_raw.values)), index=Sigma_raw.index)
            Corr = cov_to_corr(Sigma_raw)
            if denoise:
                Corr = mp_denoise_constant_corr(Corr, T=len(window))
            if detone:
                Corr = detone_market_mode(Corr)
            Sigma = nearest_psd(corr_to_cov(Corr, vols))
        else:
            Sigma = Sigma_raw

        # 뷰: '직전 월말(또는 그 이전 최근 영업일)'의 market_weight 사용
        w_target = market_weight.loc[mw_t_prev].reindex(Sigma.index).fillna(0.0)
        if w_target.sum() <= 0:
            continue
        w_target = w_target / w_target.sum()

        # BL posterior (predictive): Σ_post = Σ + M
        views = build_views_from_target_weights(Sigma, w_target, delta=delta, tau=tau, confidence=confidence)
        P = views["P"].reindex(index=Sigma.index, columns=Sigma.index)
        Q = views["Q"].reindex(index=Sigma.index)
        Omega = views["Omega"].reindex(index=Sigma.index, columns=Sigma.index)
        Sigma_post = bl_posterior_cov(Sigma, P, Q, Omega, tau=views["tau"])

        # NCO (long-only)
        try:
            w_nco = nco_weights(Sigma_post, linkage_method=linkage_method, within=within)
            w = blend_with_view(w_nco, w_target, alpha=confidence)
        except Exception:
            w = ivp_weights_from_cov(Sigma_post)

        weights_at_me[t] = w

    # 일간으로 확장
    if not weights_at_me:
        out = pd.DataFrame(index=dates, columns=returns.columns, dtype=float)
        return out.fillna(0.0) if fillna_zero else out

    daily_map = {}
    me_sorted = sorted(weights_at_me.keys())
    for j, me in enumerate(me_sorted):
        w = weights_at_me[me]
        if j < len(me_sorted) - 1:
            nxt = me_sorted[j+1]
            mask = (dates >= me) & (dates < nxt)
        else:
            mask = (dates >= me)
        for d in dates[mask]:
            daily_map[d] = w

    weights_df = pd.DataFrame(daily_map).T.reindex(dates)
    weights_df = weights_df.reindex(columns=returns.columns)
    if fillna_zero:
        weights_df = weights_df.fillna(0.0)
    return weights_df