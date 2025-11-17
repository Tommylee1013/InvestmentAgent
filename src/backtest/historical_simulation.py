import numpy as np
import pandas as pd

def build_ledger_from_daily_prices_and_weights_monthly(
    data: pd.DataFrame,
    weights: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    tc_bps: float = 25.0,
    weight_lag_days: int = 0,                 # 0=월말 당일 weight 사용
    charge_cost_on_first_rebalance: bool = False,  # 첫 리밸런싱 비용 면제(기본 False = 면제)
):
    cols = sorted(list(set(data.columns) & set(weights.columns)))
    data = data[cols].copy().sort_index().ffill()
    weights = weights[cols].copy().reindex(index=data.index, columns=cols)
    if not isinstance(data.index, pd.DatetimeIndex): data.index = pd.to_datetime(data.index)
    if not isinstance(weights.index, pd.DatetimeIndex): weights.index = pd.to_datetime(weights.index)

    # 월말(가용 영업일) 인덱스
    month_end_idx = data.index.to_series().groupby(data.index.to_period('M')).max()
    month_end_idx = month_end_idx[month_end_idx.isin(weights.index)].sort_values()
    month_end_set = set(month_end_idx.values)
    # 첫 리밸런싱 일자
    first_rebal_date = month_end_idx.iloc[0] if len(month_end_idx) else None

    # 타겟 가중 (래깅 적용 후 행정규화)
    target_w = weights.shift(weight_lag_days).fillna(0.0)
    rs = target_w.sum(axis=1).replace(0.0, np.nan)
    target_w = target_w.div(rs, axis=0).fillna(0.0)

    cost_rate = tc_bps / 10_000.0
    idx = data.index
    m = len(cols)

    shares = pd.DataFrame(0.0, index=idx, columns=cols)
    trades = pd.DataFrame(0.0, index=idx, columns=cols)
    cash = np.zeros(len(idx)); port_val = np.zeros(len(idx))
    tcost = np.zeros(len(idx)); dollar_turnover = np.zeros(len(idx))
    turnover_pct = np.zeros(len(idx)); ret_gross = np.zeros(len(idx))
    ret_net = np.zeros(len(idx)); gross_expo_ts = np.zeros(len(idx))

    cash_prev = initial_capital
    shares_prev = np.zeros(m, dtype=float)
    pv_prev = initial_capital

    for i, t in enumerate(idx):
        prices_t = data.iloc[i].values.astype(float)
        holdings_val_pre = np.nansum(shares_prev * prices_t)
        pv_pre = cash_prev + holdings_val_pre

        trade_val = np.zeros_like(shares_prev)
        dt_notional = 0.0; dt_cost = 0.0
        shares_after = shares_prev.copy(); cash_after = cash_prev

        if t in month_end_set:
            w_t = target_w.loc[t].values.astype(float)
            desired_val = w_t * pv_pre
            current_val = shares_prev * prices_t
            trade_val = desired_val - current_val
            tradable = np.isfinite(prices_t)
            trade_val = np.where(tradable, trade_val, 0.0)

            dt_notional = float(np.abs(trade_val).sum())
            # 첫 리밸런싱 비용 면제 옵션
            apply_cost = not (charge_cost_on_first_rebalance is False and (first_rebal_date is not None) and (t == first_rebal_date))
            dt_cost = (cost_rate * dt_notional) if apply_cost else 0.0

            cash_after = cash_prev - trade_val.sum() - dt_cost
            with np.errstate(divide='ignore', invalid='ignore'):
                delta_shares = np.where(tradable, trade_val / prices_t, 0.0)
                delta_shares[~np.isfinite(delta_shares)] = 0.0
            shares_after = shares_prev + delta_shares

        holdings_val_after = np.nansum(shares_after * prices_t)
        pv_after = cash_after + holdings_val_after

        gross_expo_today = (np.nansum(np.abs(shares_after * prices_t)) / pv_after) if pv_after != 0 else np.nan
        r_gross = (holdings_val_after + cash_prev - pv_prev) / pv_prev if i > 0 else 0.0
        r_net = (pv_after - pv_prev) / pv_prev if i > 0 else 0.0

        trades.iloc[i, :] = trade_val
        shares.iloc[i, :] = shares_after
        cash[i] = cash_after; port_val[i] = pv_after
        tcost[i] = dt_cost; dollar_turnover[i] = dt_notional
        turnover_pct[i] = (dt_notional / pv_pre) if pv_pre != 0 else 0.0
        ret_gross[i] = r_gross; ret_net[i] = r_net; gross_expo_ts[i] = gross_expo_today

        shares_prev = shares_after; cash_prev = cash_after; pv_prev = pv_after

    ledger = pd.DataFrame(
        {"portfolio_value": port_val, "cash": cash, "dollar_turnover": dollar_turnover,
         "turnover_pct": turnover_pct, "tcost": tcost, "ret_gross": ret_gross,
         "ret_net": ret_net, "gross_exposure": gross_expo_ts},
        index=idx,
    )
    return ledger, shares, trades