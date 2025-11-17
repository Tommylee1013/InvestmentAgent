import numpy as np
import pandas as pd

from src.utils.stats import *

def build_views_from_target_weights(
    Sigma: pd.DataFrame,
    w_target: pd.Series,
    delta: float = 3.0,
    tau: float = 0.02,
    confidence: float = 0.5
):
    """
    후험 예측용 뷰 생성:
      - μ_view = δ Σ w_target (역최적화)
      - P = I, Q = μ_view
      - Ω = ((1-c)/c) * τ * diag(Σ)
    """
    w = w_target.reindex(Sigma.index).fillna(0.0)
    if w.sum() <= 0:
        raise ValueError("w_target has zero mass on current universe.")
    w = w / w.sum()

    mu_view = delta * (Sigma.values @ w.values.reshape(-1,1))  # n x 1
    Q = pd.Series(mu_view.ravel(), index=Sigma.index)
    P = pd.DataFrame(np.eye(len(Sigma)), index=Sigma.index, columns=Sigma.index)
    scale = (1.0 - confidence) / confidence  # c=0.5 -> 1
    Omega_diag = scale * tau * np.diag(Sigma.values)          # vector length n
    Omega = pd.DataFrame(np.diag(Omega_diag), index=Sigma.index, columns=Sigma.index)
    return {"P": P, "Q": Q, "Omega": Omega, "tau": tau, "delta": delta}

def bl_posterior_mean(
    Sigma: pd.DataFrame,
    P: pd.DataFrame,
    Q: pd.Series,
    Omega: pd.DataFrame,
    tau: float,
    pi: pd.Series = None  # 사전 평형수익률(없으면 0 벡터로 가정)
) -> pd.Series:
    """
    μ_post = M [ (τΣ)^-1 π + Pᵀ Ω^-1 Q ],  M = ((τΣ)^-1 + Pᵀ Ω^-1 P)^-1
    """
    Σ = nearest_psd(Sigma)
    n = Σ.shape[0]
    π = (pi.reindex(Σ.index) if pi is not None else pd.Series(0.0, index=Σ.index)).values.reshape(-1,1)

    Σ_tau_inv = np.linalg.pinv(tau * Σ.values)
    Ω_inv = np.linalg.pinv(Omega.values)
    Pm = P.reindex(columns=Σ.index).values
    Qv = Q.reindex(index=Σ.index).values.reshape(-1,1)

    M_inv = Σ_tau_inv + Pm.T @ Ω_inv @ Pm
    M = np.linalg.pinv(M_inv)
    mu_post = M @ (Σ_tau_inv @ π + Pm.T @ Ω_inv @ Qv)
    return pd.Series(mu_post.ravel(), index=Σ.index)

def bl_posterior_cov(Sigma: pd.DataFrame, P: pd.DataFrame, Q: pd.Series, Omega: pd.DataFrame, tau: float) -> pd.DataFrame:
    """
    후험 예측 공분산: Σ_post = Σ + M,  M = ((τΣ)^-1 + PᵀΩ^-1P)^-1
    (PSD 보장/안정형)
    """
    Σ = nearest_psd(Sigma)
    Σ_tau_inv = np.linalg.pinv(tau * Σ.values)
    Ω_inv = np.linalg.pinv(Omega.values)
    A = P.values.T @ Ω_inv @ P.values
    M_inv = Σ_tau_inv + A
    M = np.linalg.pinv(M_inv)
    Σ_post = Σ.values + M
    return nearest_psd(pd.DataFrame(Σ_post, index=Σ.index, columns=Σ.columns))

def blend_with_view(w_nco: pd.Series, w_target: pd.Series, alpha: float) -> pd.Series:
    """
    최종 가중치 = (1-alpha) * w_nco + alpha * w_target
    alpha ∈ [0,1]. 보통 confidence를 그대로 사용.
    """
    w_t = w_target.reindex(w_nco.index).fillna(0.0)
    s = w_t.sum()
    if s <= 0:  # 뷰가 비어있으면 그대로 반환
        return (w_nco.clip(lower=0) / w_nco.clip(lower=0).sum())
    w_t = w_t / s
    w = (1 - alpha) * w_nco + alpha * w_t
    w = w.clip(lower=0)
    return w / w.sum()