import numpy as np
from scipy.special import polygamma, digamma
from scipy.optimize import newton
from scipy.stats import norm
from proteoflux.analysis.robust_prior import tmixture_vector

def fit_fdist_robust(s2, df):
    s0 = tmixture_vector(s2, df)
    d0 = 2 * (1 / np.var(np.log(s2)) - 1)
    return s0, d0

def squeeze_var_input_filter(s2: np.ndarray, df) -> tuple[np.ndarray, np.ndarray]:
    # If df is scalar, broadcast it to shape of s2
    if np.isscalar(df) or np.ndim(df) == 0:
        df = np.full_like(s2, df)

    mask = np.isfinite(s2) & (s2 > 0) & np.isfinite(df) & (df > 0)
    return s2[mask], df[mask]

def trigamma_inverse(y, tol=1e-8):
    # Initial guess
    if y > 1e7:
        x = 1.0 / np.sqrt(y)
    elif y < 1e-6:
        x = 1.0 / y
    else:
        x = 0.5 + 1.0 / y

    # Newton-Raphson method
    for _ in range(50):
        tri = polygamma(1, x)
        delta = (tri - y) / polygamma(2, x)
        x = x - delta
        if abs(delta) < tol:
            return x
    return x

def fit_fdist(s2: np.ndarray, df1: np.ndarray) -> tuple[float, float]:
    x = s2
    df1 = np.asarray(df1)
    x = x[(x > 0) & np.isfinite(x) & np.isfinite(df1) & (df1 > 0)]

    if x.size == 0:
        return np.nan, np.nan

    # Avoid zeros like limma does
    x = np.maximum(x, 1e-5 * np.median(x))
    z = np.log(x)

    d = df1[0] if np.all(df1 == df1[0]) else df1  # handle scalar case if needed
    e = z - digamma(d / 2.0) + np.log(d / 2.0)
    emean = np.mean(e)
    evar = np.var(e, ddof=1)

    trigam = polygamma(1, d / 2.0)
    evar_adj = evar - trigam

    if evar_adj > 0:
        df2 = 2 * trigamma_inverse(evar_adj)
        s20 = np.exp(emean + digamma(df2 / 2.0) - np.log(df2 / 2.0))
    else:
        df2 = np.inf
        s20 = np.exp(emean)

    return s20, df2

