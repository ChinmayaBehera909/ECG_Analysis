
import numpy as np
import pandas as pd
import scipy
import scipy.spatial
import scipy.linalg

def distance(X=None, method="mahalanobis"):
    """Distance.

    Compute distance using different metrics.

    Parameters
    ----------
    X : array or DataFrame
        A dataframe of values.
    method : str
        The method to use. One of 'mahalanobis' or 'mean' for the average distance from the mean.

    Returns
    -------
    array
        Vector containing the distance values.


    """
    if isinstance(X, pd.DataFrame) is False:
        X = pd.DataFrame(X)

    method = method.lower()  # remove capitalised letters
    if method in ["mahalanobis"]:
        dist = _distance_mahalanobis(X)
    elif method in ["mean", "center", "average"]:
        dist = _distance_mean(X)
    else:
        raise ValueError("EPK error: distance(): 'method' should be one of 'mahalanobis'.")

    return dist


# =============================================================================
# Methods
# =============================================================================


def _distance_mahalanobis(X=None):
    cov = X.cov().values
    cov = scipy.linalg.inv(cov)

    col_means = X.mean().values

    dist = np.full(len(X), np.nan)
    for i in range(len(X)):
        dist[i] = scipy.spatial.distance.mahalanobis(X.iloc[i, :].values, col_means, cov) ** 2
    return dist


def _distance_mean(X=None):
    Z = standardize(X)
    dist = Z.mean(axis=1).values
    return dist


def fit_loess(y, X=None, alpha=0.75, order=2):
    """Local Polynomial Regression (LOESS)

    Performs a LOWESS (LOcally WEighted Scatter-plot Smoother) regression.


    Parameters
    ----------
    y : Union[list, np.array, pd.Series]
        The response variable (the y axis).
    X : Union[list, np.array, pd.Series]
        Explanatory variable (the x axis). If 'None', will treat y as a continuous signal (useful for smoothing).
    alpha : float
        The parameter which controls the degree of smoothing, which corresponds to the proportion
        of the samples to include in local regression.
    order : int
        Degree of the polynomial to fit. Can be 1 or 2 (default).

    Returns
    -------
    array
        Prediciton of the LOESS algorithm.

    """
    if X is None:
        X = np.linspace(0, 100, len(y))

    assert order in [1, 2], "Deg has to be 1 or 2"
    assert 0 < alpha <= 1, "Alpha has to be between 0 and 1"
    assert len(X) == len(y), "Length of X and y are different"

    X_domain = X

    n = len(X)
    span = int(np.ceil(alpha * n))

    y_predicted = np.zeros(len(X_domain))
    x_space = np.zeros_like(X_domain)

    for i, val in enumerate(X_domain):
        distance = abs(X - val)
        sorted_dist = np.sort(distance)
        ind = np.argsort(distance)

        Nx = X[ind[:span]]
        Ny = y[ind[:span]]

        delx0 = sorted_dist[span - 1]

        u = distance[ind[:span]] / delx0
        w = (1 - u ** 3) ** 3

        W = np.diag(w)
        A = np.vander(Nx, N=1 + order)

        V = np.matmul(np.matmul(A.T, W), A)
        Y = np.matmul(np.matmul(A.T, W), Ny)
        Q, R = scipy.linalg.qr(V)
        p = scipy.linalg.solve_triangular(R, np.matmul(Q.T, Y))

        y_predicted[i] = np.polyval(p, val)
        x_space[i] = val

    return y_predicted


def mad(x, constant=1.4826, **kwargs):
    """Median Absolute Deviation: a "robust" version of standard deviation.

    Parameters
    ----------
    x : Union[list, np.array, pd.Series]
        A vector of values.
    constant : float
        Scale factor. Use 1.4826 for results similar to default R.

    Returns
    ----------
    float
        The MAD.

    """
    median = np.nanmedian(np.ma.array(x).compressed(), **kwargs)
    mad_value = np.nanmedian(np.abs(x - median), **kwargs)
    mad_value = mad_value * constant
    return mad_value



def rescale(data, to=[0, 1], scale=None):
    """Rescale data.

    Rescale a numeric variable to a new range.

    Parameters
    ----------
    data : Union[list, np.array, pd.Series]
        Raw data.
    to : list
        New range of values of the data after rescaling.
    scale : list
        A list or tuple of two values specifying the actual range
        of the data. If None, the minimum and the maximum of the
        provided data will be used.

    Returns
    ----------
    list
        The rescaled values.

    """

    # Return appropriate type
    if isinstance(data, list):
        data = list(_rescale(np.array(data), to=to, scale=scale))
    else:
        data = _rescale(data, to=to, scale=scale)

    return data


# =============================================================================
# Internals
# =============================================================================
def _rescale(data, to=[0, 1], scale=None):
    if scale is None:
        scale = [np.nanmin(data), np.nanmax(data)]

    return (to[1] - to[0]) / (scale[1] - scale[0]) * (data - scale[0]) + to[0]


def standardize(data, robust=False, window=None, **kwargs):
    """Standardization of data.

    Performs a standardization of data (Z-scoring), i.e., centering and scaling, so that the data is
    expressed in terms of standard deviation (i.e., mean = 0, SD = 1) or Median Absolute Deviance
    (median = 0, MAD = 1).

    Parameters
    ----------
    data : Union[list, np.array, pd.Series]
        Raw data.
    robust : bool
        If True, centering is done by substracting the median from the variables and dividing it by
        the median absolute deviation (MAD). If False, variables are standardized by substracting the
        mean and dividing it by the standard deviation (SD).
    window : int
        Perform a rolling window standardization, i.e., apply a standardization on a window of the
        specified number of samples that rolls along the main axis of the signal. Can be used for
        complex detrending.
    **kwargs : optional
        Other arguments to be passed to ``pandas.rolling()``.

    Returns
    ----------
    list
        The standardized values.

    """
    # Return appropriate type
    if isinstance(data, list):
        data = list(_standardize(np.array(data), robust=robust, window=window, **kwargs))
    elif isinstance(data, pd.DataFrame):
        data = pd.DataFrame(_standardize(data, robust=robust, window=window, **kwargs))
    elif isinstance(data, pd.Series):
        data = pd.Series(_standardize(data, robust=robust, window=window, **kwargs))
    else:
        data = _standardize(data, robust=robust, window=window, **kwargs)

    return data


# =============================================================================
# Internals
# =============================================================================
def _standardize(data, robust=False, window=None, **kwargs):

    # Compute standardized on whole data
    if window is None:
        if robust is False:
            z = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0, ddof=1)
        else:
            z = (data - np.nanmedian(data, axis=0)) / mad(data)

    # Rolling standardization on windows
    else:
        df = pd.DataFrame(data)  # Force dataframe

        if robust is False:
            z = (df - df.rolling(window, min_periods=0, **kwargs).mean()) / df.rolling(
                window, min_periods=0, **kwargs
            ).std(ddof=1)
        else:
            z = (df - df.rolling(window, min_periods=0, **kwargs).median()) / df.rolling(
                window, min_periods=0, **kwargs
            ).apply(mad)

        # Fill the created nans
        z = z.fillna(method="bfill")

        # Restore to vector or array
        if z.shape[1] == 1:
            z = z[0].values
        else:
            z = z.values

    return z
