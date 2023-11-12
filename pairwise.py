from scipy.sparse import issparse
import numpy as np
from joblib import effective_n_jobs
from validation import check_array


def check_pairwise_arrays(X, Y, precomputed=False, dtype=None, accept_sparse='csr', force_all_finite=True, copy=False):
    X, Y, dtype_float = _return_float_dtype(X, Y)

    estimator = 'check_pairwise_arrays'
    if dtype is None:
        dtype = dtype_float

    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse=accept_sparse, dtype=dtype,
                            copy=copy, force_all_finite=force_all_finite,
                            estimator=estimator)
    else:
        X = check_array(X, accept_sparse=accept_sparse, dtype=dtype,
                        copy=copy, force_all_finite=force_all_finite,
                        estimator=estimator)
        Y = check_array(Y, accept_sparse=accept_sparse, dtype=dtype,
                        copy=copy, force_all_finite=force_all_finite,
                        estimator=estimator)

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError("Precomputed metric requires shape "
                             "(n_queries, n_indexed). Got (%d, %d) "
                             "for %d indexed." %
                             (X.shape[0], X.shape[1], Y.shape[0]))
    elif X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    return X, Y


def additive_chi2_kernel(X, Y=None):
    if issparse(X) or issparse(Y):
        raise ValueError("additive_chi2 does not support sparse matrices.")
    X, Y = check_pairwise_arrays(X, Y)
    if (X < 0).any():
        raise ValueError("X contains negative values.")
    if Y is not X and (Y < 0).any():
        raise ValueError("Y contains negative values.")

    result = np.zeros((X.shape[0], Y.shape[0]), dtype=X.dtype)
    _chi2_kernel_fast(X, Y, result)
    return result


def chi2_kernel(X, Y=None, gamma=1.):
    K = additive_chi2_kernel(X, Y)
    K *= gamma
    return np.exp(K, K)


# Kernels
def linear_kernel(X, Y=None, dense_output=True):
    X, Y = check_pairwise_arrays(X, Y)
    return safe_sparse_dot(X, Y.T, dense_output=dense_output)


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = safe_sparse_dot(X, Y.T, dense_output=True)
    K *= gamma
    K += coef0
    K **= degree
    return K


def rbf_kernel(X, Y=None, gamma=None):
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = euclidean_distances(X, Y, squared=True)
    K *= -gamma
    np.exp(K, K)  # exponentiate K in-place
    return K


def laplacian_kernel(X, Y=None, gamma=None):
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = -gamma * manhattan_distances(X, Y)
    np.exp(K, K)  # exponentiate K in-place
    return K


def sigmoid_kernel(X, Y=None, gamma=None, coef0=1):
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = safe_sparse_dot(X, Y.T, dense_output=True)
    K *= gamma
    K += coef0
    np.tanh(K, K)  # compute tanh in-place
    return K


def cosine_similarity(X, Y=None, dense_output=True):
    # to avoid recursive import
    X, Y = check_pairwise_arrays(X, Y)
    X_normalized = normalize(X, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, copy=True)

    K = safe_sparse_dot(X_normalized, Y_normalized.T,
                        dense_output=dense_output)

    return K


# Helper functions - distance
PAIRWISE_KERNEL_FUNCTIONS = {
    # If updating this dictionary, update the doc in both distance_metrics()
    # and also in pairwise_distances()!
    'additive_chi2': additive_chi2_kernel,
    'chi2': chi2_kernel,
    'linear': linear_kernel,
    'polynomial': polynomial_kernel,
    'poly': polynomial_kernel,
    'rbf': rbf_kernel,
    'laplacian': laplacian_kernel,
    'sigmoid': sigmoid_kernel,
    'cosine': cosine_similarity, }


# Utility Functions
def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float is returned.
    """
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = np.float

    return X, Y, dtype

def _parallel_pairwise(X, Y, func, n_jobs, **kwds):
    """Break the pairwise matrix in n_jobs even slices
    and compute them in parallel"""
    if Y is None:
        Y = X
    X, Y, dtype = _return_float_dtype(X, Y)

    if effective_n_jobs(n_jobs) == 1:
        return func(X, Y, **kwds)

    # enforce a threading backend to prevent data communication overhead
    fd = delayed(_dist_wrapper)
    ret = np.empty((X.shape[0], Y.shape[0]), dtype=dtype, order='F')
    Parallel(backend="threading", n_jobs=n_jobs)(
        fd(func, ret, s, X, Y[s], **kwds)
        for s in gen_even_slices(_num_samples(Y), effective_n_jobs(n_jobs)))

    if (X is Y or Y is None) and func is euclidean_distances:
        # zeroing diagonal for euclidean norm.
        # TODO: do it also for other norms.
        np.fill_diagonal(ret, 0)

    return ret


def pairwise_kernels(X, Y=None, metric="linear", filter_params=False, n_jobs=None, **kwds):
    # import GPKernel locally to prevent circular imports
    from kernels import Kernel as GPKernel

    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return X
    elif isinstance(metric, GPKernel):
        func = metric.__call__
    elif metric in PAIRWISE_KERNEL_FUNCTIONS:
        if filter_params:
            kwds = {k: kwds[k] for k in kwds
                    if k in KERNEL_PARAMS[metric]}
        func = PAIRWISE_KERNEL_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(_pairwise_callable, metric=metric, **kwds)
    else:
        raise ValueError("Unknown kernel %r" % metric)

    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)


