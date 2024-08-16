from copy import deepcopy
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal, fft, ndimage


def smooth_and_interpolate_coordinates(coordinates, method='rgds',
                                       window_length=9, kernel_type='gaussian', polynomial_order=3, std_dev=2,
                                       weight_data=0.5, weight_smooth=0.5, tolerance=0.000001, remove_duplicates=True):
    if remove_duplicates:
        coordinates, indices = np.unique(coordinates, axis=0, return_index=True)
        x, y = coordinates[:, 0], coordinates[:, 1]
    # l = len(x)
    # t = np.linspace(0, 1, l - 2, endpoint=True)
    # t = np.append([0, 0, 0], t)
    # t = np.append(t, [1, 1, 1])
    # tck = [t, [x, y], 3]
    # u3 = np.linspace(0, 1, (max(l * 2, 70)), endpoint=True)
    # new_coordinates = interpolate.splev(u3, tck)

    if method == 'bspline':
        # scipy splprep
        interpolation_function, u = interpolate.splprep([x, y], s=weight_smooth, k=polynomial_order)
        # u = np.linspace(0, 1, num=50, endpoint=True)
        x_new, y_new = interpolate.splev(u, interpolation_function)
        new_coordinates = np.vstack((x_new, y_new)).T

        # # Scipy BSpline. Doesn't work
        # t, c, k = interpolation_function
        # c1 = np.asarray(c)
        # spline = interpolate.BSpline(t, c1.T, k, extrapolate=False)
        # new_coordinates = spline(coordinates[:, 0])

    elif method == 'moving_average':
        new_coordinates = moving_average(coordinates, window_length=window_length,
                                              kernel_type=kernel_type, std_dev=std_dev)

    elif method == 'savgol':
        new_coordinates = savgol_filter(coordinates, window_length=window_length,
                                             polynomial_order=polynomial_order)

    elif method == 'rgds':
        new_coordinates = recursive_gradient_descent_smoother(coordinates,
                                                                   weight_data=weight_data,
                                                                   weight_smooth=weight_smooth, tolerance=tolerance)

    # # fft. Doesn't work
    # w = fft.rfft(y)
    # f = fft.rfftfreq(window_length, x[1]-x[0])
    # spectrum = w ** 2
    # cutoff_idx = spectrum < (spectrum.max() / 1)
    # w2 = w.copy()
    # w2[cutoff_idx] = 0
    # y2 = fft.irfft(w2)

    return new_coordinates

def moving_average(noisy_points, window_length=10, kernel_type='gaussian', std_dev=2):
    """

    :param noisy_points:
    :param window_length:
    :param kernel_type: gaussian, hann, box
    :param std_dev:
    :return:
    """
    smoothed_points = np.zeros_like(noisy_points)
    kernels = {'gaussian': signal.windows.gaussian(window_length, std=std_dev),
               'hann': signal.windows.hann(window_length),
               'box': np.ones(window_length) / window_length}
    weights = kernels.setdefault(kernel_type, 'gaussian')

    for dim in range(noisy_points.shape[1]):
        column_data = noisy_points[:, dim]
        if kernel_type == 'box':
            # uniform_filter is faster than both cumsum and convolve (see https://stackoverflow.com/a/43200476)
            smoothed_points[:, dim] = ndimage.uniform_filter1d(column_data, window_length,
                                                               mode='nearest')

            # smoothed_points[:, dim] = np.convolve(column_data, weights, mode='same')
            # smoothed_points[:, dim] = signal.convolve(column_data, weights, mode='same') # same as above

            # # cumsum method is faster than the convolve method
            # cumsum = np.cumsum(np.insert(column_data, 0, 0))
            # smoothed_points = (cumsum[window_length:] - cumsum[:-window_length]) / window_length

        else:
            smoothed_points[:, dim] = np.convolve(column_data, weights, mode='same') / sum(weights)

    return smoothed_points


def savgol_filter(noisy_points, window_length=10, polynomial_order=3):
    smoothed_points = np.zeros_like(noisy_points)
    for dim in range(noisy_points.shape[1]):
        column_data = noisy_points[:, dim]
        smoothed_points[:, dim] = signal.savgol_filter(column_data, window_length=window_length,
                                                       polyorder=polynomial_order)
    return smoothed_points


def recursive_gradient_descent_smoother(path, weight_data=0.5, weight_smooth=0.1, tolerance=0.000001):
    """
    A model free recursive smoother.
    Creates a smooth path for a n-dimensional series of coordinates.
    Arguments:
        path: List containing coordinates of a path
        weight_data: Float, how much weight to update the data (alpha)
        weight_smooth: Float, how much weight to smooth the coordinates (beta).
        tolerance: Float, how much change per iteration is necessary to keep iterating.
    Output:
        new: List containing smoothed coordinates.
    """
    new = deepcopy(path)
    dims = len(path[0])
    change = tolerance

    while change >= tolerance:
        change = 0.0
        for i in range(1, len(new) - 1):
            for j in range(dims):
                x_i = path[i][j]
                y_i, y_prev, y_next = new[i][j], new[i - 1][j], new[i + 1][j]

                y_i_saved = y_i
                y_i += weight_data * (x_i - y_i) + weight_smooth * (y_next + y_prev - (2 * y_i))
                new[i][j] = y_i

                change += abs(y_i - y_i_saved)

    return new
