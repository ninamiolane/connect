"""Spike sorting."""

import numpy as np

# Spike threshold in standard deviations
SPIKE_THRESHOLD_STD = 3.1

# Peak window in number of time steps
PEAK_WINDOW_CLIP = 30
N_PCA_COMPONENTS = 10
N_CLUSTERS = 3


def mad(data, axis=None):
    """Mean Absolute Deviation."""
    nanmean = np.nanmean(data, axis)
    deviations = np.abs(data - nanmean)
    mad = np.nanmean(deviations, axis)
    return mad


def minima_mask(signals):
    """
    Find minima in the signals: True if there is a minimum, False if not.
    """
    signals = np.atleast_2d(signals)
    n_electrodes, _ = signals.shape

    slopes = np.diff(signals, axis=-1)

    neg_slope_mask = slopes[:, :-1] < 0
    pos_slope_mask = slopes[:, 1:] > 0
    minima_mask = np.logical_and(neg_slope_mask, pos_slope_mask)

    # First and last element cannot be minima
    false = np.concatenate([[False]] * n_electrodes, axis=0)
    false = np.expand_dims(false, axis=1)
    minima_mask = np.concatenate([false, minima_mask, false], axis=1)

    return minima_mask


def peaks_mask(signals, threshold_std=SPIKE_THRESHOLD_STD):
    """
    Find peaks in the signals:
    True if there is a peak, False if not.
    Note: the std of the data is estimated through the MAD,
    which is more robust to noise.
    """
    signals = np.atleast_2d(signals)
    n_electrodes, _ = signals.shape

    min_mask = minima_mask(signals)

    std = 1.4826 * mad(signals)

    threshold = - threshold_std * std
    threshold_mask = signals[:, 1:-1] < threshold

    # First and last element cannot be peaks
    false = np.concatenate([[False]] * n_electrodes, axis=0)
    false = np.expand_dims(false, axis=1)
    threshold_mask = np.concatenate([false, threshold_mask, false], axis=1)

    peaks_mask = np.logical_and(min_mask, threshold_mask)
    return peaks_mask


def extract_peaks_ids(signals, threshold_std=SPIKE_THRESHOLD_STD):
    """
    Code based from Abdul's code &
    https://www.frontiersin.org/articles/10.3389/fnins.2016.00537/full

    Peaks are defined as minima that are threshold*std away from 0
    Returns: 1 if there is a peak, 0 if not
    """
    n_electrodes, n_signals = signals.shape

    mask = peaks_mask(signals, threshold_std)

    peaks = {}
    for electrode_id in range(n_electrodes):
        peaks_ids = np.where(mask[electrode_id])[0]
        print('Electrode {}. Found {} peaks over {} recorded data.'.format(
            electrode_id+1, len(peaks_ids), n_signals))

        peaks[electrode_id] = peaks_ids

    return peaks


def extract_peaks(signals, peak_ids, clip=PEAK_WINDOW_CLIP):

    n_electrodes, n_time_steps = signals.shape
    peaks = {}
    for electrode_id in range(n_electrodes):
        electrode_peak_ids = peak_ids[electrode_id]

        selected_peak_signals = []
        selected_peak_ids = []  # peak ids that are conserved

        for peak_id in electrode_peak_ids:
            start_idx = np.int(peak_id - clip / 2)
            end_idx = np.int(peak_id + clip / 2)

            if (start_idx > 1 and end_idx < n_time_steps):
                peak_signal = signals[electrode_id, start_idx:end_idx]
                if np.sum(np.isnan(peak_signal)) == 0:
                    selected_peak_signals.append(peak_signal)
                    selected_peak_ids.append(peak_id)

        selected_peak_signals = np.array(selected_peak_signals)
        assert selected_peak_signals.shape[1] == clip

        peaks[electrode_id] = selected_peak_signals
    return peaks
