"""Filtering and signal processing."""

import numpy as np
from scipy import signal

SF = 32000  # sampling rate s^-1
NYQ = SF / 2

LOWCUT = 600  # Hz
HIGHCUT = 7000  # Hz
DELTA = 100  # Hz

CHEBY_ORDER = 6
CHEBY_RIPPLE = 0.01  # DB

CHEBY_LOWCUT_HZ = LOWCUT - 4 * DELTA  # 200 Hz
CHEBY_LOWCUT = CHEBY_LOWCUT_HZ / NYQ  # 0.0125 NYQ units
CHEBY_HIGHCUT_HZ = HIGHCUT + 4 * DELTA  # 7400 Hz
CHEBY_HIGHCUT = CHEBY_HIGHCUT_HZ / NYQ  # 0.4625 NYQ units

CHEBY_SPECS = (
    'Cheby filter cutoff frequencies: '
    '[{:.4f}, {:.4f}] = [{:.0f}Hz, {:.0f}Hz]'.format(
        CHEBY_LOWCUT, CHEBY_HIGHCUT, CHEBY_LOWCUT_HZ, CHEBY_HIGHCUT_HZ))

# Order of FIR needs to be even, so that FIR length is odd, and FIR is type I
FIR_ORDER = 50

FIR_LOWCUT_HZ = LOWCUT - DELTA  # 500 Hz
FIR_LOWCUT = FIR_LOWCUT_HZ / NYQ  # 0.03125 NYQ units
FIR_HIGHCUT_HZ = HIGHCUT + DELTA  # 7100 Hz
FIR_HIGHCUT = FIR_HIGHCUT_HZ / NYQ  # 0.44375 NYQ units

FIR_SPECS = (
    'FIR filter cutoff frequencies: '
    '[{:.4f}, {:.4f}] = [{:.0f}Hz, {:.0f}Hz]'.format(
        FIR_LOWCUT, FIR_HIGHCUT, FIR_LOWCUT_HZ, FIR_HIGHCUT_HZ))

EXTREME_AMPLITUDE_COEF = 0.9

MIN_FLAT = 5
PAD_FLAT = 100


def cheby(signals,
          order=CHEBY_ORDER, ripple=CHEBY_RIPPLE,
          lowcut=CHEBY_LOWCUT, highcut=CHEBY_HIGHCUT):
    """
    Cheby filter: bandpass filter with ripples.
    """
    cutoff_frequencies = [lowcut, highcut]

    b0, a0 = signal.cheby1(
        N=order,
        rp=ripple,
        Wn=cutoff_frequencies,
        btype='bandpass')
    filtered_signals = signal.filtfilt(b0, a0, signals, axis=-1)
    return filtered_signals, b0, a0


def firwin(signals,
           order=FIR_ORDER,
           lowcut=FIR_LOWCUT, highcut=FIR_HIGHCUT):
    """
    Low order Hann Filter: bandpass filter with ripplets.

    Length of the filter = number of coefficients = filter order + 1.
    """
    length = order + 1
    cutoff_frequencies = [lowcut, highcut]

    b1 = signal.firwin(
        numtaps=length,
        cutoff=cutoff_frequencies,
        pass_zero=False,  # signal.firwin is bandstop by default
        window='hann')
    beq = np.convolve(b1, b1)  # 2 stage filter

    # Applying filters, with 0 phase lag
    filtered_signals = signal.filtfilt(beq, 1, signals, axis=-1)
    return filtered_signals, beq


def remove_extreme_amplitudes(signals, coef=EXTREME_AMPLITUDE_COEF):
    """
    Remove the data above 90% of the extremum values.
    Extremum values are computed per electrode.
    """
    # TODO(nina): Memory optimization:
    # Consider in place version of this function.
    # TODO(nina): Speed-up:
    # Vectorize code wrt electrodes.
    signals = np.atleast_2d(signals)
    n_electrodes, n_time_steps = signals.shape

    min_amplitude = np.nanmin(signals, axis=-1)
    max_amplitude = np.nanmax(signals, axis=-1)

    min_amplitude = np.expand_dims(min_amplitude, axis=1)
    max_amplitude = np.expand_dims(max_amplitude, axis=1)
    interval = np.concatenate([min_amplitude, max_amplitude], axis=1)

    valid_interval = interval * coef

    low_bool = np.zeros_like(signals)
    high_bool = np.zeros_like(signals)
    for i in range(n_electrodes):
        low_bool[i] = signals[i] <= valid_interval[i, 0]
        high_bool[i] = signals[i] >= valid_interval[i, 1]

    bad_amplitudes_mask = np.logical_or(low_bool, high_bool)
    n_bad_signals = np.sum(bad_amplitudes_mask, axis=-1)

    for i in range(n_electrodes):
        print(
            'Electrode: {}. Found {}/{} points with extreme amplitude. '
            'Fill with NaN.'.format(i+1, n_bad_signals[i], n_time_steps))

    filtered_signals = np.copy(signals)
    filtered_signals[bad_amplitudes_mask] = np.nan
    return filtered_signals


def contiguous_regions(condition):
    """
    Finds contiguous True regions of the boolean array "condition".

    Returns a 2D array where:
    - the first column is the start index of the region,
    - the second column is the end index.
    """
    # TODO(nina): Speed-up: vectorization code wrt electrodes.
    condition = np.atleast_2d(condition)
    n_electrodes, _ = condition.shape
    # Changes in "condition"
    d = np.diff(condition, axis=-1)

    indices = {}
    for i in range(n_electrodes):
        idx, = d[i].nonzero()
        condition_i = condition[i]

        # We need to start things after the change in "condition". Therefore,
        # we'll shift the index by 1 to the right.
        idx += 1

        if condition_i[0]:
            # If the start of condition is True prepend a 0
            idx = np.r_[0, idx]

        if condition_i[-1]:
            # If the end of condition is True, append the length of the array
            idx = np.r_[idx, condition_i.size]

        # Reshape the result into two columns
        idx.shape = (-1, 2)
        indices[i] = idx

    return indices


def remove_flat(raw_signals, signals, min_flat=MIN_FLAT, pad_flat=PAD_FLAT):
    """
    Remove data in signals around flat regions of the **raw** signals.
    Minimum of min_flat consecutives 0 slopes.
    Pad with pad_flat before and after the consecutives 0 slopes.

    Then rescale around 0 mean.
    """
    # TODO(nina): Memory optimization: consider doing this function in place.
    # TODO(nina): Speed-up: vectorization code wrt electrodes.
    signals = np.atleast_2d(signals)
    n_electrodes, n_time_steps = signals.shape
    filtered_signals = np.copy(signals)

    slopes = np.diff(raw_signals, axis=-1)
    slope0_mask = slopes == 0

    contiguous_indices = contiguous_regions(slope0_mask)

    for i in range(n_electrodes):
        contiguous_ids = contiguous_indices[i]
        lengths = np.array([end - start + 1 for start, end in contiguous_ids])

        for length, ids in zip(lengths, contiguous_ids):
            if length >= min_flat:
                start_remove = np.max([0, ids[0] - pad_flat])
                end_remove = np.min([n_time_steps, ids[1] + pad_flat + 1])
                filtered_signals[i, start_remove:end_remove] = np.nan

    return filtered_signals


def normalize(signals, axis=None):
    nanmean = np.nanmean(signals, axis=axis)
    nanstd = np.nanstd(signals, axis=axis)
    normalized_signals = (signals - nanmean) / nanstd
    return normalized_signals


def remove_flat_and_normalize(raw_signals, signals,
                              min_flat=MIN_FLAT, pad_flat=PAD_FLAT):
    filtered_signals = remove_flat(
        raw_signals, signals, min_flat=min_flat, pad_flat=pad_flat)
    filtered_signals = normalize(filtered_signals)
    return filtered_signals
