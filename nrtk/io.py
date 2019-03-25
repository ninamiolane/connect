"""I/O functions."""

import numpy as np

HEADER_SIZE = 16 * 1024  # Header size in neuralynx formats


N_ELECTRODES = 32
SF = 32000

NCS_FORMAT = np.dtype([
    ('TimeStamp', np.uint64),
    ('ChannelNumber', np.uint32),
    ('SampleFreq', np.uint32),
    ('NumValidSamples', np.uint32),
    ('Samples', np.int16, 512)])

NVT_FORMAT = np.dtype([
    ('stx', np.uint16),
    ('id', np.uint16),
    ('data_size', np.uint16),
    ('TimeStamp', np.uint64),  # in um
    ('Points', np.uint32, 400),
    ('nrc', np.int16),  # Unused
    ('extracted_x', np.int32),
    ('extracted_y', np.int32),
    ('extracted_angle', np.int32),
    ('targets', np.int32, 50)])


def load_ncs(data_file, electrode_id=1):
    """
    Read Neuralynx NCS data.
    https://neuralynx.com/software/NeuralynxDataFileFormats.pdf

    Note: Electrodes are numbered 1-32 in filenames, but 0-31 in NCS.
    """
    with open(data_file, 'rb') as fid:
        fid.seek(HEADER_SIZE)
        raw = np.fromfile(fid, dtype=NCS_FORMAT)

    assert np.all(raw['SampleFreq'] == SF)
    if not np.all(raw['ChannelNumber'] == electrode_id - 1):
        print(
            '! - Warning: There are %d elements whose ChannelNumber'
            ' does not correspond to the electrode ID.' % sum(
                raw['ChannelNumber'] != electrode_id - 1))

    return raw


def load_nvt(data_file, remove_empty=False):
    """
    Read Neuralynx NVT data.
    https://neuralynx.com/software/NeuralynxDataFileFormats.pdf
    """
    with open(data_file, 'rb') as fid:
        fid.seek(HEADER_SIZE)
        raw = np.fromfile(fid, dtype=NVT_FORMAT)

    selected_raw = dict()
    selected_raw['TimeStamp'] = raw['TimeStamp']
    selected_raw['extracted_x'] = np.array(raw['extracted_x'], dtype=float)
    selected_raw['extracted_y'] = np.array(raw['extracted_y'], dtype=float)
    selected_raw['targets'] = np.array(raw['targets'], dtype=float)

    if remove_empty:
        empty_idx = (raw['extracted_x'] == 0) & (raw['extracted_y'] == 0)
        for key in selected_raw:
            selected_raw[key] = selected_raw[key][~empty_idx]

    return selected_raw


def find_nearest_idx(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def align_timestamps(stamps_1, stamps_2):
    time_origin = np.min([stamps_1[0], stamps_2[0]])

    # Align time origins
    if stamps_1[0] == time_origin:
        origin_2 = stamps_2[0]
        start_1 = find_nearest_idx(stamps_1, origin_2) - 1
        stamps_1 = stamps_1[start_1:]
    else:
        origin_1 = stamps_1[0]
        start_2 = find_nearest_idx(stamps_2, origin_1) - 1
        stamps_2 = stamps_2[start_2:]

    # We observed that the time ends are aligned.
    # We resample the over-sampled timestamps.
    diff = len(stamps_1) - len(stamps_2)
    if diff > 0:
        idx_to_remove = np.round(
            np.linspace(0, len(stamps_1) - 1, diff)).astype(int)

        idx_to_keep_1 = np.delete(np.arange(len(stamps_1)), idx_to_remove)
        stamps_1 = stamps_1[idx_to_keep_1]

        idx_to_keep_2 = np.range(len(stamps_2))
    else:
        idx_to_remove = np.round(
            np.linspace(0, len(stamps_2) - 1, np.abs(diff)).astype(int))

        idx_to_keep_2 = np.delete(np.arange(len(stamps_2)), idx_to_remove)
        stamps_2 = stamps_2[idx_to_keep_2]

        idx_to_keep_1 = np.arange(len(stamps_1))

    assert len(stamps_1) == len(stamps_2)

    start_lag = (int(stamps_1[0]) - int(stamps_2[0])) / (16000 * 32000)
    end_lag = (int(stamps_1[-1]) - int(stamps_2[-1])) / (16000 * 32000)

    assert np.abs(start_lag) < 1e-4, start_lag  # i.e. < 0.1ms
    assert np.abs(end_lag) < 1e-4, end_lag  # i.e. < 0.1ms

    return idx_to_keep_1, idx_to_keep_2
