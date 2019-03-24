"""I/O functions."""

import os

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
    ('TimeStamp', np.uint64), # in um
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

    assert np.all(raw['SampleFreq']==SF)
    if not np.all(raw['ChannelNumber']==electrode_id-1):
        print(
            '! - Warning: There are %d elements whose ChannelNumber'
            ' does not correspond to the electrode ID.' % sum(raw['ChannelNumber']!=electrode_id-1))

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
