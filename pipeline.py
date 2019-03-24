"""NeuroRoots pipeline."""

import logging
import luigi
import matplotlib
matplotlib.use('Agg')  # NOQA
import os
import warnings
warnings.filterwarnings('ignore')  # NOQA

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import nrtk.filter
import nrtk.io
import nrtk.sort
import nrtk.vis


HOME_DIR = '/scratch/users/nmiolane/sommet/'
DATA_DIR = '/neuro/recordings/2018-05-31_15-43-39'

OUTPUT_DIR = os.path.join(HOME_DIR, 'output')

DEBUG = False

N_ELECTRODES = 32
SF = 32000
N_PCA_COMPONENTS = 10
N_CLUSTERS = 3


class LoadData(luigi.Task):
    """
    Load NCS from all electrodes,
    Extract signals:
    - Array of shape: n_electrodes, 512 * n_time_steps.

    Extract 2D positions:
    - Array of shape: 2, n_time_steps

    Note: time resolution for positions and signals are different:
    There are 512 signal recordings for 1 position.
    """
    output_path = os.path.join(OUTPUT_DIR, 'load_data.npy')

    def requires(self):
        pass

    def run(self):
        signals = []
        positions = []

        filepath = os.path.join(DATA_DIR, 'VT1.nvt')
        nvt = nrtk.io.load_nvt(filepath)
        nvt_stamps = nvt['TimeStamp']

        filepath = os.path.join(DATA_DIR, 'CSC1.ncs')
        ncs = nrtk.io.load_ncs(filepath)
        ncs_stamps = ncs['TimeStamp']

        nvt_idx, ncs_idx = nrtk.io.align_timestamps(
            nvt_stamps, ncs_stamps)

        # Extract positions from nvt
        pxl_to_cm_x = 1  # placeholder
        pxl_to_cm_y = 1  # placeholder
        x_cm = nvt['extracted_x'] / pxl_to_cm_x
        y_cm = nvt['extracted_y'] / pxl_to_cm_y

        x_cm = x_cm[nvt_idx]
        y_cm = y_cm[nvt_idx]

        positions = np.vstack([x_cm, y_cm])

        # Extract signals from ncs's
        for electrode_id in range(N_ELECTRODES):
            filename = 'CSC{}.ncs'.format(electrode_id + 1)
            filepath = os.path.join(DATA_DIR, filename)

            logging.info('Loading %s...' % filepath)
            electrode_ncs = nrtk.io.load_ncs(filepath)

            electrode_signal = electrode_ncs['Samples']
            electrode_signal = electrode_signal[ncs_idx]
            electrode_signal = electrode_signal.ravel()

            sf = electrode_ncs['SampleFreq'][0]
            assert sf == SF

            signals.append(electrode_signal)

        signals = np.array(signals)
        n_electrodes, n_time_steps = signals.shape
        assert n_electrodes == N_ELECTRODES

        data = {
            'positions': positions,
            'signals': signals}
        np.save(self.output().path, data)

    def output(self):
        return luigi.LocalTarget(self.output_path)


class FilterSignals(luigi.Task):
    """
    Filter before removing bad signals, as filtering does not work with Nans.
    - Extreme amplitude: mouse bumping against a wall
    - Flat regions: saturation of the device
    """
    # TODO(nina): Parallel with joblib
    output_path = os.path.join(OUTPUT_DIR, 'filter_signals.npy')

    def requires(self):
        return {'load_data': LoadData()}

    def run(self):
        data_path = self.input()['load_data'].path
        data = np.load(data_path).item()
        signals = data['signals']

        filtered_signals, _, _ = nrtk.filter.cheby(signals)
        filtered_signals, _ = nrtk.filter.firwin(filtered_signals)

        filtered_signals = nrtk.filter.remove_extreme_amplitudes(
            filtered_signals)
        filtered_signals = nrtk.filter.remove_flat_and_normalize(
            signals, filtered_signals)

        np.save(self.output().path, filtered_signals)

    def output(self):
        return luigi.LocalTarget(self.output_path)


class ExtractSpikes(luigi.Task):
    output_path = os.path.join(OUTPUT_DIR, 'extract_spikes.npy')

    # TODO(nina): Parallel with joblib
    def requires(self):
        return {'filter_signals': FilterSignals()}

    def run(self):
        signals_path = self.input()['filter_signals'].path
        signals = np.load(signals_path)

        peaks_ids = nrtk.sort.extract_peaks_ids(signals)
        peaks = nrtk.sort.extract_peaks(signals, peaks_ids)

        pca_kmeans = {}

        n_electrodes, _ = signals.shape

        for electrode_id in range(n_electrodes):
            data = peaks[electrode_id]

            pca = PCA(n_components=N_PCA_COMPONENTS)
            projected_data = pca.fit_transform(data)

            kmeans = KMeans(
                init='k-means++', n_clusters=N_CLUSTERS,
                n_init=10, random_state=1990)
            kmeans_res = kmeans.fit(projected_data)

            pca_kmeans[electrode_id] = {
                'data': data,
                'projected_data': projected_data,
                'explained_variance': pca.explained_variance_ratio_,
                'assignments': kmeans_res.labels_,
                'centers': kmeans_res.cluster_centers_}

        np.save(self.output().path, pca_kmeans)

    def output(self):
        return luigi.LocalTarget(self.output_path)


class RunAll(luigi.Task):
    def requires(self):
        return {'extract_spikes': ExtractSpikes()}

    def output(self):
        return luigi.LocalTarget('dummy')


def init():
    for directory in [OUTPUT_DIR]:
        if not os.path.isdir(directory):
            os.mkdir(directory)
            os.chmod(directory, 0o777)

    logging.basicConfig(level=logging.INFO)
    logging.info('start')
    luigi.run(
        main_task_cls=RunAll(),
        cmdline_args=[
            '--local-scheduler',
        ])


if __name__ == "__main__":
    init()
