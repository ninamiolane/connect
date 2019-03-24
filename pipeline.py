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


class LoadSignals(luigi.Task):
    """
    Load NCS from all electrodes,
    Extract signals,
    Concatenate in an array of shape: n_signals x n_electrodes.
    """
    output_path = os.path.join(OUTPUT_DIR, 'load_signals.npy')

    def requires(self):
        pass

    def run(self):
        signals = []
        for electrode_id in range(N_ELECTRODES):
            filename = 'CSC{}.ncs'.format(electrode_id + 1)
            filepath = os.path.join(DATA_DIR, filename)

            logging.info('Loading %s...' % filepath)
            raw = nrtk.io.load_ncs(filepath)

            electrode_signal = raw['Samples'].ravel()
            sf = raw['SampleFreq'][0]
            assert sf == SF
            signals.append(electrode_signal)

        signals = np.array(signals)
        n_electrodes, n_time_steps = signals.shape
        assert n_electrodes == N_ELECTRODES

        np.save(self.output().path, signals)

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
        return {'load_signals': LoadSignals()}

    def run(self):
        signals_path = self.input()['load_signals'].path
        signals = np.load(signals_path)

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
