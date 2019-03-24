"""Visualization."""

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy import signal

import nrtk.filter

SF = nrtk.filter.SF


def get_time(signal, sf=SF):
    # TODO(nina): Add time stamps
    n_time_steps = signal.shape[-1]
    duration_secs = n_time_steps / sf

    time = np.linspace(0, duration_secs, n_time_steps)
    return time


def plot_signal(ax, signal, time=None, sf=SF,
                t_min=0., t_max=1., label='signal'):
    """Plot voltage signal between t_min seconds and t_max seconds.

    Args:
        time (ndarray): time vector in secs.
        signal (ndarray): data to plot.
        sf (uint32): sampling frequency.
        t_min: start of x axis (default 0), in secs.
        t_max: end of x axis (default 1), in secs.
    """
    if time is None:
        time = get_time(signal, sf)

    t_min_id = int(t_min * sf)
    t_max_id = int(t_max * sf)

    time_msecs = 1000 * time[t_min_id:t_max_id]
    ax.plot(
        time_msecs,
        signal[t_min_id:t_max_id],
        label=label)

    ax.set_xlim(time_msecs[0], time_msecs[-1])

    ax.set_xlabel('time [ms]', fontsize=20)
    ax.set_ylabel('signal [uV]', fontsize=20)

    return ax


def plot_signal_electrodes(ax, signal_electrodes, time=None, sf=SF,
                           t_min=0., t_max=1., electrodes_ids=None):

    if time is None:
        time = get_time(signal_electrodes, sf)

    n_electrodes, _ = signal_electrodes.shape
    if electrodes_ids is None:
        electrodes_ids = range(n_electrodes)

    for electrode_id in electrodes_ids:
        ax = plot_signal(
            ax, signal_electrodes[electrode_id], time,
            sf, t_min, t_max, 'Electrode %d' % (electrode_id + 1))

    chartBox = ax.get_position()
    ax.set_position(
        [chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1), shadow=True, ncol=4)

    return ax


def plot_frequency_response_analog(b, a):
    """Analog: use freqs."""
    w, h = signal.freqs(b, a, worN=np.logspace(-1, 2, 1000))

    plt.figure(figsize=(10, 8))

    plt.semilogx(w, 20 * np.log10(abs(h)))

    plt.xlabel('Frequency', fontsize=20)
    plt.ylabel('Amplitude response [dB]', fontsize=20)
    plt.grid()
    plt.show()


def plot_frequency_response_digital(b, a=1):
    """Digital: use freqz."""
    w, h = signal.freqz(b, a, worN=np.logspace(-1, 2, 1000))

    plt.figure(figsize=(10, 8))

    plt.semilogx(w, 20 * np.log10(abs(h)))

    plt.ylabel('Amplitude response [dB]', fontsize=20)
    plt.grid()
    plt.show()


def plot_correlation(fig, ax, signals):
    n_electrodes, _ = signals.shape
    df = pd.DataFrame(
        data=np.transpose(signals),
        columns=['%d' % (i+1) for i in range(n_electrodes)])

    corr_matrix = df.corr()

    cax = ax.imshow(corr_matrix, interpolation="nearest", cmap='viridis')
    ax.grid(True)
    plt.title('Correlation Matrix of Electrodes', fontsize=20)

    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.columns)), df.columns)

    cbar = fig.colorbar(cax, ticks=np.linspace(-1., 1, num=9))
    cbar.set_label('Correlations', rotation=270)
    return ax


def plot_pca_kmeans(ax, pca_kmeans_electrode, sf=SF):
    projected_data = pca_kmeans_electrode['projected_data']
    assignments = pca_kmeans_electrode['assignments']
    variances = pca_kmeans_electrode['explained_variance']

    n_clusters = len(pca_kmeans_electrode['centers'])
    cmap = plt.cm.get_cmap('viridis', n_clusters)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    colors = [cmaplist[a] for a in assignments]
    ax.scatter(projected_data[:, 0], projected_data[:, 1], c=colors)
    ax.set_xlabel('PC1 - variance explained: {:.4f}'.format(variances[0]))
    ax.set_ylabel('PC2 - variance explained: {:.4f}'.format(variances[1]))
    return ax


def plot_spikes_summary(ax, spikes, sf=SF, title='', color='blue'):
    time = get_time(spikes, sf)
    # Only plot first 20 spikes
    for spike in spikes[:20]:
        ax.plot(time, spike, color='grey')

    mean_spike = np.mean(spikes, axis=0)
    ax.plot(time, mean_spike, c=color, linewidth=5)

    std_spike = np.std(spikes, axis=0)
    upper_spike = mean_spike + std_spike
    lower_spike = mean_spike - std_spike

    ax.fill_between(time, lower_spike, upper_spike, color=color, alpha=0.3)

    ax.set_title(label=title, fontsize=23)
    ax.set_xlim(0, time[-1])
    ax.set_xlabel('time [ms]', fontsize=20)
    return ax


def plot_centers(ax, pca_kmeans_electrode, center_id=0):
    # TODO(nina): Add firing rate
    data = pca_kmeans_electrode['data']
    assignments = pca_kmeans_electrode['assignments']

    cluster_mask = assignments == center_id
    cluster_data = data[cluster_mask, :]

    n_clusters = len(pca_kmeans_electrode['centers'])
    cmap = plt.cm.get_cmap('viridis', n_clusters)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    ax = plot_spikes_summary(ax, cluster_data, color=cmaplist[center_id])
    title = 'Cluster {}'.format(center_id)
    ax.set_title(title)
    return ax
