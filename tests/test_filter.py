def test_filter_cheby():
    n = 200
    m = 150
    zeros = np.zeros(n)
    ones = np.ones(m)

    print('\nTest single signal')
    test_samples = np.concatenate([zeros, ones, zeros])
    times = range(len(test_samples))
    result, _, _ = filter_cheby(test_samples)

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 2, 1)
    plt.plot(times, test_samples)
    plt.subplot(2, 2, 2)
    plt.plot(times, result)

    print('\nTest multiple electrodes')
    n_electrodes = 3
    test_samples_1 = np.concatenate([zeros, ones, zeros])
    test_samples_2 = np.concatenate([np.ones_like(zeros), ones, zeros])
    test_samples_3 = np.concatenate([zeros, ones, np.ones_like(zeros)])

    test_samples = np.vstack([test_samples_1, test_samples_2, test_samples_3])
    print(test_samples.shape)
    times = range(test_samples.shape[1])
    result, _, _ = filter_cheby(test_samples)

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 2, 1)
    for i in range(n_electrodes):
        plt.plot(times, test_samples[i])
    plt.subplot(2, 2, 2)
    for i in range(n_electrodes):
        plt.plot(times, result[i])
test_filter_cheby()

# TODO(nina): Adapt SF et amplitude to see the impact


test_filter_firwin():
    n = 200
    m = 150
    zeros = np.zeros(n)
    ones = np.ones(m)

    print('\nTest single signal')
    test_samples = np.concatenate([zeros, ones, zeros])
    times = range(len(test_samples))
    result, _= filter_firwin(test_samples)

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 2, 1)
    plt.plot(times, test_samples)
    plt.subplot(2, 2, 2)
    plt.plot(times, result)

    print('\nTest multiple electrodes')
    n_electrodes = 3
    test_samples_1 = np.concatenate([zeros, ones, zeros])
    test_samples_2 = np.concatenate([np.ones_like(zeros), ones, zeros])
    test_samples_3 = np.concatenate([zeros, ones, np.ones_like(zeros)])

    test_samples = np.vstack([test_samples_1, test_samples_2, test_samples_3])
    print(test_samples.shape)
    times = range(test_samples.shape[1])
    result, _ = filter_firwin(test_samples)

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 2, 1)
    for i in range(n_electrodes):
        plt.plot(times, test_samples[i])
    plt.subplot(2, 2, 2)
    for i in range(n_electrodes):
        plt.plot(times, result[i])
test_filter_firwin()

def test_remove_extreme_amplitudes():
    n_electrodes = 3
    print('\nTest with one electrode')
    test_samples = np.array([-0.1, 0.1, -0.2, 10, -10, 0.2])
    result = remove_extreme_amplitudes(test_samples)
    # expected = np.array([-0.1, 0.1, -0.2, np.nan, np.nan, 0.2])
    print(test_samples)
    print(result)

    print('\nTest with one electrode')
    test_samples = np.array([-0.1, np.nan, -0.2, 10, -10, 0.2])
    result = remove_extreme_amplitudes(test_samples)
    # expected = np.array([-0.1, np.nan, -0.2, np.nan, np.nan, 0.2])
    print(test_samples)
    print(result)


    print('\nTest with %d electrode - axis=None' % n_electrodes)
    test_samples = np.array([
        [-0.1, np.nan, -0.2, 10, -10, 0.2],
        [-1, np.nan, -2, 100, -100, 3],
        [-0.1, -1000, -0.2, 10, -10, 0.2]])
    result = remove_extreme_amplitudes(test_samples, axis=None)
    # expected = np.array([-0.1, np.nan, -0.2, np.nan, np.nan, 0.2])
    print(test_samples)
    print(result)

    print('\nTest with %d electrode - axis=-1' % n_electrodes)
    test_samples = np.array([
        [-0.1, np.nan, -0.2, 10, -10, 0.2],
        [-1, np.nan, -2, 100, -100, 3],
        [-0.1, -1000, -0.2, 10, -10, 0.2]])
    result = remove_extreme_amplitudes(test_samples)
    # expected = np.array([-0.1, np.nan, -0.2, np.nan, np.nan, 0.2])
    print(test_samples)
    print(result)

test_remove_extreme_amplitudes()

def test_remove_flat():
    print('\ntest with not enough consecutives flatness -  no pad')
    test_samples = np.array([1, 2, 2, 2, 2, 2, 2, 2, 5, 6, 7, 7, 7, 8, 8, 9], dtype=np.float64)
    result = remove_flat(test_samples, test_samples, min_flat=9, pad_flat=0)
    print(result)

    print('\ntest with enough consecutives flatness - no pad')
    test_samples = np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 6, 7, 7, 7, 8, 8, 9], dtype=np.float64)
    result = remove_flat(test_samples, test_samples, pad_flat=0)
    print(result)

    print('\ntest with enough consecutives flatness - pad 1')
    test_samples = np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 6, 7, 7, 7, 8, 8, 9], dtype=np.float64)
    result = remove_flat(test_samples, test_samples, pad_flat=1)
    print(result)

    print('\ntest with enough consecutives flatness - pad 3')
    test_samples = np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 6, 7, 7, 7, 8, 8, 9], dtype=np.float64)
    result = remove_flat(test_samples, test_samples, pad_flat=3)
    print(result)

    print('\ntest with enough consecutives flatness - pad 100')
    test_samples = np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 6, 7, 7, 7, 8, 8, 9], dtype=np.float64)
    result = remove_flat(test_samples, test_samples, pad_flat=100)
    print(result)
test_remove_flat()
