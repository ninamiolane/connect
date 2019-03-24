def test_get_minima_mask():
    print('\nTest - a single electrode')
    test_samples = np.array([1, 2, 3, 4, 3, 4, 1, -1, 1])
    result = get_minima_mask(test_samples)
    print(result)

    print('\nTest - 3 electrodes')
    test_samples = np.array([
        [1, 2, 3, 4, 3, 4, 1, -1, 1],
        [0, 1, 2, 3, 2, 3, 0, -2, 0],
        [-1, 0, 1, 2, 1, 2, -1, -3, -1]])
    result = get_minima_mask(test_samples)
    print(result)
test_get_minima_mask()

def test_get_peaks_mask():
    print('\nTest: 1 std away from 0, small minimum - a single electrode')
    test_samples = np.array([1, -2, 6, 4, 3, 4, 1, -1, 1])
    print(test_samples)
    print('Std: {:.4}'.format(np.nanstd(test_samples)))
    result = get_peaks_mask(test_samples, threshold_std=1)
    print(result)

    print('\nTest: 1 std away from 0, large minimum - a single electrode')
    test_samples = np.array([1, -2, 6, 4, 3, 4, 1, -10, 1])
    print(test_samples)
    print('Std: {:.4}'.format(np.nanstd(test_samples)))
    result = get_peaks_mask(test_samples, threshold_std=1)
    print(result)

    print('\nTest: 3 std away from 0, large minimum - a single electrode')
    test_samples = np.array([1, -20, 6, 4, 3, 4, 1, -10, 1])
    print(test_samples)
    print('Std: {:.4}'.format(np.nanstd(test_samples)))
    result = get_peaks_mask(test_samples, threshold_std=1)
    print(result)

    print('\nTest: 3 std away from 0, large minimum 3 electrodes')
    test_samples = np.array([
        [1, -20, 6, 4, 3, 4, 1, -10, 1],
        [0, -21, 5, 3, 2, 3, 0, -11, 0],
        [-1, -22, 4, 2, 1, 2, -1, -12, -1]])
    print(test_samples)
    result = get_peaks_mask(test_samples, threshold_std=1)
    print(result)
test_get_peaks_mask()

def test_get_peaks():
    print('\nTest: 1 std away from 0, small minimum')
    test_samples = np.array([1, -2, 6, 4, 3, 4, 1, -1, 1])
    print(test_samples)
    print('Std: {:.4}'.format(np.nanstd(test_samples)))
    result = get_peaks(test_samples, threshold_std=1)
    print(result)

    print('\nTest: 1 std away from 0, large minimum')
    test_samples = np.array([1, -2, 6, 4, 3, 4, 1, -10, 1])
    print(test_samples)
    print('Std: {:.4}'.format(np.nanstd(test_samples)))
    result = get_peaks(test_samples, threshold_std=1)
    print(result)

    print('\nTest: 3 std away from 0, large minimum')
    test_samples = np.array([1, -20, 6, 4, 3, 4, 1, -10, 1])
    print(test_samples)
    print('Std: {:.4}'.format(np.nanstd(test_samples)))
    result = get_peaks(test_samples, threshold_std=1)
    print('\nTest: 3 std away from 0, large minimum 3 electrodes')
    test_samples = np.array([
        [1, -20, 6, 4, 3, 4, 1, -10, 1],
        [0, -21, 5, 3, 2, 3, 0, -11, 0],
        [-1, -22, 4, 2, 1, 2, -1, -12, -1]])
    print(test_samples)
    result = get_peaks(test_samples, threshold_std=1)
    print(result)
test_get_peaks()
