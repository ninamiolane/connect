def test_correlation():
    # TODO(nina): Why 7 correlation values? Why does one nan impact 4 values?
    print('\nTest without Nans')
    test_samples = np.array([
        [1., 2., 3., 4.],
        [2., 3., 4., 5.]])
    corr = signal.correlate(test_samples[0], test_samples[1])
    print(corr)

    print('\nTest with Nans')
    test_samples = np.array([
        [1., 2., np.nan, 4.],
        [2., 3., 4., 5.]])
    corr = signal.correlate(test_samples[0], test_samples[1])
    print(corr)
    corr_nanmean = np.nanmean(corr)
    print(corr_nanmean)

    print('\nTest with Nans using masked arrays')
    x = test_samples[0]
    y = test_samples[1]
    nan_x = np.isnan(x)
    print(nan_x)
    nan_y = np.isnan(y)
    mask = np.logical_or(nan_x, nan_y)
    print(mask)
    normx = np.linalg.norm(x[~nan_x])
    print(normx)
    normy = np.linalg.norm(y[~nan_y])
    ma_x = np.ma.array(x, mask=nan_x)
    ma_y = np.ma.array(y, mask=nan_y)
    print(ma_x)
    print(ma_y)

    corr = np.correlate(ma_x, ma_y)
    # corr = corr / (normx * normy)
    print(corr)

test_correlation()
