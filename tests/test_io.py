filepath = os.path.join(DATA_DIR, 'VT1.nvt')
nvt = nrtk.io.load_nvt(filepath)

filepath = os.path.join(DATA_DIR, 'CSC1.ncs')
ncs = nrtk.io.load_ncs(filepath)

signals = ncs['Samples']
ncs_stamps = ncs['TimeStamp']

print(signals.shape)
print(ncs_stamps.shape)

# (50863, 512)
# (50863,)

nvt_stamps = nvt['TimeStamp']
ncs_stamps = ncs['TimeStamp']
ncs2_stamps = ncs2['TimeStamp']

print(nvt_stamps.shape)
print(ncs_stamps.shape)
print(ncs2_stamps.shape)

print(nvt_stamps[:10])
print(ncs_stamps[:10])
print(ncs2_stamps[:10])



nvt_diff = np.diff(nvt_stamps)
ncs_diff = np.diff(ncs_stamps)

print('diffs')
print(nvt_diff)
print(ncs_diff)
# We remark that there is 16000 time stamps diff on average

stamps = np.intersect1d(nvt_stamps, ncs_stamps)
print(len(stamps))

time_origin = np.min([nvt_stamps[0], ncs_stamps[0]])
nvt_stamps = nvt_stamps - time_origin
ncs_stamps = ncs_stamps - time_origin
print('NVT time stamps: min = %d, max = %d' %(nvt_stamps[0], nvt_stamps[-1]))
print('NCS time stamps: min = %d, max = %d' %(ncs_stamps[0], ncs_stamps[-1]))
print(nvt_stamps)
print(ncs_stamps)


# If the time origin is the nvt
if nvt_stamps[0] == 0:
    ncs_origin = ncs_stamps[0]
    nvt_start = find_nearest_idx(nvt_stamps, ncs_origin) - 1
    nvt_stamps = nvt_stamps[nvt_start:]
else:
    nvt_origin = nvt_stamps[0]
    ncs_start = find_nearest_idx(ncs_stamps, nvt_origin) - 1
    ncs_stamps = ncs_stamps[ncs_start:]

print('\nAfter alignment')
print('NVT time stamps: min = %d, max = %d' %(nvt_stamps[0], nvt_stamps[-1]))
print('NCS time stamps: min = %d, max = %d' %(ncs_stamps[0], ncs_stamps[-1]))
print(nvt_stamps.shape)
print(ncs_stamps.shape)
print(nvt_stamps)
print(ncs_stamps)

diff = -len(nvt_stamps) + len(ncs_stamps)
print(diff)

idx_to_remove = np.round(np.linspace(0, len(ncs_stamps) - 1, diff)).astype(int)

ncs_stamps = np.delete(ncs_stamps, idx_to_remove)
# Withdrawing diff elements from ncs_stamps

print('\nAfter Crop')
print('NVT time stamps: min = %d, max = %d' %(nvt_stamps[0], nvt_stamps[-1]))
print('NCS time stamps: min = %d, max = %d' %(ncs_stamps[0], ncs_stamps[-1]))
print(nvt_stamps.shape)
print(ncs_stamps.shape)
print(nvt_stamps)
print(ncs_stamps)

print('\nEnd lag')
print((int(nvt_stamps[-1]) - int(ncs_stamps[-1]))/ 16000)
# OK, end time lag is negligible
