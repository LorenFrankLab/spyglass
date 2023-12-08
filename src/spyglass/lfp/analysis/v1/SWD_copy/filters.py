import numpy as np
from scipy.signal import butter, filtfilt

# CBroz: what's here vs in the swd_detection schema is arbitrary. I would
# keep things here that are useful across schemas/analyses and make the
# rest private methods on whatever table they're used with


def butter_bandpass(lowcut, highcut, fs, order=2):
    """Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def z_score(arr):
    return (arr - arr.mean()) / arr.std()


def compute_metric(data, metric):
    if metric == "ratio":
        max_val = data.max()
        min_val = data.min()
        return np.abs(min_val / max_val)
    elif metric == "diff":
        max_val = np.abs(data.max())
        min_val = np.abs(data.min())
        return max_val - min_val

