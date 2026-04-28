"""
processing.py
=============
Biomedical Signal Processing Engine for ECG-HRV Analysis Dashboard.

Modules
-------
1. ECG Preprocessing
   - Bandpass filter (0.5 – 40 Hz)  via Butterworth IIR filter
   - Baseline wander removal          via high-pass filter (0.5 Hz)

2. R-Peak Detection
   - Pan-Tompkins algorithm (custom NumPy implementation)
   - NeuroKit2 fallback for robustness

3. Time-Domain HRV
   - Mean RR, SDNN, RMSSD, pNN50, Mean HR

4. Frequency-Domain HRV (Welch's PSD)
   - VLF, LF, HF band powers
   - LF/HF ratio

5. Non-Linear HRV
   - Poincaré SD1, SD2
   - Sample Entropy (SampEn)
   - Approximate Entropy (ApEn)

References
----------
- Pan J, Tompkins WJ (1985). "A real-time QRS detection algorithm."
  IEEE Trans Biomed Eng. 32(3):230-6.
- Task Force (1996). "Heart rate variability: standards of measurement."
  Eur Heart J. 17, 354-381.
- Richman JS, Moorman JR (2000). "Physiological time-series analysis..."
  Am J Physiol Heart Circ Physiol. 278(6), H2039-49.

Author  : BSP Lab OEL
Version : 1.0
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.integrate import simpson
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# 1. ECG PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(ecg: np.ndarray, fs: float,
                    lowcut: float = 0.5, highcut: float = 40.0,
                    order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Removes high-frequency EMG noise and low-frequency motion artifacts
    while preserving the ECG morphology (P, QRS, T waves) in the
    clinically relevant 0.5–40 Hz band.

    Parameters
    ----------
    ecg     : np.ndarray  – Raw ECG signal
    fs      : float       – Sampling frequency (Hz)
    lowcut  : float       – Low cutoff frequency (Hz), default 0.5
    highcut : float       – High cutoff frequency (Hz), default 40.0
    order   : int         – Filter order (default 4)

    Returns
    -------
    filtered : np.ndarray – Bandpass-filtered ECG
    """
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.99)
    sos = sp_signal.butter(order, [low, high], btype='band', output='sos')
    return sp_signal.sosfiltfilt(sos, ecg)


def remove_baseline_wander(ecg: np.ndarray, fs: float,
                            cutoff: float = 0.5,
                            order: int = 3) -> np.ndarray:
    """
    Remove baseline wander using a high-pass Butterworth filter.

    Baseline wander (slow drift < 0.5 Hz) is caused by respiration and
    electrode movement.  A high-pass filter at 0.5 Hz removes this while
    keeping cardiac components intact.

    Parameters
    ----------
    ecg    : np.ndarray  – ECG signal (may contain baseline wander)
    fs     : float       – Sampling frequency (Hz)
    cutoff : float       – High-pass cutoff in Hz (default 0.5)
    order  : int         – Filter order (default 3)

    Returns
    -------
    corrected : np.ndarray  – Baseline-corrected ECG
    """
    nyq = 0.5 * fs
    norm_cutoff = min(max(cutoff / nyq, 0.001), 0.99)
    sos = sp_signal.butter(order, norm_cutoff, btype='high', output='sos')
    return sp_signal.sosfiltfilt(sos, ecg)


def preprocess_ecg(ecg_raw: np.ndarray, fs: float,
                   apply_bandpass: bool = True,
                   apply_baseline: bool = True,
                   lowcut: float = 0.5, highcut: float = 40.0) -> np.ndarray:
    """
    Full preprocessing pipeline for raw ECG.

    Steps (applied only when their toggle is True):
      1. Baseline wander removal (high-pass at 0.5 Hz)
      2. Bandpass filtering (0.5 – 40 Hz)

    Parameters
    ----------
    ecg_raw         : np.ndarray  – Raw ECG
    fs              : float       – Sampling frequency
    apply_bandpass  : bool        – Toggle bandpass filter
    apply_baseline  : bool        – Toggle baseline removal
    lowcut / highcut: float       – Bandpass cutoff frequencies

    Returns
    -------
    ecg_clean : np.ndarray  – Preprocessed ECG
    """
    ecg_clean = ecg_raw.copy().astype(float)

    if apply_baseline:
        ecg_clean = remove_baseline_wander(ecg_clean, fs)

    if apply_bandpass:
        ecg_clean = bandpass_filter(ecg_clean, fs, lowcut, highcut)

    return ecg_clean


# ─────────────────────────────────────────────────────────────────────────────
# 2. R-PEAK DETECTION  (Pan-Tompkins Algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def pan_tompkins_detect(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Pan-Tompkins QRS detection algorithm (NumPy implementation).

    Pipeline
    --------
    1. Bandpass filter      (5 – 15 Hz derivative-like bandpass)
    2. Differentiation      (5-point derivative)
    3. Squaring             (non-linear amplification)
    4. Moving-window integration (window ≈ 150 ms)
    5. Adaptive thresholding on signal & noise peaks
    6. Back-search to precise R-peak location

    Parameters
    ----------
    ecg : np.ndarray  – Pre-processed ECG signal (already bandpass filtered)
    fs  : float       – Sampling frequency (Hz)

    Returns
    -------
    r_peaks : np.ndarray  – Sample indices of detected R-peaks
    """
    # ── Step 1: bandpass 5-15 Hz (QRS energy band) ───────────────────────────
    def _bandpass(sig, low=5.0, high=15.0):
        nyq = 0.5 * fs
        sos = sp_signal.butter(2, [low / nyq, min(high / nyq, 0.99)], 'band', output='sos')
        return sp_signal.sosfiltfilt(sos, sig)

    bp = _bandpass(ecg)

    # ── Step 2: 5-point derivative ────────────────────────────────────────────
    # Derivative weights from Pan-Tompkins (1985)
    h = np.array([-1, -2, 0, 2, 1], dtype=float) * (fs / 8.0)
    diff = np.convolve(bp, h, mode='same')

    # ── Step 3: squaring ──────────────────────────────────────────────────────
    squared = diff ** 2

    # ── Step 4: moving-window integration (~150 ms) ───────────────────────────
    win = int(0.150 * fs)
    win = max(win, 1)
    kernel = np.ones(win) / win
    integrated = np.convolve(squared, kernel, mode='same')

    # ── Step 5: adaptive thresholding ────────────────────────────────────────
    # Refractory period: 200 ms minimum between R-peaks
    refractory = int(0.200 * fs)

    # Initial threshold: fraction of 8-second mean
    init_win = min(int(8 * fs), len(integrated))
    spki = 0.125 * np.max(integrated[:init_win])  # signal peak estimate
    npki = 0.125 * np.mean(integrated[:init_win])  # noise peak estimate
    threshold_i1 = npki + 0.25 * (spki - npki)

    r_peaks = []
    last_r = -refractory

    # Find candidate peaks from integrated signal
    min_distance = max(int(0.2 * fs), 1)
    candidate_peaks, props = sp_signal.find_peaks(
        integrated, distance=min_distance, height=threshold_i1 * 0.5
    )

    for peak in candidate_peaks:
        if (peak - last_r) < refractory:
            continue
        if integrated[peak] >= threshold_i1:
            # ── Step 6: back-search to exact R peak in original signal ────────
            search_start = max(0, peak - int(0.1 * fs))
            search_end = min(len(ecg), peak + int(0.1 * fs))
            local_r = search_start + np.argmax(np.abs(ecg[search_start:search_end]))

            r_peaks.append(local_r)
            last_r = peak

            # Update signal peak estimate
            spki = 0.125 * integrated[peak] + 0.875 * spki
        else:
            # Noise peak
            npki = 0.125 * integrated[peak] + 0.875 * npki

        threshold_i1 = npki + 0.25 * (spki - npki)

    return np.array(r_peaks, dtype=int)


def detect_r_peaks(ecg_signal: np.ndarray, fs: float,
                   method: str = 'neurokit') -> np.ndarray:
    """
    R-peak detection with multiple backend options.

    Parameters
    ----------
    ecg_signal: np.ndarray  – Signal to detect peaks on (ideally raw/unfiltered)
    fs        : float       – Sampling frequency (Hz)
    method    : str         – 'neurokit' | 'pantompkins' | 'scipy'

    Returns
    -------
    r_peaks : np.ndarray  – Sample indices of R-peaks
    """
    if method == 'pantompkins':
        return pan_tompkins_detect(ecg_signal, fs)

    elif method == 'neurokit':
        try:
            import neurokit2 as nk
            _, info = nk.ecg_peaks(ecg_signal, sampling_rate=int(fs), method='pantompkins1985')
            peaks = info['ECG_R_Peaks']
            if len(peaks) < 5:
                raise ValueError("Too few peaks detected by NeuroKit2")
            return np.array(peaks, dtype=int)
        except Exception:
            # Fallback to custom Pan-Tompkins
            return pan_tompkins_detect(ecg_signal, fs)

    elif method == 'scipy':
        # Simple scipy peak finding with height & distance constraints
        height_thresh = 0.3 * np.max(ecg_signal)
        dist = int(0.5 * fs)  # minimum 0.5 s between peaks (HR < 120)
        peaks, _ = sp_signal.find_peaks(ecg_signal, height=height_thresh,
                                         distance=dist)
        return peaks

    else:
        raise ValueError(f"Unknown method: {method}. Use 'neurokit', 'pantompkins', or 'scipy'.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. RR INTERVAL EXTRACTION AND VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def extract_rr_intervals(r_peaks: np.ndarray, fs: float,
                          min_rr: float = 0.30,
                          max_rr: float = 2.00) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RR intervals from R-peak sample indices.

    Applies physiological filtering to remove:
    - Ectopic beats (RR outside 300–2000 ms)
    - False detections

    Parameters
    ----------
    r_peaks : np.ndarray  – R-peak sample indices
    fs      : float       – Sampling frequency (Hz)
    min_rr  : float       – Minimum valid RR (seconds, default 0.30 s)
    max_rr  : float       – Maximum valid RR (seconds, default 2.00 s)

    Returns
    -------
    rr_intervals : np.ndarray  – Valid RR intervals in seconds
    rr_times     : np.ndarray  – Time of each RR midpoint (s) for tachogram
    """
    if len(r_peaks) < 2:
        return np.array([]), np.array([])

    rr = np.diff(r_peaks) / fs
    # Midpoint time for tachogram x-axis
    times = (r_peaks[:-1] + r_peaks[1:]) / (2.0 * fs)

    # Physiological filter
    valid = (rr >= min_rr) & (rr <= max_rr)
    return rr[valid], times[valid]


def detect_ectopic_beats(rr_intervals: np.ndarray, threshold: float = 0.20) -> np.ndarray:
    """
    Detect ectopic beats using a sliding window median filter.
    An RR interval is considered ectopic if it deviates by more than `threshold`
    (e.g., 20%) from the local median RR interval.

    Parameters
    ----------
    rr_intervals : np.ndarray  – Valid RR intervals in seconds
    threshold    : float       – Deviation threshold (default 20%)

    Returns
    -------
    ectopic_indices : np.ndarray – Indices of detected ectopic beats
    """
    if len(rr_intervals) < 5:
        return np.array([], dtype=int)

    # Use a median filter of size 11 (or smaller if signal is short)
    window_size = min(11, len(rr_intervals) | 1)  # Must be odd
    if window_size < 3:
        return np.array([], dtype=int)
        
    local_median = sp_signal.medfilt(rr_intervals, kernel_size=window_size)

    # Calculate relative deviation
    deviation = np.abs(rr_intervals - local_median) / local_median

    # Find indices where deviation exceeds threshold
    ectopic_indices = np.where(deviation > threshold)[0]
    return ectopic_indices


def interpolate_rr_intervals(rr_intervals: np.ndarray, ectopic_indices: np.ndarray, method: str = 'cubic') -> np.ndarray:
    """
    Replace ectopic beats using interpolation.

    Parameters
    ----------
    rr_intervals    : np.ndarray  – Original RR intervals (seconds)
    ectopic_indices : np.ndarray  – Indices of ectopic beats
    method          : str         – Interpolation method ('linear' or 'cubic')

    Returns
    -------
    nn_intervals : np.ndarray – Corrected normal-to-normal intervals
    """
    if len(ectopic_indices) == 0:
        return rr_intervals.copy()

    nn_intervals = rr_intervals.copy()
    valid_indices = np.setdiff1d(np.arange(len(rr_intervals)), ectopic_indices)

    if len(valid_indices) < 2:
        return nn_intervals  # Not enough valid points to interpolate

    valid_rr = rr_intervals[valid_indices]

    from scipy.interpolate import interp1d
    if method == 'cubic' and len(valid_indices) >= 4:
        f = interp1d(valid_indices, valid_rr, kind='cubic', bounds_error=False, fill_value="extrapolate")
    else:
        f = interp1d(valid_indices, valid_rr, kind='linear', bounds_error=False, fill_value="extrapolate")

    nn_intervals[ectopic_indices] = f(ectopic_indices)

    # Ensure no negative or absurdly large interpolated values
    mean_rr = np.mean(valid_rr)
    nn_intervals[nn_intervals <= 0] = mean_rr
    nn_intervals[nn_intervals > 2.0] = 2.0

    return nn_intervals



# ─────────────────────────────────────────────────────────────────────────────
# 4. TIME-DOMAIN HRV ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def time_domain_hrv(rr_intervals: np.ndarray) -> dict:
    """
    Compute standard time-domain HRV metrics (Task Force 1996).

    Metrics
    -------
    - mean_rr_ms   : Mean RR interval (ms)
    - sdnn_ms      : Standard deviation of NN intervals (ms)
    - rmssd_ms     : Root mean square of successive differences (ms)
    - pnn50        : Proportion of NN intervals > 50 ms (%)
    - nn50         : Count of NN pairs differing by > 50 ms
    - mean_hr      : Mean heart rate (bpm)
    - min_hr       : Minimum heart rate (bpm)
    - max_hr       : Maximum heart rate (bpm)
    - hr_std       : Std of instantaneous HR (bpm)

    Parameters
    ----------
    rr_intervals : np.ndarray  – Valid RR intervals in seconds

    Returns
    -------
    metrics : dict
    """
    if len(rr_intervals) < 4:
        return {}

    rr_ms = rr_intervals * 1000.0
    successive_diff = np.diff(rr_ms)

    # NN50 and pNN50
    nn50 = int(np.sum(np.abs(successive_diff) > 50.0))
    pnn50 = 100.0 * nn50 / len(successive_diff) if len(successive_diff) > 0 else 0.0

    # Heart rate
    hr = 60.0 / rr_intervals
    mean_hr = float(np.mean(hr))

    return {
        'mean_rr_ms'  : float(np.mean(rr_ms)),
        'sdnn_ms'     : float(np.std(rr_ms, ddof=1)),
        'rmssd_ms'    : float(np.sqrt(np.mean(successive_diff ** 2))),
        'pnn50'       : pnn50,
        'nn50'        : nn50,
        'mean_hr'     : mean_hr,
        'min_hr'      : float(np.min(hr)),
        'max_hr'      : float(np.max(hr)),
        'hr_std'      : float(np.std(hr, ddof=1)),
        'n_beats'     : len(rr_intervals),
        'total_time_s': float(np.sum(rr_intervals)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. FREQUENCY-DOMAIN HRV ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def frequency_domain_hrv(rr_intervals: np.ndarray,
                          method: str = 'welch',
                          fs_resample: float = 4.0) -> dict:
    """
    Power Spectral Density (PSD) analysis using Welch's method on
    uniformly re-sampled RR tachogram.

    Frequency Bands (Task Force 1996)
    ----------------------------------
    - VLF : 0.003 – 0.04  Hz  (very low frequency)
    - LF  : 0.04  – 0.15  Hz  (sympathetic + parasympathetic)
    - HF  : 0.15  – 0.40  Hz  (parasympathetic / vagal)

    Parameters
    ----------
    rr_intervals : np.ndarray  – RR intervals (seconds)
    method       : str         – 'welch' (only option currently)
    fs_resample  : float       – Resampling rate for tachogram (Hz, default 4)

    Returns
    -------
    freq_metrics : dict containing:
        - frequencies : np.ndarray  – PSD frequency axis
        - psd         : np.ndarray  – PSD (ms²/Hz)
        - vlf_power   : float       – VLF band power (ms²)
        - lf_power    : float       – LF band power (ms²)
        - hf_power    : float       – HF band power (ms²)
        - total_power : float       – Total power (ms²)
        - lf_nu       : float       – LF power in normalised units
        - hf_nu       : float       – HF power in normalised units
        - lf_hf_ratio : float       – LF/HF ratio
        - lf_pct      : float       – LF % of total
        - hf_pct      : float       – HF % of total
    """
    if len(rr_intervals) < 16:
        return {}

    # ── 1. Re-sample RR tachogram to uniform grid ─────────────────────────────
    rr_ms = rr_intervals * 1000.0
    cum_times = np.cumsum(rr_ms) / 1000.0  # cumulative time in seconds
    cum_times = np.insert(cum_times, 0, 0)[:-1]  # time of each R-peak

    t_uniform = np.arange(cum_times[0], cum_times[-1], 1.0 / fs_resample)
    rr_resampled = np.interp(t_uniform, cum_times, rr_ms)

    # ── 2. Welch's PSD ────────────────────────────────────────────────────────
    n_fft = min(2048, len(rr_resampled))
    nperseg = min(256, len(rr_resampled) // 2)

    freqs, psd = sp_signal.welch(
        rr_resampled, fs=fs_resample,
        window='hann', nperseg=nperseg,
        noverlap=nperseg // 2, nfft=n_fft,
        detrend='constant', scaling='density'
    )

    # ── 3. Band power integration ─────────────────────────────────────────────
    def _band_power(f, p, fmin, fmax):
        idx = (f >= fmin) & (f <= fmax)
        if idx.sum() < 2:
            return 0.0
        return float(simpson(p[idx], x=f[idx]))

    vlf = _band_power(freqs, psd, 0.003, 0.04)
    lf  = _band_power(freqs, psd, 0.04,  0.15)
    hf  = _band_power(freqs, psd, 0.15,  0.40)
    total = vlf + lf + hf

    # Normalised units (LF+HF denominator)
    lf_hf_sum = lf + hf if (lf + hf) > 0 else 1.0
    lf_nu = 100.0 * lf / lf_hf_sum
    hf_nu = 100.0 * hf / lf_hf_sum
    lf_hf = lf / hf if hf > 0 else float('nan')

    return {
        'frequencies' : freqs,
        'psd'         : psd,
        'vlf_power'   : vlf,
        'lf_power'    : lf,
        'hf_power'    : hf,
        'total_power' : total,
        'lf_nu'       : lf_nu,
        'hf_nu'       : hf_nu,
        'lf_hf_ratio' : lf_hf,
        'lf_pct'      : 100.0 * lf / total if total > 0 else 0,
        'hf_pct'      : 100.0 * hf / total if total > 0 else 0,
        'vlf_pct'     : 100.0 * vlf / total if total > 0 else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. NON-LINEAR HRV ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def poincare_analysis(rr_intervals: np.ndarray) -> dict:
    """
    Compute Poincaré plot parameters SD1, SD2, and the SD1/SD2 ratio.

    The Poincaré plot is a delay map where each RR_n is plotted against
    RR_{n+1}.  The ellipse fit yields:
    - SD1 : short-term (beat-to-beat) variability ‖ perpendicular axis
    - SD2 : long-term (overall) variability ‖ line-of-identity

    Parameters
    ----------
    rr_intervals : np.ndarray  – RR intervals in seconds

    Returns
    -------
    dict with keys: rr_n, rr_n1 (arrays), sd1_ms, sd2_ms, sd1_sd2_ratio
    """
    if len(rr_intervals) < 3:
        return {}

    rr_ms = rr_intervals * 1000.0
    rr_n  = rr_ms[:-1]   # RR_n
    rr_n1 = rr_ms[1:]    # RR_{n+1}

    # SD1 and SD2 from geometric decomposition
    diff = rr_n1 - rr_n   # successive differences
    sd1 = float(np.std(diff, ddof=1) / np.sqrt(2))
    sd2 = float(np.sqrt(2 * np.std(rr_ms, ddof=1) ** 2 - 0.5 * np.std(diff, ddof=1) ** 2))

    return {
        'rr_n'         : rr_n,
        'rr_n1'        : rr_n1,
        'sd1_ms'       : sd1,
        'sd2_ms'       : sd2,
        'sd1_sd2_ratio': sd1 / sd2 if sd2 > 0 else float('nan'),
        'ellipse_area' : np.pi * sd1 * sd2,
    }


def sample_entropy(rr_intervals: np.ndarray, m: int = 2, r_tol: float = 0.2) -> float:
    """
    Compute Sample Entropy (SampEn) of the RR interval series.

    SampEn measures signal regularity/complexity without self-matching
    count bias (unlike ApEn). Lower values → more regular/predictable.

    Parameters
    ----------
    rr_intervals : np.ndarray  – RR intervals (seconds or ms, both work)
    m            : int         – Template length (default 2)
    r_tol        : float       – Tolerance as fraction of std (default 0.2)

    Returns
    -------
    sampen : float  – Sample Entropy value (nats)
    """
    try:
        import antropy as ant
        return float(ant.sample_entropy(rr_intervals, order=m))
    except ImportError:
        pass

    # Manual fallback implementation
    N = len(rr_intervals)
    if N < 10:
        return float('nan')

    r = r_tol * np.std(rr_intervals, ddof=1)
    if r == 0:
        return float('nan')

    def _count_matches(x, m_len, threshold):
        count = 0
        for i in range(N - m_len):
            for j in range(i + 1, N - m_len):
                if np.max(np.abs(x[i:i+m_len] - x[j:j+m_len])) <= threshold:
                    count += 1
        return count

    A = _count_matches(rr_intervals, m + 1, r)
    B = _count_matches(rr_intervals, m, r)

    if B == 0:
        return float('nan')
    if A == 0:
        return float('inf')

    return float(-np.log(A / B))


def approximate_entropy(rr_intervals: np.ndarray, m: int = 2,
                         r_tol: float = 0.2) -> float:
    """
    Compute Approximate Entropy (ApEn) of the RR interval series.

    ApEn quantifies the regularity of a time series. Higher values
    indicate greater complexity and less predictability.

    Parameters
    ----------
    rr_intervals : np.ndarray  – RR intervals
    m            : int         – Template length
    r_tol        : float       – Tolerance as fraction of std

    Returns
    -------
    apen : float
    """
    try:
        import antropy as ant
        return float(ant.app_entropy(rr_intervals, order=m))
    except ImportError:
        pass

    N = len(rr_intervals)
    if N < 10:
        return float('nan')

    r = r_tol * np.std(rr_intervals, ddof=1)
    if r == 0:
        return float('nan')

    def _phi(m_len):
        count = np.zeros(N - m_len + 1)
        for i in range(N - m_len + 1):
            template = rr_intervals[i:i + m_len]
            for j in range(N - m_len + 1):
                if np.max(np.abs(rr_intervals[j:j+m_len] - template)) <= r:
                    count[i] += 1
        count = count / (N - m_len + 1)
        return np.sum(np.log(count)) / (N - m_len + 1)

    return float(_phi(m) - _phi(m + 1))


def nonlinear_hrv(rr_intervals: np.ndarray, entropy: str = 'sample') -> dict:
    """
    Combined non-linear HRV analysis: Poincaré + entropy.

    Parameters
    ----------
    rr_intervals : np.ndarray  – RR intervals (seconds)
    entropy      : str         – 'sample' | 'approximate'

    Returns
    -------
    dict combining Poincaré results and entropy value.
    """
    results = poincare_analysis(rr_intervals)

    if entropy == 'sample':
        en_val = sample_entropy(rr_intervals)
        results['entropy_type']  = 'Sample Entropy (SampEn)'
        results['entropy_value'] = en_val
    else:
        en_val = approximate_entropy(rr_intervals)
        results['entropy_type']  = 'Approximate Entropy (ApEn)'
        results['entropy_value'] = en_val

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. ORCHESTRATOR: Full HRV Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def full_hrv_analysis(ecg_raw: np.ndarray, fs: float,
                       apply_bandpass: bool = True,
                       apply_baseline: bool = True,
                       lowcut: float = 0.5, highcut: float = 40.0,
                       peak_method: str = 'neurokit',
                       entropy_type: str = 'sample') -> dict:
    """
    Run the complete ECG-HRV analysis pipeline and return all results.

    Parameters
    ----------
    ecg_raw       : np.ndarray  – Raw ECG signal
    fs            : float       – Sampling frequency (Hz)
    apply_bandpass: bool        – Enable bandpass filter
    apply_baseline: bool        – Enable baseline wander removal
    lowcut        : float       – Bandpass low cutoff
    highcut       : float       – Bandpass high cutoff
    peak_method   : str         – R-peak detection method
    entropy_type  : str         – 'sample' or 'approximate'

    Returns
    -------
    results : dict with keys:
        ecg_raw, ecg_clean, fs,
        r_peaks, rr_intervals, rr_times,
        time_metrics, freq_metrics, nonlinear_metrics
    """
    # ── Preprocessing ─────────────────────────────────────────────────────────
    ecg_clean = preprocess_ecg(ecg_raw, fs, apply_bandpass, apply_baseline,
                                lowcut, highcut)

    # ── R-Peak Detection ──────────────────────────────────────────────────────
    r_peaks = detect_r_peaks(ecg_raw, fs, method=peak_method)

    # ── RR Interval Extraction ────────────────────────────────────────────────
    rr_intervals, rr_times = extract_rr_intervals(r_peaks, fs)
    
    # ── Ectopic Beat Detection & Correction ───────────────────────────────────
    ectopic_indices = detect_ectopic_beats(rr_intervals)
    nn_intervals = interpolate_rr_intervals(rr_intervals, ectopic_indices, method='cubic')

    # ── HRV Analysis ──────────────────────────────────────────────────────────
    time_metrics     = time_domain_hrv(nn_intervals)
    freq_metrics     = frequency_domain_hrv(nn_intervals)
    nonlinear_metrics = nonlinear_hrv(nn_intervals, entropy=entropy_type)

    # Merge relevant metrics for interpretation layer
    combined = {}
    combined.update(time_metrics)
    combined.update({k: v for k, v in freq_metrics.items()
                     if not isinstance(v, np.ndarray)})
    combined.update({k: v for k, v in nonlinear_metrics.items()
                     if not isinstance(v, np.ndarray)})

    return {
        'ecg_raw'           : ecg_raw,
        'ecg_clean'         : ecg_clean,
        'fs'                : fs,
        'r_peaks'           : r_peaks,
        'rr_intervals'      : rr_intervals,
        'nn_intervals'      : nn_intervals,
        'ectopic_indices'   : ectopic_indices,
        'rr_times'          : rr_times,
        'time_metrics'      : time_metrics,
        'freq_metrics'      : freq_metrics,
        'nonlinear_metrics' : nonlinear_metrics,
        'combined_metrics'  : combined,
    }
