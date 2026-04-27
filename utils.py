"""
utils.py
========
Utility functions for the ECG-HRV Analysis Dashboard.

Handles:
  - Loading ECG data from .mat, .csv, and .dat (WFDB) files
  - Generating realistic synthetic ECG data for demonstration
  - Helper formatters for UI display

Author  : BSP Lab OEL
Version : 1.1
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# File Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _extract_numeric(val) -> np.ndarray | None:
    """
    Recursively extract a 1-D numeric ECG array from any MATLAB value type.

    Handles:
      - Plain numeric ndarray  (float/int)
      - Structured ndarray     (MATLAB struct: dtype with named fields)
      - scipy mat_struct       (produced by squeeze_me=True, struct_as_record=False)
      - Object ndarray         (MATLAB cell array)
      - Python lists/scalars

    Returns None if no valid signal can be extracted.
    """
    if val is None:
        return None

    # ── scipy mat_struct (squeeze_me=True + struct_as_record=False) ───────────
    # These are plain Python objects; field access is via __dict__
    val_type = type(val).__name__
    if val_type == 'mat_struct':
        ecg_priority = ['Data', 'data', 'signal', 'Signal', 'ECG', 'ecg',
                        'val', 'Val', 'samples', 'Samples', 'x', 'y']
        field_dict = {k: v for k, v in val.__dict__.items() if not k.startswith('_')}
        ordered = [f for f in ecg_priority if f in field_dict]
        ordered += [f for f in field_dict if f not in ordered]
        for field in ordered:
            try:
                result = _extract_numeric(field_dict[field])
                if result is not None and result.size > 100:
                    return result
            except Exception:
                continue
        return None

    # ── Structured ndarray: dtype has named fields ────────────────────────────
    if isinstance(val, np.ndarray) and val.dtype.names:
        ecg_priority = ['Data', 'data', 'signal', 'Signal', 'ECG', 'ecg',
                        'val', 'Val', 'samples', 'Samples', 'x', 'y']
        ordered = [f for f in ecg_priority if f in val.dtype.names]
        ordered += [f for f in val.dtype.names if f not in ordered]
        for field in ordered:
            try:
                sub = val[field]
                if hasattr(sub, 'squeeze'):
                    sub = sub.squeeze()
                result = _extract_numeric(sub)
                if result is not None and result.size > 100:
                    return result
            except Exception:
                continue
        return None

    # ── Object ndarray: MATLAB cell array ─────────────────────────────────────
    if isinstance(val, np.ndarray) and val.dtype == object:
        for elem in val.flatten():
            result = _extract_numeric(elem)
            if result is not None and result.size > 100:
                return result
        return None

    # ── Plain numeric ndarray ─────────────────────────────────────────────────
    if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
        arr = val.squeeze()
        if arr.ndim == 0:
            return None
        if arr.ndim == 1 and arr.size > 100:
            return arr.astype(float)
        if arr.ndim == 2 and max(arr.shape) > 100:
            if arr.shape[0] < arr.shape[1]:    # (channels, samples)
                return arr[0].astype(float)
            else:                              # (samples, channels)
                return arr[:, 0].astype(float)
        return None

    # ── Python list / scalar ──────────────────────────────────────────────────
    try:
        arr = np.array(val, dtype=float).squeeze()
        if arr.ndim == 1 and arr.size > 100:
            return arr
    except Exception:
        pass

    return None


def load_mat(file_obj) -> tuple[np.ndarray, float]:
    """
    Load ECG signal from a MATLAB .mat file.

    Fully supports ALL common PhysioNet and custom .mat formats:
      - Simple numeric arrays  (keys: val, ecg, signal, data, ECG …)
      - MATLAB struct arrays   (dtype with named fields like 'Data', 'Labels')
      - Nested cell arrays     (object dtype containing numeric arrays)
      - Multi-channel arrays   (first channel selected automatically)

    Parameters
    ----------
    file_obj : file-like object  – Streamlit UploadedFile or path-like

    Returns
    -------
    signal : np.ndarray  – 1-D ECG samples (mV or ADC units)
    fs     : float       – Sampling frequency in Hz (default 360 Hz)
    """
    # Reset stream position for repeated loads
    if hasattr(file_obj, 'seek'):
        file_obj.seek(0)

    # Try with squeeze_me first (cleaner output for most files)
    mat = None
    for squeeze in (True, False):
        try:
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
            mat = sio.loadmat(file_obj, squeeze_me=squeeze, struct_as_record=False)
            break
        except Exception as e:
            last_err = e
    if mat is None:
        raise ValueError(f"Could not read .mat file: {last_err}")

    fs = 360.0  # MIT-BIH / PhysioNet default

    # ── Priority key names ────────────────────────────────────────────────────
    priority_keys = [
        'val', 'ecg', 'ECG', 'signal', 'Signal', 'data', 'Data',
        'ECGData', 'ecgdata', 'MLII', 'mlii', 'V5', 'v5',
        'y', 'x', 'samples', 'recording', 'lead', 'ann'
    ]

    # Try priority keys first
    for key in priority_keys:
        if key in mat:
            result = _extract_numeric(mat[key])
            if result is not None and result.size > 100:
                return result, fs

    # Scan all remaining non-header keys
    for key, val in mat.items():
        if key.startswith('__'):
            continue
        if key in priority_keys:
            continue  # already tried above
        result = _extract_numeric(val)
        if result is not None and result.size > 100:
            return result, fs

    available = [k for k in mat if not k.startswith('__')]
    raise ValueError(
        f"No valid numeric ECG signal found in .mat file.\n"
        f"Available keys: {available}\n"
        "Tip: In MATLAB, save your ECG vector as: save('file.mat', 'ecg')"
    )


def load_csv(file_obj) -> tuple[np.ndarray, float]:
    """
    Load ECG signal from a CSV file.

    Searches for columns named 'ecg', 'ECG', 'signal', 'lead_II',
    'MLII', 'V5', 'amplitude', 'value'.  Falls back to the first
    numeric column if none found.

    Parameters
    ----------
    file_obj : file-like object

    Returns
    -------
    signal : np.ndarray
    fs     : float  (default 360 Hz for PhysioNet)
    """
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        raise ValueError(f"Could not read .csv file: {e}")

    fs = 360.0
    priority_cols = ['ecg', 'ECG', 'signal', 'lead_II', 'MLII', 'V5',
                     'amplitude', 'value', 'data', 'sample']

    for col in priority_cols:
        if col in df.columns:
            return df[col].dropna().to_numpy(dtype=float), fs

    # Fallback: first purely numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric column found in CSV file.")

    # If a 'time' or 'sample' column exists compute fs from it
    time_cols = [c for c in df.columns if 'time' in c.lower() or 'sample' in c.lower()]
    if time_cols:
        t = df[time_cols[0]].dropna().to_numpy(dtype=float)
        if t.size > 1:
            dt = np.median(np.diff(t))
            if dt > 0:
                fs = round(1.0 / dt)

    signal = df[numeric_cols[0]].dropna().to_numpy(dtype=float)
    return signal, fs


def load_dat(file_obj, record_name: str = None) -> tuple[np.ndarray, float]:
    """
    Load ECG signal from a WFDB .dat file using the wfdb library.

    Parameters
    ----------
    file_obj    : file-like object (the .dat upload)
    record_name : str, optional (stem name without extension)

    Returns
    -------
    signal : np.ndarray
    fs     : float
    """
    try:
        import wfdb
        import tempfile, os, shutil

        # WFDB needs a directory + record name – write to temp dir
        tmp_dir = tempfile.mkdtemp()
        try:
            stem = record_name or "record"
            dat_path = os.path.join(tmp_dir, stem + ".dat")
            with open(dat_path, 'wb') as f:
                f.write(file_obj.read())

            # Attempt to read without header; wfdb will infer format 212
            record = wfdb.rdrecord(os.path.join(tmp_dir, stem))
            signal = record.p_signal[:, 0].astype(float)
            fs = float(record.fs)
        finally:
            shutil.rmtree(tmp_dir)

        return signal, fs

    except Exception as e:
        # NEW ADDITION: Fallback for generic (non-WFDB) .dat files
        try:
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
            content = file_obj.read()
            try:
                # Try raw text parsing
                text = content.decode('utf-8')
                separator = ',' if ',' in text else None
                arr = np.fromstring(text, sep=separator if separator else ' ', dtype=float)
                if arr.size == 0:
                    arr = np.fromstring(text, sep='\n', dtype=float)
            except UnicodeDecodeError:
                # Try raw binary float64 then float32
                arr = np.frombuffer(content, dtype=np.float64)
                if arr.size < 100 or np.isnan(arr).all():
                    arr = np.frombuffer(content, dtype=np.float32)
            
            if arr.ndim >= 1 and arr.size > 100:
                if arr.ndim == 2:
                    arr = arr[0] if arr.shape[0] < arr.shape[1] else arr[:, 0]
                return arr.astype(float), 360.0 # Default fallback freq
        except Exception:
            pass

        raise ValueError(
            f"Could not read .dat file: {e}\n"
            "If using PhysioNet, ensure matching .hea file is uploaded. "
            "Raw generic .dat files must be flat numeric text or binary floats."
        )


def load_ecg_file(file_obj) -> tuple[np.ndarray, float]:
    """
    Dispatch loader based on file extension.

    Parameters
    ----------
    file_obj : Streamlit UploadedFile

    Returns
    -------
    signal : np.ndarray  – raw ECG samples
    fs     : float       – sampling frequency (Hz)
    """
    name = file_obj.name.lower()
    if name.endswith('.mat'):
        return load_mat(file_obj)
    elif name.endswith('.csv'):
        return load_csv(file_obj)
    elif name.endswith('.dat'):
        return load_dat(file_obj)
    else:
        raise ValueError(f"Unsupported file extension: {Path(name).suffix}")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic / Demo Data
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_ecg(duration: float = 60.0, fs: float = 360.0,
                            heart_rate: float = 72.0,
                            hrv_noise_std: float = 0.04,
                            noise_amp: float = 0.05) -> tuple[np.ndarray, float]:
    """
    Generate a physiologically realistic synthetic ECG signal.

    Uses a simple PQRST morphology template convolved with a
    Dirac comb whose inter-beat intervals follow a realistic HRV
    distribution (normally-distributed around the mean RR interval
    with slight LF oscillation to mimic respiratory sinus arrhythmia).

    Parameters
    ----------
    duration      : float  – Signal length in seconds (default 60 s)
    fs            : float  – Sampling frequency in Hz (default 360 Hz)
    heart_rate    : float  – Mean heart rate in bpm (default 72 bpm)
    hrv_noise_std : float  – Standard deviation of RR variability (seconds)
    noise_amp     : float  – Gaussian noise amplitude added to ECG

    Returns
    -------
    ecg    : np.ndarray  – ECG signal (normalised to +-1 range)
    fs     : float
    """
    rng = np.random.default_rng(42)
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    # Build PQRST template (one beat)
    beat_len = int(0.8 * fs)   # 800 ms window
    bt = np.linspace(0, 0.8, beat_len)

    def _gauss(t, mu, sigma, amp):
        return amp * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))

    template = (
        _gauss(bt, 0.10, 0.015,  0.10)  +   # P wave
        _gauss(bt, 0.20, 0.005, -0.10)  +   # Q wave
        _gauss(bt, 0.22, 0.010,  1.00)  +   # R wave
        _gauss(bt, 0.24, 0.005, -0.15)  +   # S wave
        _gauss(bt, 0.38, 0.040,  0.25)      # T wave
    )

    # Build RR intervals with HRV (LF Mayer wave + Gaussian noise)
    mean_rr = 60.0 / heart_rate
    beat_times = []
    t_beat = 0.0
    lf_amp  = hrv_noise_std * 0.6
    lf_freq = 0.1

    while t_beat < duration:
        lf_mod = lf_amp * np.sin(2 * np.pi * lf_freq * t_beat)
        rr = mean_rr + lf_mod + rng.normal(0, hrv_noise_std * 0.4)
        rr = np.clip(rr, mean_rr * 0.6, mean_rr * 1.4)
        beat_times.append(t_beat)
        t_beat += rr

    # Stamp beats onto signal
    ecg = np.zeros(n_samples)
    for bt_time in beat_times:
        start_idx = int(bt_time * fs)
        end_idx = min(start_idx + beat_len, n_samples)
        seg_len = end_idx - start_idx
        ecg[start_idx:end_idx] += template[:seg_len]

    # Add Gaussian noise + 0.2 Hz baseline wander
    ecg += rng.normal(0, noise_amp, n_samples)
    ecg += 0.10 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 2 * np.pi))

    # Normalise to [-1, 1]
    ecg = ecg / np.max(np.abs(ecg))
    return ecg, fs


# ─────────────────────────────────────────────────────────────────────────────
# Formatting Helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_metric(value: float, unit: str = "", decimals: int = 3) -> str:
    """Return a neatly formatted string for a HRV metric."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f} {unit}".strip()


def ms(seconds: float) -> float:
    """Convert seconds to milliseconds."""
    return seconds * 1000.0


def interpret_hrv(metrics: dict) -> dict:
    """
    Automatically generate clinical interpretation of HRV metrics.

    Parameters
    ----------
    metrics : dict  – Must contain keys from the HRV analysis result.

    Returns
    -------
    interpretations : dict  – {category: message}
    """
    interp = {}

    # Mean HR / RR
    mean_hr = metrics.get('mean_hr', None)
    if mean_hr:
        if mean_hr < 60:
            interp['HR Status'] = (
                "Bradycardia detected (HR < 60 bpm). "
                "May indicate high parasympathetic tone or athletic adaptation."
            )
        elif mean_hr > 100:
            interp['HR Status'] = (
                "Tachycardia detected (HR > 100 bpm). "
                "Could reflect stress, anxiety, or sympathetic dominance."
            )
        else:
            interp['HR Status'] = (
                f"Normal sinus rhythm (HR = {mean_hr:.1f} bpm)."
            )

    # RMSSD (parasympathetic marker)
    rmssd = metrics.get('rmssd_ms', None)
    if rmssd is not None:
        if rmssd > 50:
            interp['Parasympathetic Activity (RMSSD)'] = (
                f"High RMSSD ({rmssd:.1f} ms) - Strong parasympathetic (vagal) activity. "
                "Associated with good cardiovascular health and stress resilience."
            )
        elif rmssd < 20:
            interp['Parasympathetic Activity (RMSSD)'] = (
                f"Low RMSSD ({rmssd:.1f} ms) - Reduced vagal tone. "
                "May indicate fatigue, chronic stress, or cardiac risk."
            )
        else:
            interp['Parasympathetic Activity (RMSSD)'] = (
                f"Moderate RMSSD ({rmssd:.1f} ms) - Normal parasympathetic tone."
            )

    # LF/HF Ratio (sympathovagal balance)
    lf_hf = metrics.get('lf_hf_ratio', None)
    if lf_hf is not None:
        if lf_hf > 2.0:
            interp['Sympathovagal Balance (LF/HF)'] = (
                f"High LF/HF ratio ({lf_hf:.2f}) - Sympathetic dominance. "
                "Seen in stress, anxiety, and cardiovascular disease."
            )
        elif lf_hf < 0.5:
            interp['Sympathovagal Balance (LF/HF)'] = (
                f"Low LF/HF ratio ({lf_hf:.2f}) - Parasympathetic dominance. "
                "Typical in relaxed states or trained athletes."
            )
        else:
            interp['Sympathovagal Balance (LF/HF)'] = (
                f"Balanced LF/HF ratio ({lf_hf:.2f}) - Normal autonomic balance."
            )

    # HF Power (respiratory sinus arrhythmia)
    hf_pct = metrics.get('hf_pct', None)
    if hf_pct is not None:
        if hf_pct > 50:
            interp['High-Frequency Power (HF)'] = (
                f"Elevated HF power ({hf_pct:.1f}% of total) - "
                "Strong respiratory sinus arrhythmia; good vagal modulation."
            )
        else:
            interp['High-Frequency Power (HF)'] = (
                f"HF power = {hf_pct:.1f}% of total PSD."
            )

    # SDNN (overall HRV)
    sdnn = metrics.get('sdnn_ms', None)
    if sdnn is not None:
        if sdnn > 100:
            interp['Overall HRV (SDNN)'] = (
                f"High SDNN ({sdnn:.1f} ms) - Excellent overall HRV. "
                "Reflects robust autonomic nervous system activity."
            )
        elif sdnn < 50:
            interp['Overall HRV (SDNN)'] = (
                f"Low SDNN ({sdnn:.1f} ms) - Reduced overall HRV. "
                "Clinically associated with increased cardiac mortality risk."
            )
        else:
            interp['Overall HRV (SDNN)'] = (
                f"SDNN = {sdnn:.1f} ms - Within normal range."
            )

    # Poincare SD1/SD2
    sd1 = metrics.get('sd1_ms', None)
    sd2 = metrics.get('sd2_ms', None)
    if sd1 is not None and sd2 is not None and sd2 > 0:
        ratio = sd1 / sd2
        interp['Poincare Complexity (SD1/SD2)'] = (
            f"SD1={sd1:.1f} ms, SD2={sd2:.1f} ms, SD1/SD2={ratio:.3f}. "
            + ("High short-term variability relative to long-term."
               if ratio > 0.5 else
               "Long-term variability dominant - common in normal subjects.")
        )

    # NEW ADDITION: Combined Condition Insights
    if sdnn is not None and lf_hf is not None:
        if sdnn < 50 and lf_hf > 2.0:
            interp['Combined Clinical Insight'] = (
                "🔴 High stress / low recovery state: Both overall HRV (SDNN) is low "
                "and sympathovagal balance is sympathetically dominant (LF/HF)."
            )
        elif sdnn > 100 and lf_hf < 1.0:
            interp['Combined Clinical Insight'] = (
                "🟢 Excellent recovery / athletic state: High overall HRV (SDNN) "
                "mixed with parasympathetic dominance."
            )

    return interp
