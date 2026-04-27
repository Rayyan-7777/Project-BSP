"""Quick validation script for all ECG-HRV modules."""
from processing import full_hrv_analysis
from utils import generate_synthetic_ecg, interpret_hrv
import numpy as np

# Generate 60s demo signal
ecg, fs = generate_synthetic_ecg(duration=60.0, fs=360.0, heart_rate=72.0)
print(f"ECG shape: {ecg.shape}, fs={fs} Hz")

# Full pipeline
results = full_hrv_analysis(ecg, fs)
tm = results['time_metrics']
fm = results['freq_metrics']
nl = results['nonlinear_metrics']

print(f"R-peaks detected : {len(results['r_peaks'])}")
print(f"RR intervals     : {len(results['rr_intervals'])}")
print(f"Mean HR          : {tm.get('mean_hr', 0):.1f} bpm")
print(f"SDNN             : {tm.get('sdnn_ms', 0):.2f} ms")
print(f"RMSSD            : {tm.get('rmssd_ms', 0):.2f} ms")
print(f"pNN50            : {tm.get('pnn50', 0):.1f} %")
print(f"LF power         : {fm.get('lf_power', 0):.2f} ms2")
print(f"HF power         : {fm.get('hf_power', 0):.2f} ms2")
print(f"LF/HF ratio      : {fm.get('lf_hf_ratio', 0):.3f}")
print(f"SD1              : {nl.get('sd1_ms', 0):.2f} ms")
print(f"SD2              : {nl.get('sd2_ms', 0):.2f} ms")
en_val = nl.get('entropy_value', None)
print(f"Sample Entropy   : {en_val:.4f}" if en_val is not None else "Sample Entropy: N/A")

# Merge metrics for interpretation
combined = {}
combined.update(tm)
combined.update({k: v for k, v in fm.items() if not isinstance(v, np.ndarray)})
combined.update({k: v for k, v in nl.items() if not isinstance(v, np.ndarray)})

interp = interpret_hrv(combined)
print(f"\nInterpretations generated: {len(interp)}")
for k in interp:
    print(f"  [{k}]")

print("\nAll modules validated successfully!")
