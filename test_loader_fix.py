"""Test script to verify utils.load_mat handles mat_struct objects."""
import numpy as np
import scipy.io as sio
import tempfile, os
from utils import load_mat

ecg_signal = np.sin(2 * np.pi * 1.2 * np.arange(3600) / 360.0).astype(np.float32)
struct_dtype = np.dtype([('Data', 'O'), ('Labels', 'O')])
struct_arr = np.zeros(1, dtype=struct_dtype)
struct_arr['Data'][0] = ecg_signal
struct_arr['Labels'][0] = np.array(['ECG'])
mat_dict = {'ECGData': struct_arr}

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
    sio.savemat(tmp.name, mat_dict)
    mat_path = tmp.name

try:
    with open(mat_path, 'rb') as f:
        sig, fs = load_mat(f)
    print(f"SUCCESS! Extracted signal shape: {sig.shape}, fs: {fs}")
except Exception as e:
    print(f"ERROR: {e}")
finally:
    os.unlink(mat_path)
