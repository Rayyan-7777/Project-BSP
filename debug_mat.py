"""Debug script to inspect how scipy loadmat reads structured .mat files."""
import numpy as np
import scipy.io as sio
import tempfile, os

ecg_signal = np.sin(2 * np.pi * 1.2 * np.arange(3600) / 360.0).astype(np.float32)
struct_dtype = np.dtype([('Data', 'O'), ('Labels', 'O')])
struct_arr = np.zeros(1, dtype=struct_dtype)
struct_arr['Data'][0] = ecg_signal
struct_arr['Labels'][0] = np.array(['ECG'])
mat_dict = {'ECGData': struct_arr}

tmp = tempfile.NamedTemporaryFile(suffix='.mat', delete=False)
sio.savemat(tmp.name, mat_dict)
tmp.close()

for sq in (True, False):
    mat = sio.loadmat(tmp.name, squeeze_me=sq, struct_as_record=False)
    v = mat['ECGData']
    vd = str(v.dtype)
    vs = str(getattr(v, 'shape', None))
    print(f"squeeze_me={sq}: type={type(v).__name__}, dtype={vd}, shape={vs}")
    if hasattr(v, 'dtype') and v.dtype.names:
        print(f"  fields={v.dtype.names}")
        for field in v.dtype.names:
            s = v[field]
            sd = str(getattr(s, 'dtype', '?'))
            ss = str(getattr(s, 'shape', '?'))
            print(f"  [{field}]: type={type(s).__name__} dtype={sd} shape={ss}")
            if hasattr(s, 'flatten'):
                for i, e in enumerate(s.flatten()[:2]):
                    ed = str(getattr(e, 'dtype', '?'))
                    esh = str(getattr(e, 'shape', '?'))
                    print(f"    elem[{i}]: {type(e).__name__} shape={esh} dtype={ed}")

os.unlink(tmp.name)
print("\nDone.")
