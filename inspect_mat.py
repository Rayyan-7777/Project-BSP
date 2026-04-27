"""
inspect_mat.py
==============
Inspects the structure of ECGData.mat and attempts to extract the ECG signal.
Run with: python inspect_mat.py
"""
import numpy as np
import scipy.io as sio

MAT_FILE = "ECGData.mat"   # <-- change to your filename if different

print(f"Loading: {MAT_FILE}")
mat = sio.loadmat(MAT_FILE, squeeze_me=True, struct_as_record=False)

print("\n=== Top-level keys ===")
for k, v in mat.items():
    if k.startswith('__'):
        continue
    dtype_str = str(v.dtype) if hasattr(v, 'dtype') else type(v).__name__
    shape_str = str(v.shape) if hasattr(v, 'shape') else 'scalar'
    print(f"  [{k}]  dtype={dtype_str}  shape={shape_str}")

    # If structured, show field names
    if hasattr(v, 'dtype') and v.dtype.names:
        print(f"       Fields: {v.dtype.names}")
        for field in v.dtype.names:
            try:
                sub = v[field]
                if hasattr(sub, 'squeeze'):
                    sub = sub.squeeze()
                sub_type = str(sub.dtype) if hasattr(sub, 'dtype') else type(sub).__name__
                sub_shape = str(sub.shape) if hasattr(sub, 'shape') else 'scalar'
                print(f"         .{field}  dtype={sub_type}  shape={sub_shape}")
                # If the subfield is numeric, show min/max
                if hasattr(sub, 'dtype') and np.issubdtype(sub.dtype, np.number):
                    flat = sub.flatten()
                    print(f"                 min={flat.min():.4f}  max={flat.max():.4f}  samples={flat.size:,}")
                # If sub is object array, go one level deeper
                elif hasattr(sub, 'dtype') and sub.dtype == object:
                    flat = sub.flatten()
                    if flat.size > 0:
                        elem = flat[0]
                        print(f"                 Cell[0] type={type(elem).__name__}", end="")
                        if hasattr(elem, 'shape'):
                            print(f"  shape={elem.shape}", end="")
                        print()
            except Exception as e:
                print(f"         .{field}  ERROR: {e}")

print("\n=== Attempting auto-extract with load_mat ===")
try:
    from utils import load_mat
    with open(MAT_FILE, 'rb') as f:
        sig, fs = load_mat(f)
    print(f"SUCCESS! Signal extracted:")
    print(f"  Shape  : {sig.shape}")
    print(f"  Fs     : {fs} Hz")
    print(f"  Min    : {sig.min():.4f}")
    print(f"  Max    : {sig.max():.4f}")
    print(f"  Samples: {sig.size:,}  ({sig.size/fs:.1f} seconds)")
except Exception as e:
    print(f"FAILED: {e}")
