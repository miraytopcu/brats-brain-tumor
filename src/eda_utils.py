import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np

def load_nifti_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    
    img = nib.load(file_path)
    data = img.get_fdata()
    print(f"--- Loaded: {os.path.basename(file_path)} ---")
    print(f"Shape: {data.shape} | Dtype: {data.dtype}")
    return data

def show_nifti_slice(data, slice_idx=None, title="NIfTI Slice"):
    if data is None: return
    
    if slice_idx is None:
        slice_idx = data.shape[2] // 2
        
    plt.figure(figsize=(6, 6))
    plt.imshow(data[:, :, slice_idx], cmap='gray')
    plt.title(f"{title} (Slice: {slice_idx})")
    plt.axis('off')
    plt.show()