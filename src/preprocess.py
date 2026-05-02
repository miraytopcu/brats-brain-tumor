import nibabel as nib
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import glob

def split_data(patient_ids, test_size=0.2, val_size=0.1):
    
    train_val_ids, test_ids = train_test_split(patient_ids, test_size=test_size, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size, random_state=42)
    
    return train_ids, val_ids, test_ids

def preprocess_nifti_slice(image_volume, mask_volume, slice_idx, target_size=(128, 128)):
    """
    Takes 3D volumes, extracts a slice, crops, resizes and normalizes it.
    """
    # 1. Extract Slice (Shape: 240, 240, 4) and (240, 240)
    image = image_volume[:, :, slice_idx, :]
    mask = mask_volume[:, :, slice_idx]

    # 2. Brain Crop (Removing black borders based on image intensity)
    brain_mask = np.max(image, axis=-1) > 0
    coords = np.where(brain_mask)
    
    if coords[0].size > 0:
        min_r, max_r = np.min(coords[0]), np.max(coords[0])
        min_c, max_c = np.min(coords[1]), np.max(coords[1])
        image = image[min_r:max_r, min_c:max_c, :]
        mask = mask[min_r:max_r, min_c:max_c]

    # 3. Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # 4. Normalization (Per Channel)
    image = image.astype(np.float32)
    for i in range(4):
        c_min, c_max = image[:,:,i].min(), image[:,:,i].max()
        if c_max > c_min:
            image[:,:,i] = (image[:,:,i] - c_min) / (c_max - c_min)
        else:
            image[:,:,i] = 0.0

    # 5. Mask Multi-Channel Encoding (Conversion for BraTS labels: 1, 2, 4)
    # Background, NCR(1), ED(2), ET(4) -> 4 Channels
    h, w = target_size
    final_mask = np.zeros((h, w, 3), dtype=np.float32)
    final_mask[:,:,0] = (mask == 1).astype(np.float32) # Necrotic and Non-Enhancing Tumor Core
    final_mask[:,:,1] = (mask == 2).astype(np.float32) # Peritumoral Edema
    final_mask[:,:,2] = (mask == 4).astype(np.float32) # GD-enhancing Tumor
    
    return image, final_mask

def load_patient_volumes(patient_path, patient_id):
    """
    Loads all 4 modalities and the mask for a single patient.
    Returns: (240, 240, 155, 4) Image volume and (240, 240, 155) Mask volume.
    """
    mods = ['flair', 't1', 't1ce', 't2']
    images = []
    
    try:
        for mod in mods:
            pattern = os.path.join(patient_path, f"*{mod}.nii")
            matching_files = glob.glob(pattern)
            
            if not matching_files:
                return None, None
            
            img = nib.load(matching_files[0]).get_fdata()
            images.append(img)
    
        # Stack to (240, 240, 155, 4)
        image_vol = np.stack(images, axis=-1)
        
        mask_pattern = os.path.join(patient_path, "*seg.nii")
        mask_matching = glob.glob(mask_pattern)
        mask_vol = nib.load(mask_matching[0]).get_fdata()
        
        return image_vol, mask_vol
    
    except Exception as e:
        print(f"Error: Mistake in {patient_id} folder: {e}")
        return None, None