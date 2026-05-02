import numpy as np
import tensorflow as tf
import os
from preprocess import load_patient_volumes, preprocess_nifti_slice

class BraTSDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, patient_ids, base_path, batch_size=8, dim=(128, 128), n_channels=4, shuffle=True):
        self.patient_ids = patient_ids
        self.base_path = base_path
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.patient_ids) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.patient_ids[index * self.batch_size:(index + 1) * self.batch_size]

        X = []
        y = []

        for p_id in batch_ids:
            patient_path = os.path.join(self.base_path, p_id)
            
            img_vol, msk_vol = load_patient_volumes(patient_path, p_id)
            
            if img_vol is not None:
                slice_idx = np.random.randint(30, 120) 
            
                img_slice, msk_slice = preprocess_nifti_slice(img_vol, msk_vol, slice_idx, target_size=self.dim)

                # Augmentation
                if self.shuffle:
                    if np.random.rand() > 0.5:
                        img_slice = np.flip(img_slice, axis=1)
                        msk_slice = np.flip(msk_slice, axis=1)

                    if np.random.rand() > 0.5:
                        img_slice = np.flip(img_slice, axis=0)
                        msk_slice = np.flip(msk_slice, axis=0)
            
                X.append(img_slice)
                y.append(msk_slice)
                
        while len(X) < self.batch_size and len(X) > 0:
            X.append(X[-1])
            y.append(y[-1])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.patient_ids)