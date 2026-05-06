import os
import pandas as pd
import tensorflow as tf
from model import build_model, hybrid_loss, dice_coefficient
from dataset import BraTSDataGenerator
from preprocess import split_data
import argparse

DATA_PATH = "/content/drive/MyDrive/BraTS_Project/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
REPO_PATH = "/content/drive/MyDrive/BraTS_Project/brats-brain-tumor"
BATCH_SIZE = 8
EPOCHS = 70

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="unet", help="unet, attention, or resunet")
args = parser.parse_args()

MODEL_TYPE = args.model

patient_ids = sorted([f for f in os.listdir(DATA_PATH) if f.startswith("BraTS20")])
train_ids, val_ids, test_ids = split_data(patient_ids)

train_gen = BraTSDataGenerator(train_ids, DATA_PATH, batch_size=BATCH_SIZE, shuffle=True)
val_gen = BraTSDataGenerator(val_ids, DATA_PATH, batch_size=BATCH_SIZE, shuffle=False)

model = build_model(model_type=MODEL_TYPE, input_shape=(128, 128, 4))

os.makedirs(os.path.join(REPO_PATH, "models"), exist_ok=True)
checkpoint_path = os.path.join(REPO_PATH, f"models/best_{MODEL_TYPE}_model.keras")
log_path = os.path.join(REPO_PATH, f"models/{MODEL_TYPE}_training_log.csv")

start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"Pretrained {MODEL_TYPE} is loading...")
    model = tf.keras.models.load_model(checkpoint_path, custom_objects={
        'hybrid_loss': hybrid_loss,
        'dice_coefficient': dice_coefficient
    })
    
    if os.path.exists(log_path):
        try:
            existing_logs = pd.read_csv(log_path)
            start_epoch = existing_logs['epoch'].max() + 1
            print(f"Training will continue from {start_epoch}. epoch")
        except Exception as e:
            print("Could not read log file. start_epoch will remain as default (0).")

initial_lr = 1e-4
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
    loss=hybrid_loss,
    metrics=[dice_coefficient, "accuracy"]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_dice_coefficient", mode="max", save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7, verbose=1),
    tf.keras.callbacks.CSVLogger(log_path, append=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_ids) // BATCH_SIZE, 
    validation_steps=len(val_ids) // BATCH_SIZE, 
    epochs=EPOCHS,
    initial_epoch = start_epoch,
    callbacks=callbacks
)

model.save(os.path.join(REPO_PATH, f"models/final_{MODEL_TYPE}_model.keras"))