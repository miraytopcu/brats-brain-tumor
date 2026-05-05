import os
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

checkpoint_path = os.path.join(REPO_PATH, f"models/best_{MODEL_TYPE}_model.keras")
start_epoch = 0

if os.path.exists(checkpoint_path):
    print(f"{checkpoint_path} is loading...")
    model.load_weights(checkpoint_path)
    
    if MODEL_TYPE == "unet":
        start_epoch = 50

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=hybrid_loss,
    metrics=[dice_coefficient, "accuracy"]
)

os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_dice_coefficient", mode="max", save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=50, 
    validation_steps=25, 
    epochs=EPOCHS,
    initial_epoch = start_epoch,
    callbacks=callbacks
)

model.save(os.path.join(REPO_PATH, f"models/final_{MODEL_TYPE}_model.keras"))