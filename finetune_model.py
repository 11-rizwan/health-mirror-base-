# finetune_model.py
import tensorflow as tf
import os

print("TensorFlow Version:", tf.__version__)

# --- Configuration ---
BASE_MODEL_PATH = 'emotion_model.h5'
FINETUNED_MODEL_PATH = 'emotion_model_finetuned.h5'
DATA_DIR = 'finetune_data'
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 8 # Small batch size for a small dataset
EPOCHS = 40 # Train for a few epochs

# --- Load Your Custom Data ---
print("Loading custom dataset for fine-tuning...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

# --- Load the Pre-Trained Model ---
print(f"Loading base model from: {BASE_MODEL_PATH}")
model = tf.keras.models.load_model(BASE_MODEL_PATH)

# --- Fine-Tuning ---
# We use a very low learning rate to avoid destroying the original weights.
# This is the most important step in fine-tuning.
LOW_LEARNING_RATE = 0.0001

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LOW_LEARNING_RATE), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting fine-tuning...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# --- Save the New, Fine-Tuned Model ---
model.save(FINETUNED_MODEL_PATH)
print(f"\nFine-tuning complete. New model saved as '{FINETUNED_MODEL_PATH}'")