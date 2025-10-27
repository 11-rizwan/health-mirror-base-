# train_emotion_model.py (UPDATED FOR IMAGE FOLDERS)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

print("TensorFlow Version:", tf.__version__)

# --- Configuration ---
DATA_DIR = 'data/fer'
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64

# --- Load Data From Directories ---
# This function automatically finds the class folders (angry, happy, etc.)
# and loads the images.
train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale' # FER-2013 is grayscale
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'), # Use the same training directory for validation split
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

# Get the class names from the directory structure
class_names = train_dataset.class_names
print("Found classes:", class_names)

# --- Configure for Performance ---
# Use caching and prefetching to optimize data loading pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- Data Augmentation as Layers ---
# This is a more modern way to do data augmentation inside the model
data_augmentation = Sequential(
  [
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
  ]
)

# --- Build the CNN Model ---
model = Sequential([
    # Input layer: Rescale pixel values from [0, 255] to [0, 1]
    Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    
    # Augmentation
    data_augmentation,
    
    # Convolutional blocks (same architecture as before)
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    # Output layer: Number of neurons must match number of classes
    Dense(len(class_names), activation='softmax') 
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', # Use this loss for integer labels
              metrics=['accuracy'])

model.summary()

# --- Callbacks and Training ---
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('emotion_model.h5', monitor='val_accuracy', save_best_only=True)

epochs = 50
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint]
)

print("\nTraining complete. Best model saved as 'emotion_model.h5'")

# --- Optional: Evaluate on the test set ---
test_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'test'),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("\nEvaluating on test data...")
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")