import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import datetime

# Set Paths
dataset_dir = 'capsule/capsule/'
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# Preprocessing parameters
img_height, img_width = 256, 256 # adjust this to your input image size
batch_size = 64 # was 32 - use range of 16, 32, 64, 128, 256, 512, 1024

# Setup TensorBoard
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Initialise the TensorBoard callback object
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training Data Generators
# In this case only 'good' images for training are needed
train_datatgen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datatgen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='input', # 'input' because the output is the same as the input 
    subset='training', # specify this is training data
    color_mode='rgb',
    shuffle=True
)

validation_generator = train_datatgen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='input', # 'input' because the output is the same as the input
    subset='validation', # specify this is validation data
    color_mode='rgb',
    shuffle=True
)

# Test Data Generators (All images for testing)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None, # No labels for test data
    color_mode='rgb',
    shuffle=False
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

def build_autoencoder():
    input_img = Input(shape=(256, 256, 3))  # Input shape matches the preprocessed images

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Autoencoder model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder

# Create the autoencoder model
autoencoder = build_autoencoder()
autoencoder.summary()  # This will print the summary of the model

# Create a ModelCheckpoint callback that saves the model's weights
checkpoint_path = f'saved_models/model_checkpoints/model-{epoch:02d}-{val_loss:.2f}.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

# Initialise the EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=10,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Whether to restore model weights from the epoch with the best value of the monitored quantity
)

# Train the autoencoder
history = autoencoder.fit(
    train_generator,
    epochs=50,  # You can start with a lower number of epochs and increase as needed
    validation_data=validation_generator,
    callbacks=[tensorboard_callback, model_checkpoint_callback, early_stopping_callback]
)

# After training the model, save it to a file 
model_save_path = '/saved_models/model_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5'
autoencoder.save(model_save_path)

print(f'Model saved to: {model_save_path}')

# ------------------------- Evaluation -------------------------

# ------------------------- Model Load -------------------------
from tensorflow.keras.models import load_model

# Load the model form the file
autoencoder = load_model(model_save_path)

# Model is ready for use. 
