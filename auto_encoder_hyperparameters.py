import keras_tuner as kt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
import datetime

# ------------------ DATA PREPROCESSING AND SETUP ------------------ #

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

# --------------------------- MODEL DEFINITION --------------------------- #

def build_autoencoder(hp):
    input_img = Input(shape=(256, 256, 3))  # Adjust the input shape as needed

    # Encoder
    x = Conv2D(
        hp.Int('encoder_filters_1', min_value=128, max_value=128, step=32),  # 32 filters for the first layer
        (3, 3),
        activation='relu',
        padding='same'
    )(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(
        hp.Int('encoder_filters_2', min_value=64, max_value=64, step=32),  # 64 filters for the second layer
        (3, 3),
        activation='relu',
        padding='same'
    )(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(
        hp.Int('encoder_filters_3', min_value=16, max_value=32, step=32),  # 128 filters for the third layer
        (3, 3),
        activation='relu',
        padding='same'
    )(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(
        hp.Int('decoder_filters_1', min_value=16, max_value=32, step=32),  # 128 filters for the first layer
        (3, 3),
        activation='relu',
        padding='same'
    )(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(
        hp.Int('decoder_filters_2', min_value=64, max_value=64, step=32),  # 64 filters for the second layer
        (3, 3),
        activation='relu',
        padding='same'
    )(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(
        hp.Int('decoder_filters_3', min_value=128, max_value=128, step=32),  # 32 filters for the third layer
        (3, 3),
        activation='relu',
        padding='same'
    )(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Define the autoencoder model
    autoencoder = Model(input_img, decoded)
    
    # Compile the model
    autoencoder.compile(
        optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='mean_squared_error'
    )
    return autoencoder


# Initialize the tuner
print("Initializing tuner...")
tuner = kt.RandomSearch(
    build_autoencoder,
    objective='val_loss',
    max_trials=10,  # Adjust the number of trials here
    executions_per_trial=1,
    directory='autoencoder_tuning',
    project_name='autoencoder_tuning'
)

# You will also need to include the EarlyStopping callback here
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Start the search
print("Starting hyperparameter search...")
tuner.search(
    train_generator,
    epochs=10,  # Adjust the number of epochs
    validation_data=validation_generator,
    callbacks=[early_stopping_callback]
)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]
print(f"""
The hyperparameter search is complete. The optimal hyperparameters are:
- Encoder layer 1 filters: {best_hps.get('encoder_filters_1')}
- Encoder layer 2 filters: {best_hps.get('encoder_filters_2')}
- Encoder layer 3 filters: {best_hps.get('encoder_filters_3')}
- Decoder layer 1 filters: {best_hps.get('decoder_filters_1')}
- Decoder layer 2 filters: {best_hps.get('decoder_filters_2')}
- Decoder layer 3 filters: {best_hps.get('decoder_filters_3')}
- Learning rate: {best_hps.get('learning_rate')}
- Batch size: {batch_size}
- Epochs: {10}
- Optimizer: Adam
""")


# save the best hyperparameters to a JSON file
best_hps_dict = best_hps.get_config()

with open('best_hyperparameters.json', 'w') as f:
    json.dump(best_hps_dict, f)

print("Saved best hyperparameters to file to best_hyperparameters.json")