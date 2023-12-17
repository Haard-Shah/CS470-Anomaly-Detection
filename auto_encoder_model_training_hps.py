import os
import json
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from auto_encoder_hyperparameters import build_autoencoder
import datetime
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

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

TRAIN_MODEL = False

if TRAIN_MODEL:
    # Create a ModelCheckpoint callback that saves the model's weights
    num_epochs = 100  # Adjust based on your needs
    checkpoint_path = f'saved_models/model_checkpoints/model-{num_epochs:02d}-.h5'
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
        patience=15,          # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Whether to restore model weights from the epoch with the best value of the monitored quantity
    )

    # Load the hyperparameters from the JSON file
    with open('best_hyperparameters.json', 'r') as f:
        best_hps_dict = json.load(f)

    # print the hyperparameters
    print("Best Hyperparameters loaded from best_hyperparameters.json file:")
    print(best_hps_dict)

    # Convert the dictionary back to a HyperParameters object
    best_hps = HyperParameters.from_config(best_hps_dict)

    # Now you can use best_hps to rebuild the model with the best hyperparameters
    best_autoencoder = build_autoencoder(best_hps)

    print("Model Summary:")
    best_autoencoder.summary()

    print("\n\nModel Ready for Training.")

    # Continue with model training as before
    best_history = best_autoencoder.fit(
        train_generator,
        epochs=num_epochs,  # Adjust based on your needs
        validation_data=validation_generator,
        callbacks=[tensorboard_callback, early_stopping_callback, model_checkpoint_callback]
    )

    # Save the trained model
    best_autoencoder.save('saved_models/final_best_autoencoder.h5')

    # Save the training history to a JSON file
    with open('best_history.json', 'w') as f:
        json.dump(best_history.history, f)

    print("[SAVE] Model saved to: saved_models/final_best_autoencoder.h5")

else:
    # Load the trained model
    best_autoencoder = load_model('saved_models/final_best_autoencoder.h5')

    print("[LOAD] Model loaded from: saved_models/final_best_autoencoder.h5")
    print("Model Summary:")
    best_autoencoder.summary()

    # Load the training history from the JSON file
    with open('best_history.json', 'r') as f:
        best_history = json.load(f)

    # Print the training history
    print("Training History:")
    print(best_history)

    # Plot the training history
    plt.figure()
    plt.plot(best_history['loss'], label='Training Loss')
    plt.plot(best_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # save the plot to a PNG file
    plt.savefig('training_history.png')
    print(f'[SAVE] Training history plot saved to: training_history.png')

# --------------------------- MODEL PREDICTION --------------------------- #
# Assuming test_generator is your test dataset generator
# test_images, _ = next(test_generator)  # Get a batch of test images
test_images = next(test_generator)  # Get a batch of test images
print(test_images.shape)
print(test_images[0].shape)
print(test_images[0])

reconstructed_images = best_autoencoder.predict(test_images)

# Calculate reconstruction error
reconstruction_error = np.mean(np.square(test_images - reconstructed_images), axis=(1, 2, 3))

# Plot the reconstruction error
plt.hist(reconstruction_error, bins=50)
plt.xlabel('Reconstruction Error')
plt.ylabel('# of Images')
plt.show()

# save the plot to a PNG file
plt.savefig('reconstruction_error.png')
print(f'[SAVE] Reconstruction error plot saved to: reconstruction_error.png')

# save the reconstruction error to a CSV file
np.savetxt('reconstruction_error.csv', reconstruction_error, delimiter=',')
print(f'[SAVE] Reconstruction error saved to: reconstruction_error.csv')

# --------------------------- MODEL EVALUATION --------------------------- #
from sklearn.metrics import roc_auc_score, roc_curve
import cv2
import numpy as np
import os

# Function to load ground truth masks
def load_ground_truth_masks(ground_truth_dir, test_images_filenames, img_height, img_width):
    ground_truth_masks = []
    for filename in test_images_filenames:
        # Construct the path to the ground truth mask
        mask_path = os.path.join(ground_truth_dir, filename)
        # Load the mask image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Resize the mask to match the model input size
        mask = cv2.resize(mask, (img_height, img_width))
        # Binarize the mask: pixels with a value will be set to 1, the rest to 0
        _, binary_mask = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)
        ground_truth_masks.append(binary_mask)

    # Convert the list to a numpy array and return
    return np.stack(ground_truth_masks)

# Define the path to your ground truth directory
ground_truth_dir = 'capsule/capsule/ground_truth'

# Assume test_images_filenames is a list of filenames of the test images

# iterate through each directory and get the list of filenames in each directory in test_dir
test_images_filenames = []
for root, dirs, files in os.walk(test_dir):
    for name in files:
        test_images_filenames.append(os.join(root+"/"+dirs, name))

print(test_images_filenames)

# Define the image size
img_height, img_width = 256, 256  # The size to which images are resized

# Load the ground truth masks
ground_truth_masks = load_ground_truth_masks(ground_truth_dir, test_images_filenames, img_height, img_width)


# Flatten the images and ground truth masks to 1D arrays for ROC calculation
ground_truth_masks_flatten = ground_truth_masks.reshape(-1)
reconstructed_images_flatten = reconstructed_images.reshape(-1)

# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(ground_truth_masks_flatten, reconstructed_images_flatten)
roc_auc = roc_auc_score(ground_truth_masks_flatten, reconstructed_images_flatten)

print(f"Pixel-wise ROC-AUC: {roc_auc}")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# save the plot to a PNG file
plt.savefig('roc_curve.png')
print(f'[SAVE] ROC curve plot saved to: roc_curve.png')

# save the ROC curve to a CSV file
roc_curve_data = np.stack((fpr, tpr, thresholds)).T
np.savetxt('roc_curve.csv', roc_curve_data, delimiter=',')
print(f'[SAVE] ROC curve saved to: roc_curve.csv')


# --------------------------- MODEL INFERENCE --------------------------- #
