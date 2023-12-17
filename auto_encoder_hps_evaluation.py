import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# Set Paths
dataset_dir = 'capsule/capsule/'
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# Load your trained model
model = load_model('saved_models/final_best_autoencoder.h5')

# Process test images (resize, normalize, etc.)
# Assuming test_images is a numpy array of your test images
# test_images = ...

# Process ground truth masks
ground_truth_masks = load_ground_truth_masks('path/to/ground_truth_masks', test_images_filenames, 256, 256)

# Predict with the model
predicted_images = model.predict(test_images)

# Calculate reconstruction error
reconstruction_error = np.mean(np.square(test_images - predicted_images), axis=(1, 2, 3))

# Flatten the images and masks for ROC calculation
ground_truth_masks_flatten = ground_truth_masks.reshape(-1)
predicted_images_flatten = predicted_images.reshape(-1)

# Calculate ROC and AUC
fpr, tpr, thresholds = roc_curve(ground_truth_masks_flatten, predicted_images_flatten)
roc_auc = roc_auc_score(ground_truth_masks_flatten, predicted_images_flatten)

# Plot ROC curve
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
