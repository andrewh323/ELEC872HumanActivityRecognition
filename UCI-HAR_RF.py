import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import MinMaxScaler

# Define file paths (Paths must be set for each unique environment)
train_path = r'C:\Users\rando\OneDrive\Desktop\UCI HAR Dataset\train\\'
test_path = r'C:\Users\rando\OneDrive\Desktop\UCI HAR Dataset\test\\'

# Load the features (Path must be set for each unique environment)
features = pd.read_csv('C:/Users/rando/OneDrive/Desktop/UCI HAR Dataset/features.txt', sep=r'\s+', header=None)
feature_names = features[1].values

# Create a dictionary to count occurrences of feature names
name_counts = {}
unique_feature_names = []

# Loop through feature names to create unique names
for name in feature_names:
    if name in name_counts:
        name_counts[name] += 1
        unique_name = f"{name}_{name_counts[name]}"
    else:
        name_counts[name] = 11
        unique_name = name
    unique_feature_names.append(unique_name)

feature_names = np.array(unique_feature_names)

# Load the activity labels (Path must be set for each unique environment)
activity_labels = pd.read_csv('C:/Users/rando/OneDrive/Desktop/UCI HAR Dataset/activity_labels.txt', sep=r'\s+', header=None)
activity_labels.columns = ['activity_id', 'activity_name']

# Load the training data
X_train = pd.read_csv(train_path + 'X_train.txt', sep=r'\s+', header=None, names=feature_names)
y_train = pd.read_csv(train_path + 'y_train.txt', sep=r'\s+', header=None, names=['activity_id'])
subject_train = pd.read_csv(train_path + 'subject_train.txt', sep=r'\s+', header=None, names=['subject'])

# Combine training data
train_data = pd.concat([subject_train, y_train, X_train], axis=1)

# Load the testing data
X_test = pd.read_csv(test_path + 'X_test.txt', sep=r'\s+', header=None, names=feature_names)
y_test = pd.read_csv(test_path + 'y_test.txt', sep=r'\s+', header=None, names=['activity_id'])
subject_test = pd.read_csv(test_path + 'subject_test.txt', sep=r'\s+', header=None, names=['subject'])

# Combine testing data
test_data = pd.concat([subject_test, y_test, X_test], axis=1)

# Combine training and testing data
full_data = pd.concat([train_data, test_data], axis=0)

# Now full_data contains all combined data from training and testing sets


# ================================================================= VISUALIZATION OF THE DATA LOADED ===================================================================

# Activity Distribution

# import matplotlib.pyplot as plt

# # Map activity IDs to names for readability
# full_data = full_data.merge(activity_labels, how='left', left_on='activity_id', right_on='activity_id')

# # Count samples for each activity
# activity_counts = full_data['activity_name'].value_counts()

# # Plot activity distribution
# plt.figure(figsize=(10, 6))
# plt.bar(activity_counts.index, activity_counts.values, color='skyblue')
# plt.title("Activity Distribution")
# plt.xlabel("Activity")
# plt.ylabel("Count")
# plt.xticks(rotation=45)
# plt.show()


# ======================= Gaussian Smoothing =======================
# Apply Gaussian smoothing to reduce noise and smooth the wave-like data
def smooth_data(X, sigma=1.75):
    """
    Apply Gaussian smoothing to each feature in the dataset.
    
    Parameters:
    X (ndarray): Input data (features)
    sigma (float): Standard deviation for the Gaussian filter. 
                   Higher values result in more smoothing.
    
    Returns:
    ndarray: Smoothed data.
    """
    # Apply Gaussian smoothing along each column (feature) of the dataset
    return gaussian_filter1d(X, sigma=sigma, axis=0)

# ======================= Scaling and Smoothing =======================

# Smooth the training and test data
X_train_smoothed = smooth_data(X_train, sigma=1.75)
X_test_smoothed = smooth_data(X_test, sigma=1.75)

# Now apply MinMaxScaler after smoothing
scaler = MinMaxScaler()

# Fit the scaler on the smoothed training data and transform both the training and testing data
X_train_scaled = scaler.fit_transform(X_train_smoothed)
X_test_scaled = scaler.transform(X_test_smoothed)


# ======================= Random Forest Model =======================

# Initialize RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    bootstrap=True,
    random_state=50,
    oob_score=True
)

# Train the model on normalized training data
rf_model.fit(X_train_scaled, y_train.values.ravel())  # y_train.values.ravel() is used to flatten the target variable

# Make predictions on the normalized test data
y_pred = rf_model.predict(X_test_scaled)

# ======================= Evaluation Metrics =======================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')