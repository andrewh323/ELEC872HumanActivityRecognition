import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.ndimage import gaussian_filter1d
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import MinMaxScaler

# Define file paths
train_path = r'C:\Users\18ah11\Documents\QueensU\Masters\ELEC872\ELEC872Project\UCI HAR Dataset\UCI HAR Dataset\train\\'
test_path = r'C:\Users\18ah11\Documents\QueensU\Masters\ELEC872\ELEC872Project\UCI HAR Dataset\UCI HAR Dataset\test\\'

# Load the features
features = pd.read_csv('C:/Users/18ah11/Documents/QueensU/Masters/ELEC872/ELEC872Project/UCI HAR Dataset/UCI HAR Dataset/features.txt', sep=r'\s+', header=None)
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

# Load the activity labels
activity_labels = pd.read_csv('C:/Users/18ah11/Documents/QueensU/Masters/ELEC872/ELEC872Project/UCI HAR Dataset/UCI HAR Dataset/activity_labels.txt', sep=r'\s+', header=None)
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

# The full_data is created by concatenating the training data (train_data) and testing data (test_data) together. This is done using the Pandas concat function.
# Specifically:
# X_train and X_test contain the accelerometer data (along with other features) for the training and testing datasets, respectively.
# These datasets are concatenated to form full_data, which includes all subjects and all activities across both training and testing sets
# The visualization data is absolutely correct. 1 > shows the activity distribition and 2 > shows the Accelerometer data for specific users. The graph was also compared to a manual graph created in excel and both of them are the same 

# 1. Activity Distribution

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

# 2. Time Series of Accelerometer Data for a Specific Subject and Activity

# import matplotlib.pyplot as plt

# # Ask for subject ID input
# subject_id = int(input("Enter subject ID (1-30): "))

# # Ask for activity input
# activity_id = int(input("Enter activity ID (1-6) [1: Walking, 2: Walking Upstairs, 3: Walking Downstairs, 4: Sitting, 5: Standing, 6: Laying]: "))

# # Get the activity name based on the activity ID
# try:
#     activity = activity_labels[activity_labels['activity_id'] == activity_id]['activity_name'].values[0]
# except IndexError:
#     print("Invalid activity ID. Please try again.")
#     exit()

# # Filter the full_data for the selected subject
# sample_data = full_data[full_data['subject'] == subject_id]

# # Further filter sample_data by the chosen activity
# sample_data = sample_data[sample_data['activity_id'] == activity_id]

# # Check if there's data for the selected subject and activity
# if not sample_data.empty:
#     # Reset index for clean sample representation
#     sample_data.reset_index(drop=True, inplace=True)
    
#     # Plot accelerometer signals over time
#     plt.figure(figsize=(15, 6))
#     plt.plot(sample_data.index, sample_data['tBodyAcc-mean()-X'], label="Body Acc X", color='blue', marker='o', markersize=3)
#     plt.plot(sample_data.index, sample_data['tBodyAcc-mean()-Y'], label="Body Acc Y", color='orange', marker='x', markersize=3)
#     plt.plot(sample_data.index, sample_data['tBodyAcc-mean()-Z'], label="Body Acc Z", color='green', marker='s', markersize=3)
#     plt.title(f"Accelerometer Data for Subject {subject_id} - Activity: {activity}", fontsize=14)
#     plt.xlabel("Sample Index", fontsize=12)
#     plt.ylabel("Mean Acceleration (g)", fontsize=12)
#     plt.grid(alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
# else:
#     print(f"No data found for Subject {subject_id} performing Activity {activity}.")


# # ============================= EXCEL CODE =============================
# To visualize the data for a specific person and activity in excel uncomment the below code 

# import pandas as pd

# # Ask for subject ID input
# subject_id = int(input("Enter subject ID (1-30): "))

# # Ask for activity input
# activity_id = int(input("Enter activity ID (1-6) [1: Walking, 2: Walking Upstairs, 3: Walking Downstairs, 4: Sitting, 5: Standing, 6: Laying]: "))

# # Filter the full_data for the selected subject
# sample_data = full_data[full_data['subject'] == subject_id]

# # Further filter sample_data by the chosen activity
# sample_data = sample_data[sample_data['activity_id'] == activity_id]

# # Check if there's data for the selected subject and activity
# if not sample_data.empty:
#     # Define the filename for the Excel file
#     filename = f"subject_{subject_id}_activity_{activity_id}.xlsx"
    
#     # Save the filtered data to an Excel file
#     sample_data.to_excel(filename, index=False)
    
#     print(f"Data for Subject {subject_id} performing Activity {activity_id} has been saved to '{filename}'.")
# else:
#     print(f"No data found for Subject {subject_id} performing Activity {activity_id}.")


# ======================= XGBoost Model =======================


# ======================= Handle Missing Values =======================
# Use SimpleImputer to handle missing values by replacing them with the mean of each column
imputer = SimpleImputer(strategy='mean')

# Apply imputation on training and testing data
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# ======================= Reindex Activity Labels =======================
# Reindex the labels to start from 0 instead of 1
y_train = y_train - 1
y_test = y_test - 1

# ======================= Gaussian Smoothing =======================
def smooth_data(X, sigma=1.75):
    """
    Apply Gaussian smoothing to each feature in the dataset.
    
    Parameters:
    X (ndarray): Input data (features).
    sigma (float): Standard deviation for the Gaussian filter. Higher values result in more smoothing.
    
    Returns:
    ndarray: Smoothed data.
    """
    return gaussian_filter1d(X, sigma=sigma, axis=0)

# Apply smoothing to the imputed training and testing data
X_train_smoothed = smooth_data(X_train_imputed, sigma=1.75)
X_test_smoothed = smooth_data(X_test_imputed, sigma=1.75)

# ======================= Min-Max Scaling =======================
scaler = MinMaxScaler()

# Fit the scaler on the smoothed training data and transform both the training and testing data
X_train_scaled = scaler.fit_transform(X_train_smoothed)
X_test_scaled = scaler.transform(X_test_smoothed)

# ======================= CatBoost Model =======================
import catboost as cb

# Initialize CatBoostClassifier
catboost_model = cb.CatBoostClassifier(
    iterations=2000,        # Number of trees
    depth=11,               # Depth of trees
    learning_rate=0.25,      # Learning rate
    loss_function='MultiClass',  # Multi-class classification
    task_type='GPU',        # Add this line to enable GPU training
    devices='0',
    cat_features=[],        # Specify categorical features if available
    random_seed=20,
    verbose=100             # Output training progress every 100 iterations
)

# Train the model on the preprocessed training data
catboost_model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = catboost_model.predict(X_test_scaled)

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