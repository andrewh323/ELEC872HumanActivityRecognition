import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import zscore
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

# Select device to use for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define file paths (Paths must be set for each unique environment)
train_path = r'C:\Users\18ah11\Documents\QueensU\Masters\ELEC872\ELEC872Project\UCI HAR Dataset\UCI HAR Dataset\train\\'
test_path = r'C:\Users\18ah11\Documents\QueensU\Masters\ELEC872\ELEC872Project\UCI HAR Dataset\UCI HAR Dataset\test\\'

# Load the features (Path must be set for each unique environment)
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
        name_counts[name] = 1
        unique_name = name
    unique_feature_names.append(unique_name)

feature_names = np.array(unique_feature_names)

# Load the activity labels (Path must be set for each unique environment)
activity_labels = pd.read_csv('C:/Users/18ah11/Documents/QueensU/Masters/ELEC872/ELEC872Project/UCI HAR Dataset/UCI HAR Dataset/activity_labels.txt', sep=r'\s+', header=None)
activity_labels.columns = ['activity_id', 'activity_name']

# Load the training data
X_train = pd.read_csv(train_path + 'X_train.txt', sep=r'\s+', header=None, names=feature_names)
y_train = pd.read_csv(train_path + 'y_train.txt', sep=r'\s+', header=None, names=['activity_id'])
subject_train = pd.read_csv(train_path + 'subject_train.txt', sep=r'\s+', header=None, names=['subject'])

# Load the testing data
X_test = pd.read_csv(test_path + 'X_test.txt', sep=r'\s+', header=None, names=feature_names)
y_test = pd.read_csv(test_path + 'y_test.txt', sep=r'\s+', header=None, names=['activity_id'])
subject_test = pd.read_csv(test_path + 'subject_test.txt', sep=r'\s+', header=None, names=['subject'])

# Calculate mean and standard deviation of training data
features_to_scale = X_train.columns
train_mean = X_train[features_to_scale].mean()
train_std = X_train[features_to_scale].std()

# Standardize training set
X_train[features_to_scale] = (X_train[features_to_scale] - train_mean) / train_std
# Standardize test set based on the mean and std_dev from training set
X_test[features_to_scale] = (X_test[features_to_scale] - train_mean) / train_std


print("Data preprocessing completed successfully.")

# Define training and testing tensors for PyTorch
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.LongTensor(y_train.values.flatten())

X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.LongTensor(y_test.values.flatten())

# Zero-index output tensors
y_train_tensor -= 1
y_test_tensor -= 1

# Define training and testing datasets
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

# Define training and testing dataloaders
trainLoader = DataLoader(train_data, batch_size=32, shuffle=True)
testLoader = DataLoader(test_data, batch_size=32, shuffle=False)


hiddenSize = 256
inputSize = X_train.shape[1]
outputSize = len(activity_labels['activity_id'].unique())
dropout = 0.3
epochs = 50
learningRate = 0.0004

model = nn.Sequential(
    # Connect input layer to hidden layer
    nn.Linear(inputSize, hiddenSize),
    # Apply ReLU activation function to hidden layer
    nn.ReLU(),
    # Set dropout
    nn.Dropout(dropout),
    # Connect hidden layer to output layer
    nn.Linear(hiddenSize, outputSize)
).to(device)

# Initialize the weights of each layer
def initialize_weights(m):
    # Check to make sure weight initialization is done only on linear layers
    if isinstance(m, nn.Linear):
        # Initialize parameters using He initialization
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Apply the weight initialization to the model
model.apply(initialize_weights)
# Set optimizer to Adam
optimizer = optim.Adam(model.parameters(), lr=learningRate)
# Set loss function to Cross Entropy
criterion = nn.CrossEntropyLoss()

# Training neural network
def train(model, trainLoader, criterion, optimizer, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainLoader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Calculate loss function
            loss = criterion(outputs, labels)
            # Back propagation
            loss.backward()
            # Update weights
            optimizer.step()
            running_loss += loss.item()
            # Report progress every 300 steps
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainLoader):.4f}")

# Testing neural network
def test(model, testLoader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in testLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    # Calculate accuracy
    accuracy = 100*correct/total
    # Calculate F1 Score
    f1 = 100*f1_score(all_labels, all_predictions, average='weighted')
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'F1 Score on test set: {f1:.2f}%')
    
train(model, trainLoader, criterion, optimizer, device)
test(model, testLoader, device)
