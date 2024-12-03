import flwr as fl

import numpy as np
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

import cv2
from PIL import Image

# Optional: Disable oneDNN custom operations for exact reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_images(dir,limit=-1):
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for file in files[0:limit]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                
                image = cv2.resize(image, (224,224))
                
                image = image.reshape(-1)
                images.append(image) 
                labels.append(root.split("/")[-1])
    return np.array(images), labels


# Load a random subset of data
def get_random_subset(X, y, sample_size):
    indices = np.random.choice(len(X), size=sample_size, replace=False).astype(int)
    X_sample = []
    y_sample = []
    for i in indices:
       X_sample.append(X[i])
       y_sample.append(y[i])
    return np.array(X_sample), np.array(y_sample)

# Simulated client class
class XGBClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, train_labels, val_data, val_labels):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels

    def get_parameters(self):
        # Return model parameters as numpy arrays if model has been trained
        if hasattr(self.model, 'coef_'):
            return [self.model.feature_importances_]
        else:
            # Return a default array if model is not yet fitted
            return [np.zeros(self.train_data.shape[1])]

    def set_parameters(self, parameters):
        # Fit model with dummy data to initialize it, then set booster weights manually
        self.model.fit(self.train_data, self.train_labels)
        booster = self.model.get_booster()
        booster.set_attr(features=str(parameters))

    def fit(self, parameters, config):
        # Set parameters, train model, and return new parameters
        self.set_parameters(parameters)
        self.model.fit(self.train_data, self.train_labels)
        return self.get_parameters(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        predictions = self.model.predict(self.val_data)
        accuracy = accuracy_score(self.val_labels, predictions)
        return accuracy, len(self.val_data), {}

# Weight aggregation function
def aggregate_weights(weights):
    # Average the weights across clients
    avg_weights = [np.mean([w[i] for w in weights], axis=0) for i in range(len(weights[0]))]
    return avg_weights

# Load data
image_dir = "../dcnn/spark22/train"
X, y = load_images(image_dir)
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_sample_size = 1500
val_sample_size = 40

#X_train, y_train = get_random_subset(X_train, y_train, train_sample_size)
#X_val, y_val = get_random_subset(X_val, y_val, val_sample_size)

# Apply PCA
pca = PCA(n_components=16, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

# Split dataset for multiple clients
num_clients = 5
client_data_splits = np.array_split(X_train_pca, num_clients)
client_label_splits = np.array_split(y_train, num_clients)

# Initialize clients
clients = []
for i in range(num_clients):
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        client_data_splits[i], client_label_splits[i], test_size=0.2, random_state=42
    )
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    clients.append(XGBClient(model, X_train_split, y_train_split, X_val_split, y_val_split))

# Define federated learning strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=num_clients,
    min_available_clients=num_clients,
)

# Define simulation function
def fit_round(server_round):
    print(f"\nRound {server_round}")
    weights = []
    client_metrics = []

    # Each client trains on its data and returns metrics
    for client in clients:
        fit_weights, _, _ = client.fit(client.get_parameters(), config={})
        weights.append(fit_weights)
        
    # Aggregate weights using manual averaging
    aggregated_weights = aggregate_weights(weights)
    
    # Distribute aggregated weights to clients for the next round
    for client in clients:
        client.set_parameters(aggregated_weights)

    # Evaluate on each client and average the metrics
    for client in clients:
        acc, _, _ = client.evaluate(client.get_parameters(), config={})
        client_metrics.append(acc)
    
    avg_accuracy = np.mean(client_metrics)
    print(f"Round {server_round} Average Accuracy: {avg_accuracy:.4f}")

# Run simulation for multiple rounds
for round_num in range(1, 4):  # Adjust the number of rounds as needed
    fit_round(round_num)

# Evaluate the final model on the validation set
print("\nEvaluating final global model on validation set...")
global_model = clients[0].model  # Use one client's model as the global model after training
y_val_pred = global_model.predict(X_val_pca)

# Calculate metrics
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print("Validation Accuracy:", val_accuracy)
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)
print("Validation F1 Score:", val_f1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'.")

# ROC Curve and AUC
y_val_bin = label_binarize(y_val, classes=[x for x in range(11)])
y_score = global_model.predict_proba(X_val_pca)

print(y_val_bin.shape)
print(y_score.shape)

fpr, tpr, _ = roc_curve(y_val_bin[:,-1].ravel(), y_score[:, 1].ravel())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
print("ROC curve saved as 'roc_curve.png'.")


results = classification_report(y_true=y_val, y_pred=y_val_pred)
with open(f"metrics.txt", "w") as f:
    # Write some text to the file
    f.write(str(results))
