import flwr as fl
import numpy as np
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
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
import rasterio
from PIL import Image

# Optional: Disable oneDNN custom operations for exact reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load GeoTIFF Image and stack bands
def load_geotiff_image(tiff_file_path):
    with rasterio.open(tiff_file_path) as src:
        band_data = src.read()  # Shape will be (bands, height, width)
    return band_data

# Flatten image bands for input to PCA
def flatten_image_bands(band_data):
    return band_data.reshape(band_data.shape[0], -1).T

# Load Mask/Label (forest vs background)
def load_mask(mask_file_path):
    mask = Image.open(mask_file_path)
    return np.array(mask).flatten()

# Load a random subset of data
def load_random_subset(image_dir, mask_dir, sample_size):
    all_image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    sampled_files = random.sample(all_image_files, sample_size)

    all_images = []
    all_masks = []

    for file_name in sampled_files:
        tiff_image_path = os.path.join(image_dir, file_name)
        mask_image_path = os.path.join(mask_dir, file_name)

        image_data = load_geotiff_image(tiff_image_path)
        flattened_image = flatten_image_bands(image_data)

        mask_data = load_mask(mask_image_path)

        all_images.append(flattened_image)
        all_masks.append(mask_data)

    return np.vstack(all_images), np.hstack(all_masks)

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
train_image_dir = "C:\\Users\\giese\\481\\AMAZON\\AMAZON\\Training\\image"
train_mask_dir = "C:\\Users\\giese\\481\\AMAZON\\AMAZON\\Training\\label"
val_image_dir = "C:\\Users\\giese\\481\\AMAZON\\AMAZON\\Validation\\images"
val_mask_dir = "C:\\Users\\giese\\481\\AMAZON\\AMAZON\\Validation\\masks"
train_sample_size = 50
val_sample_size = 10

X_train, y_train = load_random_subset(train_image_dir, train_mask_dir, train_sample_size)
X_val, y_val = load_random_subset(val_image_dir, val_mask_dir, val_sample_size)

# Apply PCA
pca = PCA(n_components=4, whiten=True, random_state=42)
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
y_val_bin = label_binarize(y_val, classes=[0, 1])
y_score = global_model.predict_proba(X_val_pca)
fpr, tpr, _ = roc_curve(y_val_bin.ravel(), y_score[:, 1].ravel())
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