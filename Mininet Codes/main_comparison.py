import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error
import time

# --- Dataset Preparation (from prepare_dataset.py) ---
class ANN:
    def __init__(self, input_dim, hidden_layers, output_dim):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.weights = []
        self.biases = []

        layer_dims = [input_dim] + hidden_layers + [output_dim]
        for i in range(len(layer_dims) - 1):
            W = np.random.randn(layer_dims[i], layer_dims[i+1]) * 0.01
            b = np.zeros((1, layer_dims[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        A = X
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.sigmoid(Z)
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self.sigmoid(Z)
        return A

    def predict(self, X):
        probabilities = self.forward(X)
        return (probabilities > 0.5).astype(int)

    def get_params(self):
        params = []
        for W in self.weights:
            params.extend(W.flatten())
        for b in self.biases:
            params.extend(b.flatten())
        return np.array(params)

    def set_params(self, params):
        current_idx = 0
        for i in range(len(self.weights)):
            W_shape = self.weights[i].shape
            W_size = W_shape[0] * W_shape[1]
            self.weights[i] = params[current_idx : current_idx + W_size].reshape(W_shape)
            current_idx += W_size

            b_shape = self.biases[i].shape
            b_size = b_shape[0] * b_shape[1]
            self.biases[i] = params[current_idx : current_idx + b_size].reshape(b_shape)
            current_idx += b_size

class SOA:
    def __init__(self, num_seagulls, max_iter, lower_bound, upper_bound, ann_model, X_train, y_train, X_val, y_val):
        self.num_seagulls = num_seagulls
        self.max_iter = max_iter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.ann_model = ann_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.best_seagull_pos = None
        self.best_seagull_fitness = float("inf")

        self.positions = np.random.uniform(lower_bound, upper_bound, (num_seagulls, len(ann_model.get_params())))

    def calculate_fitness(self, position):
        self.ann_model.set_params(position)
        predictions = self.ann_model.forward(self.X_val)
        mse = np.mean(np.square(self.y_val - predictions))
        return mse

    def optimize(self):
        for iteration in range(self.max_iter):
            for i in range(self.num_seagulls):
                fitness = self.calculate_fitness(self.positions[i])

                if fitness < self.best_seagull_fitness:
                    self.best_seagull_fitness = fitness
                    self.best_seagull_pos = self.positions[i].copy()

                C = np.random.rand()
                f = 2
                r = C * f

                # Apply SOA equations component-wise
                # Each component of the position vector is treated as x, y, z for movement
                Mx = self.positions[i] + (2 * np.random.rand() * self.positions[i] * np.cos(r))
                My = self.positions[i] + (2 * np.random.rand() * self.positions[i] * np.sin(r))
                Mz = self.positions[i] + (2 * np.random.rand() * self.positions[i] * r)
                M = (Mx + My + Mz) / 3 # Simplified combination

                A = 2 * (1 - (iteration / self.max_iter))
                B = 2 * np.random.rand()

                if self.best_seagull_pos is not None:
                    Dist = np.abs(self.best_seagull_pos - self.positions[i])
                    self.positions[i] = Dist * B * A + M
                else:
                    self.positions[i] = M

                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            print(f"Iteration {iteration+1}/{self.max_iter}, Best Fitness: {self.best_seagull_fitness:.4f}")
        return self.best_seagull_pos


# --- Evaluation Metrics Function ---
def evaluate_model(model, X_test, y_test, model_name="Model"):
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()

    # Ensure y_pred is 1D for metrics calculation if it's a column vector
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Network-specific metrics (simulated/derived)
    # These are illustrative and would ideally come from the dataset or a more complex simulation
    # For 'Network Anomaly Dataset', 'anomaly' is the target. We'll assume '1' is anomaly.
    total_requests = len(y_test)
    anomalies_detected = np.sum(y_pred == 1)
    true_anomalies = np.sum(y_test == 1)
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    true_negatives = np.sum((y_pred == 0) & (y_test == 0))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))

    # Average service time (simulated)
    # Assuming a base service time and increased time for anomalies
    avg_service_time = 0.1 + (anomalies_detected / total_requests) * 0.5 # seconds

    # Throughput (requests/min) - simulated based on processing time
    processing_time_per_request = (end_time - start_time) / total_requests
    throughput_req_min = (1 / processing_time_per_request) * 60 if processing_time_per_request > 0 else 0

    # Success rate (SR) (%) - proportion of non-anomalous predictions that are correct
    # Or, proportion of correctly handled requests (non-anomalous correctly predicted, or anomalous correctly predicted)
    success_rate = (true_positives + true_negatives) / total_requests * 100

    # Availability (%) - proportion of time the system is operational (not in a 'line down' state)
    # Assuming '0' is operational and '1' is 'line down' (anomaly)
    availability = (np.sum(y_test == 0) / total_requests) * 100 # Actual availability based on true labels
    # Or, predicted availability based on model's non-anomaly predictions
    predicted_availability = (np.sum(y_pred == 0) / total_requests) * 100

    results = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "MAE": mae,
        "Average Service Time (s)": avg_service_time,
        "Throughput (req/min)": throughput_req_min,
        "Success Rate (%)": success_rate,
        "Availability (%)": availability, # Using actual availability for ground truth comparison
        "Predicted Availability (%)": predicted_availability # Model's view of availability
    }
    return results


# --- Main Comparison Logic ---
if __name__ == '__main__':
    import pandas as pd

    # Load and preprocess data
    file_path = 'network_dataset_labeled.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Please ensure it's in the current directory.")
        print("You can download it from: https://www.kaggle.com/datasets/kaiser14/network-anomaly-dataset")
        exit()

    # Drop irrelevant columns
    cols_to_drop = ['timestamp', 'Routers', 'Planned route', 'Network measure', 'Network target', 'Video target']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)

    X = df.drop('anomaly', axis=1)
    y = df['anomaly']

    # Handle categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Split data for training, validation, and testing
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

    input_dim = X_train.shape[1]
    hidden_layers = [10, 5] # Example: two hidden layers with 10 and 5 neurons
    output_dim = 1 # Binary classification (anomaly or not)

    print(f"Input Dimension: {input_dim}")
    print(f"Hidden Layers: {hidden_layers}")
    print(f"Output Dimension: {output_dim}")

    # --- SOANN Training ---
    print("\n--- Training SOANN ---")
    soann_ann = ANN(input_dim, hidden_layers, output_dim)
    # Define bounds for ANN parameters (weights and biases)
    # A rough estimate for the range of initial weights and biases
    param_bounds = 1.0 # Parameters will be between -param_bounds and +param_bounds
    soa_optimizer = SOA(
        num_seagulls=20, # Number of seagulls (population size)
        max_iter=50,    # Number of iterations
        lower_bound=-param_bounds,
        upper_bound=param_bounds,
        ann_model=soann_ann,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    best_soann_params = soa_optimizer.optimize()
    soann_ann.set_params(best_soann_params)

    # --- Standard ANN Training ---
    print("\n--- Training Standard ANN ---")
    # Using scikit-learn's MLPClassifier for a robust standard ANN implementation
    std_ann_model = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layers),
        max_iter=200, # Number of epochs
        activation='logistic', # Matches custom ANN's sigmoid
        solver='adam', # A common and effective optimizer
        random_state=42,
        verbose=False
    )
    std_ann_model.fit(X_train_full, y_train_full) # Train on full training set for standard ANN

    # --- Evaluation ---
    print("\n--- Evaluation Results ---")
    soann_results = evaluate_model(soann_ann, X_test, y_test, model_name="SOANN")
    std_ann_results = evaluate_model(std_ann_model, X_test, y_test, model_name="Standard ANN")

    results_df = pd.DataFrame([soann_results, std_ann_results])
    print(results_df.set_index('Model').T)

    print("\nComparison complete.")
