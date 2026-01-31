import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class ANN:
    def __init__(self, input_dim, hidden_layers, output_dim):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
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
        # Output layer (no activation for regression, sigmoid for binary classification)
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self.sigmoid(Z) # Assuming binary classification
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

        # Initialize seagull positions (ANN parameters)
        self.positions = np.random.uniform(lower_bound, upper_bound, (num_seagulls, len(ann_model.get_params())))

    def calculate_fitness(self, position):
        self.ann_model.set_params(position)
        predictions = self.ann_model.forward(self.X_val)
        # Using Mean Squared Error as fitness for optimization
        mse = np.mean(np.square(self.y_val - predictions))
        return mse

    def optimize(self):
        for iteration in range(self.max_iter):
            for i in range(self.num_seagulls):
                fitness = self.calculate_fitness(self.positions[i])

                if fitness < self.best_seagull_fitness:
                    self.best_seagull_fitness = fitness
                    self.best_seagull_pos = self.positions[i].copy()

                # Update seagull position (simplified SOA movement)
                # SOA equations (simplified for demonstration, refer to original paper for full details)
                # 1. Movement behavior (x, y, z are position components)
                # C is a random number between 0 and 1
                C = np.random.rand()
                # f is a frequency controlling the change of r
                f = 2
                # r is a random number between 0 and 2pi
                r = C * f
                # x, y, z components of the seagull's position
                x = self.positions[i]
                y = self.positions[i]
                z = self.positions[i]

                # Movement in x, y, z directions
                Mx = x + (2 * np.random.rand() * x * np.cos(r))
                My = y + (2 * np.random.rand() * y * np.sin(r))
                Mz = z + (2 * np.random.rand() * z * r)
                M = (Mx + My + Mz) / 3 # Simplified combination

                # 2. Attack behavior (moving towards the best position)
                # A is the attacking density factor
                A = 2 * (1 - (iteration / self.max_iter))
                # B is a random number between 0 and 2
                B = 2 * np.random.rand()

                # Distance between the current seagull and the best seagull
                if self.best_seagull_pos is not None:
                    Dist = np.abs(self.best_seagull_pos - self.positions[i])
                    # New position based on attack
                    self.positions[i] = Dist * B * A + M
                else:
                    # If no best position yet, just use movement behavior
                    self.positions[i] = M

                # Apply bounds
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            print(f"Iteration {iteration+1}/{self.max_iter}, Best Fitness: {self.best_seagull_fitness:.4f}")
        return self.best_seagull_pos


# Standard ANN training (using scikit-learn's MLPClassifier for simplicity and robustness)
def train_standard_ann(X_train, y_train, X_test, y_test, input_dim, hidden_layers, output_dim, max_iter=200):
    print("\nTraining Standard ANN...")
    # MLPClassifier is a good representation of a standard ANN
    # The hidden_layer_sizes parameter takes a tuple, e.g., (10, 5) for two hidden layers with 10 and 5 neurons
    model = MLPClassifier(hidden_layer_sizes=tuple(hidden_layers), max_iter=max_iter, random_state=42, activation=\'logistic\', solver=\'adam\')
    model.fit(X_train, y_train)
    print("Standard ANN training complete.")
    return model


if __name__ == '__main__':
    # This part is for testing the model implementations, not for full comparison yet.
    # You would typically run this after prepare_dataset.py
    print("Running dummy test for ANN and SOA classes...")

    # Dummy data
    input_dim_test = 10
    hidden_layers_test = [5]
    output_dim_test = 1
    num_samples = 100

    X_train_test = np.random.rand(num_samples, input_dim_test)
    y_train_test = np.random.randint(0, 2, num_samples)
    X_val_test = np.random.rand(num_samples // 4, input_dim_test)
    y_val_test = np.random.randint(0, 2, num_samples // 4)

    # Test ANN
    ann_test = ANN(input_dim_test, hidden_layers_test, output_dim_test)
    initial_params = ann_test.get_params()
    print(f"Initial ANN parameters size: {len(initial_params)}")
    predictions = ann_test.predict(X_train_test)
    print(f"Initial ANN accuracy (dummy data): {accuracy_score(y_train_test, predictions):.4f}")

    # Test SOA (simplified)
    soa = SOA(
        num_seagulls=10,
        max_iter=5,
        lower_bound=-1,
        upper_bound=1,
        ann_model=ann_test,
        X_train=X_train_test,
        y_train=y_train_test,
        X_val=X_val_test,
        y_val=y_val_test
    )
    best_params = soa.optimize()
    ann_test.set_params(best_params)
    soa_predictions = ann_test.predict(X_val_test)
    print(f"SOA optimized ANN accuracy (dummy data): {accuracy_score(y_val_test, soa_predictions):.4f}")

    # Test Standard ANN
    std_ann_model = train_standard_ann(X_train_test, y_train_test, X_val_test, y_val_test, input_dim_test, hidden_layers_test, output_dim_test)
    std_ann_predictions = std_ann_model.predict(X_val_test)
    print(f"Standard ANN accuracy (dummy data): {accuracy_score(y_val_test, std_ann_predictions):.4f}")

    print("Dummy test complete.")
