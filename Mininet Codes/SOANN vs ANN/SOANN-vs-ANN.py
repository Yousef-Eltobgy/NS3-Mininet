#Fixes applied (scientifically correct):
## Fix A — F1-driven fitness (core fix) --> SOA now directly optimizes F1-score (with recall awareness).
## Fix B — Threshold optimization inside SOA --> Threshold ∈ [0.2 – 0.8] is now part of the solution vector.
## Fix C — Class-weighted F1 --> Failure/congestion class is explicitly emphasized.
## Fix D — SOA stagnation control --> Worst 20% of seagulls are re-initialized every 10 iterations.

## testing on 15k samples

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_absolute_error
)
from sklearn.neural_network import MLPClassifier

# ==========================================
# 1. Custom ANN
# ==========================================
class CustomANN:
    def __init__(self, input_dim, hidden_layers):
        self.weights = []
        self.biases = []

        layer_dims = [input_dim] + hidden_layers + [1]
        for i in range(len(layer_dims) - 1):
            self.weights.append(np.random.randn(layer_dims[i], layer_dims[i+1]) * 0.05)
            self.biases.append(np.zeros((1, layer_dims[i+1])))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        A = X
        for i in range(len(self.weights)):
            A = self.sigmoid(np.dot(A, self.weights[i]) + self.biases[i])
        return A.flatten()

    def get_params(self):
        params = []
        for W, b in zip(self.weights, self.biases):
            params.extend(W.flatten())
            params.extend(b.flatten())
        return np.array(params)

    def set_params(self, params):
        idx = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            self.weights[i] = params[idx:idx+w_size].reshape(self.weights[i].shape)
            idx += w_size

            b_size = self.biases[i].size
            self.biases[i] = params[idx:idx+b_size].reshape(self.biases[i].shape)
            idx += b_size


# ==========================================
# 2. Seagull Optimization Algorithm (FIXED)
# ==========================================
class SOA:
    def __init__(self, pop, iters, lb, ub, ann, X_val, y_val):
        self.pop = pop
        self.iters = iters
        self.lb = lb
        self.ub = ub
        self.ann = ann
        self.X_val = X_val
        self.y_val = y_val

        self.dim = len(ann.get_params()) + 1  # +1 for threshold
        self.positions = np.random.uniform(lb, ub, (pop, self.dim))
        self.best_pos = None
        self.best_fit = -np.inf

    def fitness(self, pos):
        self.ann.set_params(pos[:-1])
        threshold = np.clip(pos[-1], 0.2, 0.8)

        probs = self.ann.forward(self.X_val)
        preds = (probs > threshold).astype(int)

        f1 = f1_score(self.y_val, preds, zero_division=0)
        recall = recall_score(self.y_val, preds, zero_division=0)

        return 0.7 * f1 + 0.3 * recall

    def optimize(self):
        for t in range(self.iters):
            fitness_vals = []

            for i in range(self.pop):
                fit = self.fitness(self.positions[i])
                fitness_vals.append(fit)

                if fit > self.best_fit:
                    self.best_fit = fit
                    self.best_pos = self.positions[i].copy()

                A = 2 * (1 - t / self.iters)
                B = np.random.rand()
                r = np.random.rand()

                D = abs(self.best_pos - self.positions[i])
                self.positions[i] = self.best_pos - A * D * B * np.cos(2 * np.pi * r)
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

            # Anti-stagnation
            if (t + 1) % 10 == 0:
                worst = np.argsort(fitness_vals)[:int(0.2 * self.pop)]
                self.positions[worst] = np.random.uniform(self.lb, self.ub, (len(worst), self.dim))
                print(f"SOA Iteration {t+1}/{self.iters}, Best F1-based Fitness: {self.best_fit:.4f}")

        return self.best_pos


# ==========================================
# 3. Evaluation
# ==========================================
def evaluate(model, X, y, threshold=0.5, is_soann=False):
    start = time.time()

    if is_soann:
        probs = model.forward(X)
        preds = (probs > threshold).astype(int)
        roc_probs = probs
    else:
        preds = model.predict(X)
        roc_probs = model.predict_proba(X)[:, 1]

    end = time.time()

    return {
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds, zero_division=0),
        "Recall": recall_score(y, preds, zero_division=0),
        "F1-Score": f1_score(y, preds, zero_division=0),
        "ROC-AUC": roc_auc_score(y, roc_probs),
        "MAE": mean_absolute_error(y, preds),
        "Avg Service Time (s)": (end - start) / len(y),
        "Throughput (req/min)": (len(y) / (end - start)) * 60
    }


# ==========================================
# 4. Main
# ==========================================
if __name__ == "__main__":
    df = pd.read_csv("/kaggle/input/dc-64-mininet/dc_64_dataset.csv")

    X = df.drop("line_down", axis=1)
    y = df["line_down"]

    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # ---- SOANN ----
    print("\n--- Training SOANN (FIXED) ---")
    soann = CustomANN(X_train.shape[1], [10, 5])
    soa = SOA(30, 50, -5, 5, soann, X_val, y_val)
    best = soa.optimize()

    soann.set_params(best[:-1])
    best_threshold = np.clip(best[-1], 0.2, 0.8)

    # ---- ANN ----
    print("\n--- Training Standard ANN ---")
    ann = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
    ann.fit(X_train, y_train)

    # ---- Evaluation ----
    soann_res = evaluate(soann, X_test, y_test, best_threshold, True)
    ann_res = evaluate(ann, X_test, y_test)

    results = pd.DataFrame([soann_res, ann_res], index=["SOANN", "Standard ANN"])
    print("\n--- Comparison Results ---")
    print(results.T)

