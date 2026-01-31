import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CONFIGURATION
# ===============================
WINDOW = 7
RANDOM_SEED = 42

# SOANN Configuration
SOA_POP = 15
SOA_ITERS = 40
SOA_EPOCHS = 8

# Final Training Configuration
FINAL_EPOCHS = 50
BATCH_SIZE = 64

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ===============================
# LOAD AND PREPARE DATA
# ===============================
print("Loading dataset...")
df = pd.read_csv("dc_64_dataset.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Class distribution:\n{df['line_down'].value_counts()}")

X = df.drop("line_down", axis=1).values
y = df["line_down"].values

# Scale features
scaler = RobustScaler()
X = scaler.fit_transform(X)

# ===============================
# CREATE TEMPORAL SEQUENCES
# ===============================
def make_sequences(X, y, window, step=1):
    """Create sliding window sequences for temporal analysis"""
    xs, ys = [], []
    for i in range(0, len(X) - window, step):
        xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(xs), np.array(ys)

print(f"\nCreating temporal sequences with window={WINDOW}...")
X_seq, y_seq = make_sequences(X, y, WINDOW, step=1)
print(f"Sequence shape: {X_seq.shape}")

# Split data
Xtr, Xte, ytr, yte = train_test_split(
    X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=RANDOM_SEED, shuffle=True
)

Xtr, Xval, ytr, yval = train_test_split(
    Xtr, ytr, test_size=0.1, stratify=ytr, random_state=RANDOM_SEED
)

print(f"\nData splits:")
print(f"  Training: {Xtr.shape[0]} samples")
print(f"  Validation: {Xval.shape[0]} samples")
print(f"  Test: {Xte.shape[0]} samples")

# ===============================
# CLASS WEIGHTS
# ===============================
neg, pos = np.bincount(ytr)
total = neg + pos
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print(f"\nClass weights: 0={weight_for_0:.2f}, 1={weight_for_1:.2f}")

# ===============================
# MODEL ARCHITECTURES
# ===============================
def build_lstm_model(units, dropout, lr, use_bidirectional=True):
    """Build LSTM model for temporal network metrics prediction"""
    model = Sequential()
    model.add(Input(shape=(WINDOW, X_seq.shape[2])))
    
    if use_bidirectional:
        model.add(Bidirectional(LSTM(units, return_sequences=False)))
    else:
        model.add(LSTM(units, return_sequences=False))
    
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(units // 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout * 0.8))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

# ===============================
# SEAGULL OPTIMIZATION FITNESS
# ===============================
def soann_fitness(agent):
    """Evaluate fitness of SOANN agent parameters"""
    # Decode parameters
    units = int(np.clip(agent[0], 32, 128))
    dropout = np.clip(agent[1], 0.1, 0.6)
    lr = np.clip(agent[2], 5e-5, 1e-3)
    threshold = np.clip(agent[3], 0.3, 0.7)
    use_bidirectional = agent[4] > 0.5
    
    # Build and train model
    model = build_lstm_model(units, dropout, lr, use_bidirectional)
    
    early_stop = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, mode='min'
    )
    
    model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=SOA_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        class_weight=class_weight,
        callbacks=[early_stop]
    )
    
    # Evaluate
    probs = model.predict(Xval, verbose=0).ravel()
    preds = (probs > threshold).astype(int)
    
    # Anti-collapse mechanism
    if preds.sum() == 0:
        preds = (probs > threshold * 0.5).astype(int)
        if preds.sum() == 0:
            return 0.0
    
    # Combined fitness metric
    f1 = f1_score(yval, preds, zero_division=0)
    prec = precision_score(yval, preds, zero_division=0)
    rec = recall_score(yval, preds, zero_division=0)
    
    return 0.4 * f1 + 0.3 * prec + 0.3 * rec

# ===============================
# SEAGULL OPTIMIZATION ALGORITHM
# ===============================
def seagull_optimization():
    """Run Seagull Optimization to find best hyperparameters"""
    print("\n" + "="*60)
    print("SEAGULL OPTIMIZATION - HYPERPARAMETER SEARCH")
    print("="*60)
    
    # Initialize population [units, dropout, lr, threshold, bidirectional_flag]
    agents = np.random.uniform(
        [32, 0.1, 5e-5, 0.3, 0.0],
        [128, 0.6, 1e-3, 0.7, 1.0],
        (SOA_POP, 5)
    )
    
    best_agent = None
    best_score = -1
    history = []
    
    momentum = 0.1
    exploration_rate = 0.3
    
    for iteration in range(SOA_ITERS):
        scores = []
        
        # Evaluate all agents
        for idx, agent in enumerate(agents):
            score = soann_fitness(agent)
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_agent = agent.copy()
                print(f"  Iter {iteration+1}: New best fitness = {best_score:.4f}")
        
        # Update agents with seagull-inspired movement
        for j in range(SOA_POP):
            if np.random.rand() < exploration_rate:
                # Exploration phase
                agents[j] += np.random.randn(5) * 0.2 * (1 - iteration/SOA_ITERS)
            else:
                # Exploitation phase
                move = (best_agent - agents[j]) * momentum
                agents[j] += move + np.random.randn(5) * 0.1 * np.exp(-iteration/10)
            
            # Enforce bounds
            agents[j][0] = np.clip(agents[j][0], 32, 128)
            agents[j][1] = np.clip(agents[j][1], 0.1, 0.6)
            agents[j][2] = np.clip(agents[j][2], 5e-5, 1e-3)
            agents[j][3] = np.clip(agents[j][3], 0.3, 0.7)
            agents[j][4] = np.clip(agents[j][4], 0.0, 1.0)
        
        history.append(best_score)
        
        if (iteration + 1) % 5 == 0:
            print(f"  Iter {iteration+1}/{SOA_ITERS}: Best={best_score:.4f}, Avg={np.mean(scores):.4f}")
    
    return best_agent, best_score, history

# ===============================
# TRAIN SOANN MODEL
# ===============================
print("\n🔵 Training SOANN-LSTM Model...")
best_params, best_fitness, soa_history = seagull_optimization()

# Decode optimized parameters
soann_units = int(best_params[0])
soann_dropout = best_params[1]
soann_lr = best_params[2]
soann_threshold = best_params[3]
soann_bidirectional = best_params[4] > 0.5

print(f"\n✅ Optimized SOANN Parameters:")
print(f"  Units: {soann_units}")
print(f"  Dropout: {soann_dropout:.3f}")
print(f"  Learning Rate: {soann_lr:.6f}")
print(f"  Threshold: {soann_threshold:.3f}")
print(f"  Bidirectional: {soann_bidirectional}")

# Train final SOANN model
print("\n🔵 Training final SOANN-LSTM model with optimized parameters...")
soann_model = build_lstm_model(soann_units, soann_dropout, soann_lr, soann_bidirectional)

soann_callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

soann_history = soann_model.fit(
    Xtr, ytr,
    validation_data=(Xval, yval),
    epochs=FINAL_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=soann_callbacks,
    verbose=1
)

# ===============================
# TRAIN STANDARD ANN MODEL
# ===============================
print("\n" + "="*60)
print("STANDARD ANN-LSTM MODEL (BASELINE)")
print("="*60)

# Fixed hyperparameters for baseline
ann_units = 64
ann_dropout = 0.3
ann_lr = 0.001
ann_bidirectional = True

print(f"\nStandard ANN Parameters:")
print(f"  Units: {ann_units}")
print(f"  Dropout: {ann_dropout}")
print(f"  Learning Rate: {ann_lr}")
print(f"  Bidirectional: {ann_bidirectional}")

print("\n🔵 Training standard ANN-LSTM model...")
ann_model = build_lstm_model(ann_units, ann_dropout, ann_lr, ann_bidirectional)

ann_callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

ann_history = ann_model.fit(
    Xtr, ytr,
    validation_data=(Xval, yval),
    epochs=FINAL_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=ann_callbacks,
    verbose=1
)

# ===============================
# EVALUATION FUNCTION
# ===============================
def evaluate_model(model, name, threshold=0.5):
    """Evaluate model performance on test set"""
    print(f"\n{'='*60}")
    print(f"{name} - TEST SET EVALUATION")
    print(f"{'='*60}")
    
    # Predict
    test_probs = model.predict(Xte, verbose=0).ravel()
    
    # Find optimal threshold on validation set
    val_probs = model.predict(Xval, verbose=0).ravel()
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_thr = 0.5
    best_f1 = 0
    
    for thr in thresholds:
        preds = (val_probs > thr).astype(int)
        f1 = f1_score(yval, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    
    print(f"Optimal threshold: {best_thr:.3f}")
    
    # Test predictions with optimal threshold
    test_preds = (test_probs > best_thr).astype(int)
    
    # Metrics
    acc = accuracy_score(yte, test_preds)
    prec = precision_score(yte, test_preds, zero_division=0)
    rec = recall_score(yte, test_preds, zero_division=0)
    f1 = f1_score(yte, test_preds, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(yte, test_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'threshold': best_thr,
        'confusion_matrix': cm
    }

# ===============================
# COMPARE MODELS
# ===============================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

soann_results = evaluate_model(soann_model, "SOANN-LSTM", soann_threshold)
ann_results = evaluate_model(ann_model, "Standard ANN-LSTM")

# ===============================
# COMPARISON SUMMARY
# ===============================
print("\n" + "="*60)
print("FINAL COMPARISON SUMMARY")
print("="*60)

metrics = ['accuracy', 'precision', 'recall', 'f1']
print(f"\n{'Metric':<15} {'SOANN-LSTM':<15} {'ANN-LSTM':<15} {'Improvement':<15}")
print("-" * 60)

for metric in metrics:
    soann_val = soann_results[metric]
    ann_val = ann_results[metric]
    improvement = ((soann_val - ann_val) / ann_val * 100) if ann_val > 0 else 0
    
    print(f"{metric.capitalize():<15} {soann_val:<15.4f} {ann_val:<15.4f} {improvement:>+14.2f}%")

# ===============================
# VISUALIZATION
# ===============================
print("\n🔵 Generating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Training loss comparison
axes[0, 0].plot(soann_history.history['loss'], label='SOANN-LSTM Train', linewidth=2)
axes[0, 0].plot(soann_history.history['val_loss'], label='SOANN-LSTM Val', linewidth=2, linestyle='--')
axes[0, 0].plot(ann_history.history['loss'], label='ANN-LSTM Train', linewidth=2, alpha=0.7)
axes[0, 0].plot(ann_history.history['val_loss'], label='ANN-LSTM Val', linewidth=2, linestyle='--', alpha=0.7)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# SOA convergence
axes[0, 1].plot(soa_history, linewidth=2, color='green')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Best Fitness')
axes[0, 1].set_title('Seagull Optimization Convergence')
axes[0, 1].grid(True, alpha=0.3)

# Metrics comparison
metrics_data = [soann_results[m] for m in metrics]
ann_metrics_data = [ann_results[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

axes[1, 0].bar(x - width/2, metrics_data, width, label='SOANN-LSTM', color='steelblue')
axes[1, 0].bar(x + width/2, ann_metrics_data, width, label='ANN-LSTM', color='coral')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Performance Metrics Comparison')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels([m.capitalize() for m in metrics])
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Confusion matrices side by side
cm_soann = soann_results['confusion_matrix']
cm_ann = ann_results['confusion_matrix']

im1 = axes[1, 1].imshow(cm_soann, cmap='Blues', alpha=0.6)
axes[1, 1].set_title(f'SOANN Confusion Matrix\n(F1={soann_results["f1"]:.3f})')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_xticklabels(['No Failure', 'Failure'])
axes[1, 1].set_yticklabels(['No Failure', 'Failure'])

# Add text annotations
for i in range(2):
    for j in range(2):
        axes[1, 1].text(j, i, str(cm_soann[i, j]), ha='center', va='center', fontsize=14, fontweight='bold')

plt.colorbar(im1, ax=axes[1, 1])
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Comparison plot saved!")

print("\n" + "="*60)
print("✅ TRAINING AND COMPARISON COMPLETED!")
print("="*60)
print(f"\n🏆 Winner: {'SOANN-LSTM' if soann_results['f1'] > ann_results['f1'] else 'ANN-LSTM'}")
print(f"   F1 Score: {max(soann_results['f1'], ann_results['f1']):.4f}")


