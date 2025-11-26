#!/usr/bin/env python3
"""
Train an improved TinyML model that detects fast-approaching objects.

This model uses velocity and acceleration features to distinguish between:
- Normal: Slow movements, people walking around
- Anomaly: Fast approaches toward the sensor

Features:
1. Current distance (normalized)
2. Velocity (rate of change)
3. Acceleration (rate of velocity change)
4. Distance variance (stability)
5. Min distance in window
6. Max distance in window
"""

import numpy as np
import pandas as pd
import os

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("Install TensorFlow: pip install tensorflow")
    exit(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================================
# CONFIGURATION
# ============================================================================

WINDOW_SIZE = 10          # Readings to look at
FEATURE_COUNT = 6         # Number of features per sample
NORMALIZE_DISTANCE = 400.0
NORMALIZE_VELOCITY = 50.0  # Max expected velocity (cm per reading)
TEST_SPLIT = 0.2
EPOCHS = 150
BATCH_SIZE = 32

# Thresholds for velocity-based anomaly detection
VELOCITY_THRESHOLD = -15.0  # Fast approach = negative velocity

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_features(distances):
    """
    Compute motion features from a window of distance readings.
    
    Returns: [distance, velocity, acceleration, variance, min, max]
    """
    if len(distances) < 3:
        return None
    
    current = distances[-1]
    
    # Velocity: change from previous reading (negative = approaching)
    velocities = np.diff(distances)
    velocity = velocities[-1] if len(velocities) > 0 else 0
    
    # Acceleration: change in velocity
    if len(velocities) >= 2:
        acceleration = velocities[-1] - velocities[-2]
    else:
        acceleration = 0
    
    # Variance: how stable is the distance?
    variance = np.std(distances)
    
    # Min/Max in window
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    return np.array([
        current / NORMALIZE_DISTANCE,           # Normalized distance
        velocity / NORMALIZE_VELOCITY,          # Normalized velocity
        acceleration / NORMALIZE_VELOCITY,      # Normalized acceleration
        variance / NORMALIZE_DISTANCE,          # Normalized variance
        min_dist / NORMALIZE_DISTANCE,          # Normalized min
        max_dist / NORMALIZE_DISTANCE,          # Normalized max
    ], dtype=np.float32)

def create_feature_dataset(df):
    """Create features from raw distance data."""
    distances = df['Distance'].values.astype(np.float32)
    labels = df['Label'].values.astype(np.int32)
    
    X, y = [], []
    
    for i in range(WINDOW_SIZE, len(distances)):
        window = distances[i-WINDOW_SIZE:i+1]  # Include current reading
        features = compute_features(window)
        
        if features is not None:
            X.append(features)
            y.append(labels[i])
    
    return np.array(X), np.array(y)

# ============================================================================
# MODEL
# ============================================================================

def create_model():
    """Create a model that uses velocity/acceleration features."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(FEATURE_COUNT,)),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# SYNTHETIC DATA GENERATION (for testing without real data)
# ============================================================================

def generate_synthetic_data(n_samples=2000):
    """Generate synthetic training data if no real data exists."""
    print("Generating synthetic training data...")
    
    data = []
    
    # Normal scenarios: slow random walk
    for _ in range(n_samples // 2):
        base = np.random.uniform(100, 300)
        distances = base + np.cumsum(np.random.randn(20) * 2)  # Small changes
        distances = np.clip(distances, 10, 400)
        for d in distances:
            data.append({'Distance': d, 'Label': 0})
    
    # Anomaly scenarios: fast approach
    for _ in range(n_samples // 4):
        # Start far, quickly approach
        start = np.random.uniform(200, 350)
        end = np.random.uniform(20, 80)
        steps = np.random.randint(5, 15)
        distances = np.linspace(start, end, steps)
        for d in distances:
            data.append({'Distance': d, 'Label': 1})
    
    # More normal: stationary
    for _ in range(n_samples // 4):
        base = np.random.uniform(50, 300)
        distances = base + np.random.randn(15) * 3  # Very small noise
        for d in distances:
            data.append({'Distance': d, 'Label': 0})
    
    df = pd.DataFrame(data)
    df['Timestamp'] = range(len(df))
    return df

# ============================================================================
# EXPORT FOR PICO
# ============================================================================

def export_for_pico(model):
    """Export model weights as C code for Pico."""
    
    weights = []
    for layer in model.layers:
        w = layer.get_weights()
        if w:
            weights.append(w)
    
    code = '''/**
 * Velocity-based Anomaly Detection Model
 * 
 * Features: [distance, velocity, acceleration, variance, min, max]
 * All normalized to 0-1 range
 */

#ifndef VELOCITY_MODEL_H
#define VELOCITY_MODEL_H

#define FEATURE_COUNT 6
#define NORMALIZE_DISTANCE 400.0f
#define NORMALIZE_VELOCITY 50.0f

'''
    
    for i, (w, b) in enumerate(weights):
        code += f"// Layer {i}: {w.shape}\n"
        code += f"static const float layer{i}_weights[{w.shape[0]}][{w.shape[1]}] = {{\n"
        for row in w:
            code += "  {" + ", ".join(f"{v:.6f}f" for v in row) + "},\n"
        code += "};\n"
        code += f"static const float layer{i}_bias[{b.shape[0]}] = {{\n"
        code += "  " + ", ".join(f"{v:.6f}f" for v in b) + "\n"
        code += "};\n\n"
    
    code += "#endif // VELOCITY_MODEL_H\n"
    
    with open('velocity_model.h', 'w') as f:
        f.write(code)
    
    print("Exported: velocity_model.h")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("🚀 Velocity-Based Anomaly Detection Training")
    print("=" * 60)
    
    # Load or generate data
    if os.path.exists('training_data.csv'):
        print("\n📂 Loading training_data.csv...")
        df = pd.read_csv('training_data.csv')
    else:
        print("\n⚠️  No training_data.csv found!")
        print("   Run: python collect_training_data.py")
        print("   Or generating synthetic data for testing...\n")
        df = generate_synthetic_data()
        df.to_csv('synthetic_training_data.csv', index=False)
    
    print(f"   Total samples: {len(df)}")
    print(f"   Normal: {(df['Label'] == 0).sum()}")
    print(f"   Anomaly: {(df['Label'] == 1).sum()}")
    
    # Create features
    print("\n🔧 Computing motion features...")
    X, y = create_feature_dataset(df)
    print(f"   Feature shape: {X.shape}")
    print(f"   Sample features: {X[0]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    n_normal = (y_train == 0).sum()
    n_anomaly = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: max(1.0, n_normal / max(n_anomaly, 1))}
    print(f"\n   Class weights: {class_weight}")
    
    # Train
    print("\n🎯 Training model...")
    model = create_model()
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
        ],
        verbose=1
    )
    
    # Evaluate
    print("\n📊 Evaluation:")
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    # Save
    model.save('velocity_model.keras')
    print("\n💾 Saved: velocity_model.keras")
    
    # Export for Pico
    export_for_pico(model)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print("\nTo use on Pico:")
    print("1. Update ml_model.cpp with velocity_model.h weights")
    print("2. Modify feature computation to include velocity/acceleration")

if __name__ == '__main__':
    main()

