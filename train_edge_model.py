#!/usr/bin/env python3
"""
Train a TinyML model for edge anomaly detection.

This creates a very small neural network that can run on a Raspberry Pi Pico 2.
The model takes a window of recent sensor readings and predicts if the current
reading is anomalous.

Output:
- anomaly_model.tflite (quantized model for Pico)
- anomaly_model.h (C header for embedding in firmware)
- Training metrics and plots
"""

import numpy as np
import pandas as pd
import os

# Check for TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed. Install with:")
    print("  pip install tensorflow")
    exit(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - Adjust these parameters
# ============================================================================

WINDOW_SIZE = 10          # Number of past readings to use as input
NORMALIZE_MAX = 400.0     # Max sensor distance for normalization
TEST_SPLIT = 0.2          # Fraction of data for testing
EPOCHS = 100              # Training epochs
BATCH_SIZE = 32           # Batch size

# Model architecture (keep small for MCU!)
HIDDEN_UNITS_1 = 8        # First hidden layer
HIDDEN_UNITS_2 = 4        # Second hidden layer

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess(csv_path: str):
    """Load CSV and create windowed sequences."""
    print(f"\n📂 Loading data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   Total samples: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Handle different column names
    if 'Distance' in df.columns:
        distance_col = 'Distance'
        anomaly_col = 'Anomaly'
    elif 'SensorValue' in df.columns:
        distance_col = 'SensorValue'
        anomaly_col = 'Label'
    else:
        raise ValueError(f"Unknown column format: {df.columns}")
    
    distances = df[distance_col].values.astype(np.float32)
    labels = df[anomaly_col].values.astype(np.int32)
    
    print(f"   Anomalies: {np.sum(labels)} ({100*np.mean(labels):.1f}%)")
    print(f"   Distance range: {distances.min():.0f} - {distances.max():.0f}")
    
    # Create windowed sequences
    X, y = [], []
    for i in range(WINDOW_SIZE, len(distances)):
        window = distances[i-WINDOW_SIZE:i]
        X.append(window)
        y.append(labels[i])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Normalize to 0-1 range
    X = X / NORMALIZE_MAX
    
    print(f"   Created {len(X)} sequences of length {WINDOW_SIZE}")
    
    return X, y

# ============================================================================
# MODEL DEFINITION
# ============================================================================

def create_model():
    """Create a tiny neural network for anomaly detection."""
    model = tf.keras.Sequential([
        # Input: window of normalized distance readings
        tf.keras.layers.InputLayer(input_shape=(WINDOW_SIZE,)),
        
        # Hidden layers (keep small!)
        tf.keras.layers.Dense(HIDDEN_UNITS_1, activation='relu'),
        tf.keras.layers.Dense(HIDDEN_UNITS_2, activation='relu'),
        
        # Output: probability of anomaly
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def print_model_summary(model):
    """Print model architecture and size estimate."""
    print("\n🏗️  Model Architecture:")
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    # Rough size estimate: 4 bytes per float32 param, 1 byte per int8 quantized
    float_size = total_params * 4
    quant_size = total_params * 1
    
    print(f"\n   Total parameters: {total_params}")
    print(f"   Float32 size: ~{float_size/1024:.1f} KB")
    print(f"   Quantized size: ~{quant_size/1024:.1f} KB (estimated)")

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with early stopping."""
    print("\n🎯 Training model...")
    
    # Handle class imbalance with class weights
    n_normal = np.sum(y_train == 0)
    n_anomaly = np.sum(y_train == 1)
    class_weight = {0: 1.0, 1: n_normal / n_anomaly}
    print(f"   Class weights: normal=1.0, anomaly={class_weight[1]:.1f}")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    print("\n📊 Evaluation Results:")
    
    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"   FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    return y_pred_prob

# ============================================================================
# TFLITE CONVERSION
# ============================================================================

def convert_to_tflite(model, X_train, output_path='anomaly_model.tflite'):
    """Convert model to quantized TFLite format."""
    print(f"\n🔄 Converting to TFLite...")
    
    # Representative dataset for quantization
    def representative_dataset():
        for i in range(min(100, len(X_train))):
            yield [X_train[i:i+1]]
    
    # Convert with full integer quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"   Full int8 failed, trying float16: {e}")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
    
    # Save model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"   Saved: {output_path}")
    print(f"   Size: {size_kb:.2f} KB")
    
    return tflite_model

def convert_to_c_header(tflite_model, output_path='anomaly_model.h'):
    """Convert TFLite model to C header for embedding in firmware."""
    print(f"\n📝 Generating C header...")
    
    # Convert bytes to C array
    hex_lines = []
    for i in range(0, len(tflite_model), 12):
        chunk = tflite_model[i:i+12]
        hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
        hex_lines.append(f'  {hex_str},')
    
    c_code = f'''// Auto-generated TFLite model
// Model size: {len(tflite_model)} bytes
// Window size: {WINDOW_SIZE}
// Normalization: divide by {NORMALIZE_MAX}

#ifndef ANOMALY_MODEL_H
#define ANOMALY_MODEL_H

const unsigned char anomaly_model[] = {{
{chr(10).join(hex_lines)}
}};

const unsigned int anomaly_model_len = {len(tflite_model)};

// Input: {WINDOW_SIZE} normalized float values (0-1)
// Output: anomaly probability (0-1)

#endif // ANOMALY_MODEL_H
'''
    
    with open(output_path, 'w') as f:
        f.write(c_code)
    
    print(f"   Saved: {output_path}")

# ============================================================================
# VERIFY TFLITE MODEL
# ============================================================================

def verify_tflite(tflite_path, X_test, y_test):
    """Verify TFLite model produces similar results."""
    print(f"\n✅ Verifying TFLite model...")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input dtype: {input_details[0]['dtype']}")
    print(f"   Output dtype: {output_details[0]['dtype']}")
    
    # Get quantization parameters
    input_scale = input_details[0].get('quantization_parameters', {}).get('scales', [1.0])
    input_zero = input_details[0].get('quantization_parameters', {}).get('zero_points', [0])
    
    # Test a few samples
    correct = 0
    for i in range(min(50, len(X_test))):
        input_data = X_test[i:i+1]
        
        # Quantize if needed
        if input_details[0]['dtype'] == np.int8:
            input_data = (input_data / input_scale[0] + input_zero[0]).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize if needed
        if output_details[0]['dtype'] == np.int8:
            output_scale = output_details[0].get('quantization_parameters', {}).get('scales', [1.0])
            output_zero = output_details[0].get('quantization_parameters', {}).get('zero_points', [0])
            output_data = (output_data.astype(np.float32) - output_zero[0]) * output_scale[0]
        
        pred = 1 if output_data[0] > 0.5 else 0
        if pred == y_test[i]:
            correct += 1
    
    accuracy = correct / min(50, len(X_test))
    print(f"   TFLite accuracy (50 samples): {accuracy*100:.1f}%")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("🧠 TinyML Anomaly Detection - Model Training")
    print("=" * 60)
    
    # Find dataset
    for csv_path in ['sensor_data.csv', 'anomoly_data.csv']:
        if os.path.exists(csv_path):
            break
    else:
        print("❌ No dataset found! Run collect.py first.")
        return
    
    # Load data
    X, y = load_and_preprocess(csv_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\n📊 Data splits:")
    print(f"   Train: {len(X_train)} samples ({np.sum(y_train)} anomalies)")
    print(f"   Val:   {len(X_val)} samples ({np.sum(y_val)} anomalies)")
    print(f"   Test:  {len(X_test)} samples ({np.sum(y_test)} anomalies)")
    
    # Create and train model
    model = create_model()
    print_model_summary(model)
    
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Save Keras model
    model.save('anomaly_model.keras')
    print("\n💾 Saved Keras model: anomaly_model.keras")
    
    # Convert to TFLite
    tflite_model = convert_to_tflite(model, X_train)
    
    # Generate C header
    convert_to_c_header(tflite_model)
    
    # Verify TFLite
    verify_tflite('anomaly_model.tflite', X_test, y_test)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Copy anomaly_model.h to your Pico 2 project")
    print("2. Use TFLite Micro to run inference")
    print(f"3. Remember: normalize inputs by dividing by {NORMALIZE_MAX}")


if __name__ == '__main__':
    main()

