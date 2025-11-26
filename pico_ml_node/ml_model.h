/**
 * TinyML Anomaly Detection Model
 * 
 * This is a manually implemented neural network based on the trained model.
 * Architecture: Input(10) -> Dense(8, relu) -> Dense(4, relu) -> Dense(1, sigmoid)
 * 
 * The weights are extracted from the trained Keras model.
 */

#ifndef ML_MODEL_H
#define ML_MODEL_H

#include <stdint.h>

// Model configuration
#define INPUT_SIZE 10
#define HIDDEN1_SIZE 8
#define HIDDEN2_SIZE 4
#define OUTPUT_SIZE 1

// Normalization constant (must match training)
#define NORMALIZE_MAX 400.0f

// Anomaly threshold
#define ANOMALY_THRESHOLD 0.5f

/**
 * Initialize the ML model (loads weights)
 */
void ml_model_init(void);

/**
 * Run inference on a window of sensor readings
 * 
 * @param input Array of INPUT_SIZE raw distance readings (will be normalized internally)
 * @return Anomaly probability (0.0 - 1.0)
 */
float ml_model_predict(const float* input);

/**
 * Check if prediction indicates anomaly
 * 
 * @param probability Output from ml_model_predict
 * @return 1 if anomaly, 0 if normal
 */
int ml_model_is_anomaly(float probability);

#endif // ML_MODEL_H

