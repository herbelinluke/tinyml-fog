/**
 * Velocity-Based TinyML Anomaly Detection Model
 * 
 * This model detects fast-approaching objects by analyzing:
 * - Velocity (rate of distance change)
 * - Acceleration (rate of velocity change)
 * - Distance variance (stability)
 * 
 * Features: [distance, velocity, acceleration, variance, min, max]
 */

#ifndef ML_MODEL_H
#define ML_MODEL_H

#include <stdint.h>

// Model configuration
#define WINDOW_SIZE 10            // Number of readings to keep in history
#define FEATURE_COUNT 6           // Number of computed features
#define HIDDEN1_SIZE 12           // First hidden layer
#define HIDDEN2_SIZE 6            // Second hidden layer
#define OUTPUT_SIZE 1

// Normalization constants (must match training)
#define NORMALIZE_DISTANCE 400.0f
#define NORMALIZE_VELOCITY 50.0f

// Anomaly threshold (higher = less sensitive, fewer false positives)
// 0.5 = very sensitive, 0.7 = balanced, 0.85 = conservative
#define ANOMALY_THRESHOLD 0.7f

/**
 * Initialize the ML model
 */
void ml_model_init(void);

/**
 * Add a new distance reading to the history buffer
 * 
 * @param distance Raw distance reading in cm
 */
void ml_model_add_reading(float distance);

/**
 * Run inference on the current history buffer
 * 
 * @return Anomaly probability (0.0 - 1.0)
 */
float ml_model_predict(void);

/**
 * Check if we have enough readings to make predictions
 * 
 * @return 1 if buffer is ready, 0 otherwise
 */
int ml_model_is_ready(void);

/**
 * Check if prediction indicates anomaly
 * 
 * @param probability Output from ml_model_predict
 * @return 1 if anomaly, 0 if normal
 */
int ml_model_is_anomaly(float probability);

#endif // ML_MODEL_H
