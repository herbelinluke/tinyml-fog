/**
 * Velocity-Based TinyML Anomaly Detection Model
 * 
 * This model uses velocity and acceleration features to detect
 * fast-approaching objects vs normal movement.
 */

#include "ml_model.h"
#include <math.h>
#include <string.h>

// ============================================================================
// MODEL WEIGHTS (from velocity_model.h - trained on your data!)
// ============================================================================

// Layer 0: Input(6) -> Dense(12)
static const float layer0_weights[6][12] = {
  {0.387261f, -0.332460f, -0.292667f, 0.118109f, -0.032074f, -0.347523f, -0.440345f, -0.477795f, 0.218620f, -0.867517f, 0.080515f, 0.024464f},
  {-0.047505f, -0.330219f, 0.037214f, -0.087660f, -0.443055f, -0.409413f, -0.874368f, -0.487395f, -0.450714f, -0.263652f, -0.031044f, 0.055062f},
  {0.019224f, -0.098385f, -0.498280f, 0.033631f, 0.089105f, 0.482391f, -0.420444f, 0.597352f, 0.170925f, 0.524204f, 0.009185f, -0.147526f},
  {-0.575260f, -0.395804f, -0.144997f, -0.410547f, 0.803103f, 0.046980f, 0.386072f, -0.513782f, -0.041824f, 0.018926f, -0.549339f, -0.278441f},
  {1.575496f, 0.934077f, 0.028041f, 1.760804f, -1.371529f, 0.471924f, -0.293671f, 0.514026f, -1.475951f, -1.078016f, 1.750951f, -1.431085f},
  {-0.046556f, -0.326845f, -0.443476f, 0.009740f, 0.568812f, -0.230694f, -0.288209f, -0.119395f, 0.175485f, -0.013711f, 0.258301f, 0.107969f},
};
static const float layer0_bias[12] = {
  -0.108214f, -0.165671f, -0.242598f, -0.021455f, 0.112047f, -0.232315f, 0.227239f, -0.299683f, 0.150179f, 0.399915f, -0.166727f, -0.163087f
};

// Layer 1: Dense(12) -> Dense(6)
static const float layer1_weights[12][6] = {
  {1.088511f, 0.025872f, -1.047072f, 0.975989f, -0.624394f, -1.127654f},
  {0.145536f, 0.124924f, -0.188646f, 0.695775f, -0.624649f, -0.528294f},
  {-0.091962f, 0.430308f, 0.258699f, -0.230842f, -0.706717f, -0.520500f},
  {0.752731f, -0.212321f, -0.861784f, 0.789613f, -0.282005f, -0.575482f},
  {-0.122434f, -0.327441f, 0.116379f, -0.160292f, 0.800713f, -0.498646f},
  {0.480305f, -0.546920f, -0.196883f, 0.151248f, -0.438578f, -0.205371f},
  {-1.176893f, -0.046567f, 0.183903f, 0.393992f, 0.152227f, 0.559130f},
  {-0.345463f, -0.380870f, -0.682587f, 0.009564f, -0.630266f, 0.068485f},
  {-0.810781f, -0.454471f, 0.784011f, -0.454361f, 0.655426f, 0.158565f},
  {-0.322522f, -0.311877f, 0.719904f, 0.362692f, 0.486865f, 0.373095f},
  {0.588995f, -0.515867f, -0.863079f, 0.750727f, -0.835204f, -0.737857f},
  {-0.057193f, 0.153582f, 0.418910f, -0.106476f, -0.065761f, 0.613881f},
};
static const float layer1_bias[6] = {
  0.516955f, -0.151833f, 0.649389f, 0.202238f, 0.603951f, 0.598122f
};

// Layer 2: Dense(6) -> Dense(1, sigmoid)
static const float layer2_weights[6][1] = {
  {-1.286565f},
  {0.997193f},
  {1.037104f},
  {-1.140760f},
  {1.450165f},
  {0.519883f},
};
static const float layer2_bias[1] = {
  -0.010462f
};

// ============================================================================
// HISTORY BUFFER
// ============================================================================

static float history[WINDOW_SIZE];
static int history_index = 0;
static int history_count = 0;

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ============================================================================
// FEATURE COMPUTATION
// ============================================================================

static void compute_features(float* features) {
    // Get current distance
    int current_idx = (history_index - 1 + WINDOW_SIZE) % WINDOW_SIZE;
    float current = history[current_idx];
    
    // Compute velocities (differences between consecutive readings)
    float velocities[WINDOW_SIZE - 1];
    for (int i = 0; i < history_count - 1; i++) {
        int idx1 = (history_index - history_count + i + WINDOW_SIZE) % WINDOW_SIZE;
        int idx2 = (idx1 + 1) % WINDOW_SIZE;
        velocities[i] = history[idx2] - history[idx1];
    }
    
    // Current velocity (most recent change)
    float velocity = 0.0f;
    if (history_count >= 2) {
        int prev_idx = (current_idx - 1 + WINDOW_SIZE) % WINDOW_SIZE;
        velocity = current - history[prev_idx];
    }
    
    // Acceleration (change in velocity)
    float acceleration = 0.0f;
    if (history_count >= 3) {
        int prev_idx = (current_idx - 1 + WINDOW_SIZE) % WINDOW_SIZE;
        int prev2_idx = (current_idx - 2 + WINDOW_SIZE) % WINDOW_SIZE;
        float prev_velocity = history[prev_idx] - history[prev2_idx];
        acceleration = velocity - prev_velocity;
    }
    
    // Compute statistics over the window
    float sum = 0.0f;
    float min_val = history[0];
    float max_val = history[0];
    
    for (int i = 0; i < history_count; i++) {
        int idx = (history_index - history_count + i + WINDOW_SIZE) % WINDOW_SIZE;
        float val = history[idx];
        sum += val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    
    float mean = sum / history_count;
    
    // Compute variance
    float variance = 0.0f;
    for (int i = 0; i < history_count; i++) {
        int idx = (history_index - history_count + i + WINDOW_SIZE) % WINDOW_SIZE;
        float diff = history[idx] - mean;
        variance += diff * diff;
    }
    variance = sqrtf(variance / history_count);  // Standard deviation
    
    // Build normalized feature vector
    features[0] = current / NORMALIZE_DISTANCE;
    features[1] = velocity / NORMALIZE_VELOCITY;
    features[2] = acceleration / NORMALIZE_VELOCITY;
    features[3] = variance / NORMALIZE_DISTANCE;
    features[4] = min_val / NORMALIZE_DISTANCE;
    features[5] = max_val / NORMALIZE_DISTANCE;
}

// ============================================================================
// PUBLIC API
// ============================================================================

void ml_model_init(void) {
    memset(history, 0, sizeof(history));
    history_index = 0;
    history_count = 0;
}

void ml_model_add_reading(float distance) {
    history[history_index] = distance;
    history_index = (history_index + 1) % WINDOW_SIZE;
    if (history_count < WINDOW_SIZE) {
        history_count++;
    }
}

int ml_model_is_ready(void) {
    return history_count >= 3;  // Need at least 3 readings for velocity/acceleration
}

float ml_model_predict(void) {
    if (!ml_model_is_ready()) {
        return 0.0f;
    }
    
    // Compute features
    float features[FEATURE_COUNT];
    compute_features(features);
    
    // Layer 0: Input -> Hidden1 (Dense + ReLU)
    float hidden1[HIDDEN1_SIZE];
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
        float sum = layer0_bias[j];
        for (int i = 0; i < FEATURE_COUNT; i++) {
            sum += features[i] * layer0_weights[i][j];
        }
        hidden1[j] = relu(sum);
    }
    
    // Layer 1: Hidden1 -> Hidden2 (Dense + ReLU)
    float hidden2[HIDDEN2_SIZE];
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
        float sum = layer1_bias[j];
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            sum += hidden1[i] * layer1_weights[i][j];
        }
        hidden2[j] = relu(sum);
    }
    
    // Layer 2: Hidden2 -> Output (Dense + Sigmoid)
    float output = layer2_bias[0];
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        output += hidden2[i] * layer2_weights[i][0];
    }
    output = sigmoid(output);
    
    return output;
}

int ml_model_is_anomaly(float probability) {
    return probability > ANOMALY_THRESHOLD ? 1 : 0;
}
