/**
 * TinyML Anomaly Detection Model Implementation
 * 
 * Manually implemented neural network for Raspberry Pi Pico 2.
 * This avoids the complexity of TFLite Micro while still running real ML.
 */

#include "ml_model.h"
#include <math.h>

// ============================================================================
// MODEL WEIGHTS (extracted from trained Keras model)
// ============================================================================

// Layer 0: Input(10) -> Dense(8)
static const float layer0_weights[10][8] = {
  {-0.547934f, 0.368171f, -0.310684f, 0.551349f, -0.126685f, 0.208559f, -0.365375f, 0.075177f},
  {-0.312322f, 0.496048f, 0.209948f, -0.111037f, -0.207391f, -0.339548f, 0.203184f, 0.118198f},
  {0.570218f, 0.561503f, 0.159296f, 0.459575f, 0.293842f, 0.283696f, 0.415165f, 0.005862f},
  {-0.368935f, 0.004170f, 0.538589f, -0.269605f, 0.511725f, -0.485512f, 0.010571f, -0.487563f},
  {0.505109f, -0.574965f, 0.164953f, 0.394082f, 0.010531f, -0.552365f, -0.334434f, -0.065905f},
  {0.359542f, -0.405387f, 0.180085f, 0.168944f, -0.074708f, -0.110911f, -0.155789f, -0.524581f},
  {-0.420307f, 0.426860f, 0.557613f, 0.255339f, 0.287399f, -0.422156f, -0.430643f, -0.270856f},
  {-0.461581f, -0.058715f, 0.370266f, -0.147029f, 0.213512f, 0.144605f, 0.513596f, 0.038397f},
  {0.255745f, 0.473398f, 0.000395f, -0.071642f, -0.273866f, 0.473182f, -0.033184f, 0.304584f},
  {-0.046502f, 0.243109f, -0.118775f, 0.308159f, 0.382390f, -0.548670f, -0.434916f, 0.271029f},
};
static const float layer0_bias[8] = {
  0.000402f, -0.012709f, 0.012689f, -0.012712f, 0.012704f, 0.000000f, 0.003373f, 0.003100f
};

// Layer 1: Dense(8) -> Dense(4)
static const float layer1_weights[8][4] = {
  {-0.124593f, 0.070006f, 0.520561f, -0.550806f},
  {-0.402958f, -0.593736f, 0.648580f, -0.342894f},
  {-0.639053f, -0.273512f, -0.106319f, -0.213241f},
  {-0.527721f, -0.682552f, 0.315310f, -0.021264f},
  {0.195438f, -0.102784f, -0.556783f, -0.266771f},
  {-0.179854f, 0.523734f, -0.205138f, -0.245908f},
  {-0.476619f, 0.576454f, -0.256109f, 0.113325f},
  {-0.706053f, -0.337521f, 0.028793f, 0.599764f},
};
static const float layer1_bias[4] = {
  0.000000f, 0.000000f, -0.012707f, 0.000000f
};

// Layer 2: Dense(4) -> Dense(1, sigmoid)
static const float layer2_weights[4][1] = {
  {0.607788f},
  {-1.008546f},
  {-0.881951f},
  {0.279320f},
};
static const float layer2_bias[1] = {
  0.012666f
};

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
// INFERENCE
// ============================================================================

void ml_model_init(void) {
    // No initialization needed for this simple model
    // Weights are compiled into the binary
}

float ml_model_predict(const float* input) {
    float normalized[INPUT_SIZE];
    float hidden1[HIDDEN1_SIZE];
    float hidden2[HIDDEN2_SIZE];
    float output;
    
    // Normalize input
    for (int i = 0; i < INPUT_SIZE; i++) {
        normalized[i] = input[i] / NORMALIZE_MAX;
    }
    
    // Layer 0: Input -> Hidden1 (Dense + ReLU)
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
        float sum = layer0_bias[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += normalized[i] * layer0_weights[i][j];
        }
        hidden1[j] = relu(sum);
    }
    
    // Layer 1: Hidden1 -> Hidden2 (Dense + ReLU)
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
        float sum = layer1_bias[j];
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            sum += hidden1[i] * layer1_weights[i][j];
        }
        hidden2[j] = relu(sum);
    }
    
    // Layer 2: Hidden2 -> Output (Dense + Sigmoid)
    output = layer2_bias[0];
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        output += hidden2[i] * layer2_weights[i][0];
    }
    output = sigmoid(output);
    
    return output;
}

int ml_model_is_anomaly(float probability) {
    return probability > ANOMALY_THRESHOLD ? 1 : 0;
}

