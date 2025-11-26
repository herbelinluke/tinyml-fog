/**
 * Pico 2 ML Node - TinyML Anomaly Detection
 * 
 * This node receives sensor readings from an Arduino (or directly reads a sensor)
 * and runs ML inference to detect anomalies. Results are sent to the fog node.
 * 
 * Communication:
 * - UART0 (GP0/GP1): Receives data from Arduino: "distance\n"
 * - USB CDC: Sends results to fog node: "pico_ml,distance,anomaly,confidence\n"
 * 
 * Wiring for Arduino -> Pico communication:
 *   Arduino TX  -> Pico GP1 (UART0 RX)
 *   Arduino GND -> Pico GND
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/uart.h"
#include "ml_model.h"

// ============================================================================
// CONFIGURATION
// ============================================================================

#define UART_ID uart0
#define BAUD_RATE 9600
#define UART_TX_PIN 0
#define UART_RX_PIN 1

#define LED_PIN PICO_DEFAULT_LED_PIN

// Reading buffer for ML input
#define BUFFER_SIZE INPUT_SIZE
static float reading_buffer[BUFFER_SIZE];
static int buffer_index = 0;
static int buffer_filled = 0;

// UART receive buffer
#define RX_BUFFER_SIZE 32
static char rx_buffer[RX_BUFFER_SIZE];
static int rx_index = 0;

// ============================================================================
// LED INDICATOR
// ============================================================================

static void led_init(void) {
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
}

static void led_blink(int times, int delay_ms) {
    for (int i = 0; i < times; i++) {
        gpio_put(LED_PIN, 1);
        sleep_ms(delay_ms);
        gpio_put(LED_PIN, 0);
        sleep_ms(delay_ms);
    }
}

// ============================================================================
// UART COMMUNICATION (from Arduino)
// ============================================================================

static void uart_init_custom(void) {
    uart_init(UART_ID, BAUD_RATE);
    gpio_set_function(UART_TX_PIN, GPIO_FUNC_UART);
    gpio_set_function(UART_RX_PIN, GPIO_FUNC_UART);
    
    // Set UART flow control CTS/RTS, we don't want these
    uart_set_hw_flow(UART_ID, false, false);
    uart_set_format(UART_ID, 8, 1, UART_PARITY_NONE);
    uart_set_fifo_enabled(UART_ID, false);
}

static int process_uart_char(char c) {
    if (c == '\n' || c == '\r') {
        if (rx_index > 0) {
            rx_buffer[rx_index] = '\0';
            int distance = atoi(rx_buffer);
            rx_index = 0;
            return distance;
        }
    } else if (rx_index < RX_BUFFER_SIZE - 1) {
        rx_buffer[rx_index++] = c;
    }
    return -1;  // No complete reading yet
}

// ============================================================================
// ML INFERENCE
// ============================================================================

static void add_reading(float distance) {
    reading_buffer[buffer_index] = distance;
    buffer_index = (buffer_index + 1) % BUFFER_SIZE;
    
    if (!buffer_filled && buffer_index == 0) {
        buffer_filled = 1;
    }
}

static void run_inference(float distance) {
    if (!buffer_filled) {
        // Not enough data yet, just output raw reading
        printf("pico_ml,%d,0,0.00\n", (int)distance);
        return;
    }
    
    // Create input array in correct order (oldest to newest)
    float input[BUFFER_SIZE];
    for (int i = 0; i < BUFFER_SIZE; i++) {
        int idx = (buffer_index + i) % BUFFER_SIZE;
        input[i] = reading_buffer[idx];
    }
    
    // Run ML inference
    float confidence = ml_model_predict(input);
    int anomaly = ml_model_is_anomaly(confidence);
    
    // Output: node_id,distance,anomaly,confidence
    printf("pico_ml,%d,%d,%.2f\n", (int)distance, anomaly, confidence);
    
    // Visual feedback
    if (anomaly) {
        led_blink(3, 50);  // Fast blink for anomaly
    } else {
        gpio_put(LED_PIN, 1);
        sleep_ms(10);
        gpio_put(LED_PIN, 0);
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    // Initialize stdio for USB output
    stdio_init_all();
    
    // Wait for USB connection
    sleep_ms(2000);
    
    // Initialize components
    led_init();
    uart_init_custom();
    ml_model_init();
    
    // Startup indication
    led_blink(5, 100);
    printf("# Pico ML Node started\n");
    printf("# Waiting for sensor data on UART0...\n");
    printf("# Output format: pico_ml,distance,anomaly,confidence\n");
    
    // Main loop
    while (true) {
        // Check for data from Arduino via UART
        while (uart_is_readable(UART_ID)) {
            char c = uart_getc(UART_ID);
            int distance = process_uart_char(c);
            
            if (distance >= 0) {
                // Got a complete reading
                add_reading((float)distance);
                run_inference((float)distance);
            }

        }
        
        // Small delay to prevent busy loop
        sleep_ms(1);
    }
    
    return 0;
}


