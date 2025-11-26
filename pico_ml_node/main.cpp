/**
 * Pico 2 ML Node - Velocity-Based Anomaly Detection
 * 
 * This node receives sensor readings from an Arduino via UART
 * and uses ML to detect fast-approaching objects.
 * 
 * The model analyzes velocity and acceleration to distinguish between:
 * - Normal: People walking, slow movements
 * - Anomaly: Fast approaching objects
 * 
 * Communication:
 * - UART0 (GP1 RX): Receives "distance\n" from Arduino
 * - USB CDC: Sends "pico_ml,distance,anomaly,confidence\n" to fog node
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

#define LED_PIN 25  // Built-in LED

// UART receive buffer
#define RX_BUFFER_SIZE 32
static char rx_buffer[RX_BUFFER_SIZE];
static int rx_index = 0;

// Statistics
static uint32_t readings_count = 0;
static uint32_t anomaly_count = 0;

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
// UART COMMUNICATION
// ============================================================================

static void uart_init_custom(void) {
    uart_init(UART_ID, BAUD_RATE);
    gpio_set_function(UART_TX_PIN, GPIO_FUNC_UART);
    gpio_set_function(UART_RX_PIN, GPIO_FUNC_UART);
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
    } else if (rx_index < RX_BUFFER_SIZE - 1 && c >= '0' && c <= '9') {
        rx_buffer[rx_index++] = c;
    }
    return -1;
}

// ============================================================================
// ML INFERENCE
// ============================================================================

static void process_reading(int distance) {
    readings_count++;
    
    // Add reading to model history
    ml_model_add_reading((float)distance);
    
    // Check if we have enough data
    if (!ml_model_is_ready()) {
        printf("pico_ml,%d,0,0.00,buffering\n", distance);
        gpio_put(LED_PIN, 1);
        sleep_ms(10);
        gpio_put(LED_PIN, 0);
        return;
    }
    
    // Run inference
    float confidence = ml_model_predict();
    int anomaly = ml_model_is_anomaly(confidence);
    
    if (anomaly) {
        anomaly_count++;
    }
    
    // Output: node_id,distance,anomaly,confidence
    printf("pico_ml,%d,%d,%.2f\n", distance, anomaly, confidence);
    
    // LED feedback
    if (anomaly) {
        // Fast blink for anomaly
        led_blink(3, 30);
    } else {
        // Quick flash for normal
        gpio_put(LED_PIN, 1);
        sleep_ms(5);
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
    
    printf("\n");
    printf("========================================\n");
    printf("  Pico 2 ML Node - Velocity Detection\n");
    printf("========================================\n");
    printf("  UART: GP1 (RX) at 9600 baud\n");
    printf("  Model: Velocity-based anomaly detector\n");
    printf("  Features: distance, velocity, accel, var, min, max\n");
    printf("  Waiting for sensor data...\n");
    printf("========================================\n\n");
    
    // Main loop
    while (true) {
        // Check for UART data from Arduino
        while (uart_is_readable(UART_ID)) {
            char c = uart_getc(UART_ID);
            int distance = process_uart_char(c);
            
            if (distance >= 0 && distance < 500) {
                process_reading(distance);
            }
        }
        
        sleep_ms(1);
    }
    
    return 0;
}
