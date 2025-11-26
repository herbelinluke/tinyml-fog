/**
 * Arduino Sensor Node
 * 
 * Simple sensor reader that sends distance readings to Pico 2 for ML processing.
 * This demonstrates distributed computing where:
 *   - Arduino = Sensor interface (dumb node)
 *   - Pico 2  = ML processor (smart node)
 * 
 * Wiring:
 *   HC-SR04 VCC  -> Arduino 5V
 *   HC-SR04 GND  -> Arduino GND (shared with Pico GND!)
 *   HC-SR04 TRIG -> Arduino D8
 *   HC-SR04 ECHO -> Arduino D9
 *   Arduino TX   -> Pico GP1 (UART0 RX)
 *   Arduino GND  -> Pico GND
 * 
 * Output format: Just the distance value followed by newline
 *   Example: "324\n"
 */

const int trigPin = 8;
const int echoPin = 9;

void setup() {
    pinMode(trigPin, OUTPUT);
    pinMode(echoPin, INPUT);
    
    // Serial for sending to Pico (and can also be monitored via USB)
    Serial.begin(9600);
    
    Serial.println("# Arduino Sensor Node");
    Serial.println("# Sending distance readings to Pico...");
}

long measureDistance() {
    // Clear trigger
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    
    // Send trigger pulse
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    
    // Measure echo
    long duration = pulseIn(echoPin, HIGH, 30000);  // 30ms timeout
    
    // Calculate distance in cm
    // Speed of sound = 343 m/s = 0.0343 cm/µs
    // Distance = (time * speed) / 2 (round trip)
    long distance = (duration * 0.034) / 2;
    
    return distance;
}

void loop() {
    long distance = measureDistance();
    
    // Only send valid readings
    if (distance > 0 && distance < 400) {
        Serial.println(distance);
    }
    
    // ~10Hz sample rate
    delay(100);
}

