// HC-SR04 with Rolling Average Anomaly Detection
// Output format: distance,anomaly_flag (e.g., "324,0" or "73,1")

const int trigPin = 8;
const int echoPin = 9;

// Rolling average configuration
const int WINDOW_SIZE = 10;
long readings[WINDOW_SIZE];
int readIndex = 0;
long total = 0;
int validReadings = 0;

// Anomaly detection thresholds
const float DEVIATION_THRESHOLD = 0.30;  // 30% deviation from rolling average
const long MIN_VALID_DISTANCE = 2;       // Minimum valid distance in cm
const long MAX_VALID_DISTANCE = 400;     // Maximum valid distance in cm

long duration;

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  Serial.begin(9600);
  
  // Initialize readings array
  for (int i = 0; i < WINDOW_SIZE; i++) {
    readings[i] = 0;
  }
}

long measureDistance() {
  // Clear the trigPin
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // Trigger the sensor
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Measure echo duration
  duration = pulseIn(echoPin, HIGH);

  // Calculate distance in cm
  long distance = (duration * 0.034) / 2;
  return distance;
}

float getRollingAverage() {
  if (validReadings == 0) return 0;
  return (float)total / validReadings;
}

void updateRollingAverage(long newReading) {
  // Subtract oldest reading from total
  total -= readings[readIndex];
  
  // Add new reading
  readings[readIndex] = newReading;
  total += newReading;
  
  // Move to next index
  readIndex = (readIndex + 1) % WINDOW_SIZE;
  
  // Track number of valid readings (up to WINDOW_SIZE)
  if (validReadings < WINDOW_SIZE) {
    validReadings++;
  }
}

int detectAnomaly(long distance, float avgDistance) {
  // Not enough data yet - can't determine anomaly
  if (validReadings < 3) {
    return 0;
  }
  
  // Check if reading is outside valid sensor range
  if (distance < MIN_VALID_DISTANCE || distance > MAX_VALID_DISTANCE) {
    return 1;
  }
  
  // Check if deviation from rolling average exceeds threshold
  if (avgDistance > 0) {
    float deviation = abs(distance - avgDistance) / avgDistance;
    if (deviation > DEVIATION_THRESHOLD) {
      return 1;
    }
  }
  
  return 0;
}

void loop() {
  long distance = measureDistance();
  float avgDistance = getRollingAverage();
  
  // Detect anomaly BEFORE updating rolling average
  // (so current reading is compared against previous readings)
  int anomaly = detectAnomaly(distance, avgDistance);
  
  // Only update rolling average with valid, non-anomalous readings
  // This prevents anomalies from corrupting the baseline
  if (anomaly == 0 && distance >= MIN_VALID_DISTANCE && distance <= MAX_VALID_DISTANCE) {
    updateRollingAverage(distance);
  }
  
  // Output: distance,anomaly_flag
  Serial.print(distance);
  Serial.print(",");
  Serial.println(anomaly);

  delay(100);  // 10 Hz sampling rate
}
