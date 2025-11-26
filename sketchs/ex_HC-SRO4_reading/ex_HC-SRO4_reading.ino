// HC-SR04 example (Arduino)
const int trigPin = 8;
const int echoPin = 9;

long duration;

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  Serial.begin(9600);
}

void loop() {
  // Clear the trigPin by setting it LOW for a brief moment
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // Trigger the sensor by setting the trigPin HIGH for 10 microseconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Measure the duration of the echo pulse using pulseIn()
  // pulseIn() returns the duration in microseconds
  duration = pulseIn(echoPin, HIGH);

  long distance = Distance(duration);

  // Print the distance to the Serial Monitor
  //Serial.print("Distance = ");
  Serial.println(distance);
  //Serial.println(" cm");

  // Add a small delay to avoid continuous readings
  delay(1000);
}

long Distance(long time)
{
  long DistanceCalc;

  DistanceCalc = ((time * 0.034) / 2);
  // DistanceCalc = time / 74 / 2; inches
  return DistanceCalc;
}

