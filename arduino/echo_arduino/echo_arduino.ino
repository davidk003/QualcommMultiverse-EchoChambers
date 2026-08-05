#include <arduinoFFT.h>

  #define MIC_PIN A0
  #define SAMPLES 1024              // increased from 256 for finer frequency resolution
  #define SAMPLING_FREQUENCY 27270  // your MEASURED achievable rate

  ArduinoFFT<double> FFT = ArduinoFFT<double>();

  double vReal[SAMPLES];
  double vImag[SAMPLES];
  unsigned int samplingPeriodUs;

  void setup() {
    Serial.begin(115200);
    delay(2000); // give serial monitor time to connect
    samplingPeriodUs = round(1000000.0 / SAMPLING_FREQUENCY);

    Serial.print("Frequency resolution (Hz/bin): ");
    Serial.println((double)SAMPLING_FREQUENCY / SAMPLES);
  }

  void loop() {
    // ---- STEP 1: Capture samples, and measure ACTUAL achieved rate during this exact loop ----
    unsigned long captureStart = micros();
    unsigned long microseconds = captureStart;

    for (int i = 0; i < SAMPLES; i++) {
      vReal[i] = analogRead(MIC_PIN);
      vImag[i] = 0;
      while (micros() - microseconds < samplingPeriodUs) {
        // busy-wait to hold timing
      }
      microseconds += samplingPeriodUs;
    }

    unsigned long captureElapsedUs = micros() - captureStart;
    double actualSampleRate = SAMPLES / (captureElapsedUs / 1000000.0);

    Serial.print("Actual sample rate this loop (Hz): ");
    Serial.println(actualSampleRate);

    // ---- STEP 2: Print raw values so you can see if the mic is actually oscillating ----
    // Print min/max/average to check for DC bias / signal presence, then a short raw dump
    double minVal = 4095, maxVal = 0, sumVal = 0;
    for (int i = 0; i < SAMPLES; i++) {
      if (vReal[i] < minVal) minVal = vReal[i];
      if (vReal[i] > maxVal) maxVal = vReal[i];
      sumVal += vReal[i];
    }
    double avgVal = sumVal / SAMPLES;

    Serial.print("Raw signal - min: "); Serial.print(minVal);
    Serial.print(" max: "); Serial.print(maxVal);
    Serial.print(" avg: "); Serial.print(avgVal);
    Serial.print(" peak-to-peak: "); Serial.println(maxVal - minVal);

    // Dump first 100 raw samples so you can visually check for a repeating wave pattern
    Serial.println("First 100 raw samples:");
    for (int i = 0; i < 100 && i < SAMPLES; i++) {
      Serial.print(vReal[i]);
      Serial.print(",");
    }
    Serial.println();

    // ---- STEP 3: Remove DC offset before FFT (important if signal rides on a bias voltage) ----
    double vRealCopy[SAMPLES];
    for (int i = 0; i < SAMPLES; i++) {
      vRealCopy[i] = vReal[i] - avgVal;   // center around 0 for cleaner FFT
      vImag[i] = 0;
    }

    // ---- STEP 4: Run FFT on the DC-removed signal ----
    FFT.windowing(vRealCopy, SAMPLES, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
    FFT.compute(vRealCopy, vImag, SAMPLES, FFT_FORWARD);
    FFT.complexToMagnitude(vRealCopy, vImag, SAMPLES);

    double peakFrequencyHz = FFT.majorPeak(vRealCopy, SAMPLES, actualSampleRate); // use ACTUAL measured rate, not assumed

    Serial.print(">>> Dominant frequency (Hz): ");
    Serial.println(peakFrequencyHz);
    Serial.println("----------------------------------------");

    delay(1000);
  }




