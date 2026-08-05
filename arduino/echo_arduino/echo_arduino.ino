// Echo Chamber -- Arduino UNO Q mic bring-up / calibration sketch (v2)
//
// Goal: find out the REAL maximum sample rate this SparkFun SPH8878LR5H-1
// breakout + UNO Q ADC can achieve, and whether that's enough to see
// Echo Chamber's 16-24 kHz target band -- per the proposal's own Day-1
// "verify the mic against a calibrated tone sweep before any model work"
// step. This is a diagnostic/calibration tool, not the production capture
// path (that's deploy/arduino_uno_q/capture_agent.py, which streams raw
// audio to the host over WebSocket instead of running FFT on-device).
//
// What changed from v1, and why:
//
// 1. REMOVED the fixed-rate busy-wait pacing (the old code targeted a fixed
//    27270 Hz by busy-waiting between every analogRead()). The UNO Q's
//    Arduino core (ArduinoCore-zephyr) implements analogRead() as a
//    blocking, one-sample-at-a-time Zephyr adc_read() call -- per the
//    Arduino forum (https://forum.arduino.cc/t/uno-q-adc-sampling-rates-1mhz/1437572),
//    a free-running polling loop on this exact board has been reported
//    reaching roughly ~160 kS/s, well above the old 27270 Hz target. That
//    number was an artificial self-imposed limit, not a hardware ceiling --
//    this version removes the limit and MEASURES the real number instead
//    of assuming one.
//
//    NOTE: true hardware-timer-triggered + DMA continuous ADC capture (the
//    "proper" high-rate technique) is, as of this writing, a documented
//    unresolved gap in ArduinoCore-zephyr for this board -- see
//    https://forum.arduino.cc/t/uno-q-simultaneous-adc1-adc4-timer-triggered-dma-capture-at-up-to-2-5-ms-s/1448253
//    So this sketch intentionally stays on the simple polling API rather
//    than reaching for DMA/HAL code that isn't supported yet.
//
// 2. ADDED a band-limited peak/energy search restricted to 16-24 kHz.
//    FFT.majorPeak() searches the WHOLE spectrum for the single loudest
//    frequency -- in a real room that will almost always be low-frequency
//    noise (voice, footsteps, HVAC hum), which would silently bury any real
//    ultrasonic activity even if the mic is picking it up fine. This
//    version reports the overall peak (unchanged, still useful context)
//    AND a separate peak + energy figure computed only from bins that fall
//    in the actual target band.
//
// 3. ADDED an explicit Nyquist-coverage check each loop: since sample rate
//    is now measured, not assumed, this prints plainly whether the achieved
//    rate this loop was even fast enough to see up to 24 kHz at all.
//
// Reminder from the SparkFun hookup guide: this breakout's own onboard
// OPA344 amplifier has a frequency response of only ~7.2Hz-19.7kHz, even
// though the raw MEMS capsule is rated to 36kHz -- so even a perfect sample
// rate here cannot show you signal above ~19.7kHz on the AUD pin. Getting
// this sketch's sample rate right fixes the SOFTWARE half of the problem;
// it does not fix that hardware ceiling. See ARCHITECTURE.md.

#include <arduinoFFT.h>

#define MIC_PIN A0
#define SAMPLES 1024              // FFT window size

#define BAND_LO_HZ 16000.0        // Echo Chamber's target band -- see echo_chamber/__init__.py BAND_HZ
#define BAND_HI_HZ 24000.0

ArduinoFFT<double> FFT = ArduinoFFT<double>();

double vReal[SAMPLES];
double vImag[SAMPLES];

void setup() {
  Serial.begin(115200);
  delay(2000); // give serial monitor time to connect
  Serial.println("Echo Chamber mic calibration -- free-running sample rate (no artificial pacing)");
}

void loop() {
  // ---- STEP 1: Capture SAMPLES back-to-back as fast as analogRead() will go ----
  // No busy-wait, no target period -- this measures the board's real ceiling
  // instead of assuming one.
  unsigned long captureStart = micros();
  for (int i = 0; i < SAMPLES; i++) {
    vReal[i] = analogRead(MIC_PIN);
    vImag[i] = 0;
  }
  unsigned long captureElapsedUs = micros() - captureStart;
  double actualSampleRate = SAMPLES / (captureElapsedUs / 1000000.0);
  double nyquistHz = actualSampleRate / 2.0;

  Serial.println("----------------------------------------");
  Serial.print("Achieved sample rate this loop (Hz): ");
  Serial.println(actualSampleRate);
  Serial.print("Nyquist (max detectable freq, Hz): ");
  Serial.println(nyquistHz);
  if (nyquistHz < BAND_HI_HZ) {
    Serial.print("  [WARN] Nyquist is BELOW the ");
    Serial.print(BAND_HI_HZ);
    Serial.println("Hz target -- this loop cannot see the full 16-24kHz band.");
  } else {
    Serial.println("  [OK] Nyquist covers the full 16-24kHz target band.");
  }

  // ---- STEP 2: Raw signal sanity check (unchanged from v1) ----
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

  // ---- STEP 3: Remove DC offset before FFT ----
  double vRealCopy[SAMPLES];
  for (int i = 0; i < SAMPLES; i++) {
    vRealCopy[i] = vReal[i] - avgVal;
    vImag[i] = 0;
  }

  // ---- STEP 4: FFT ----
  FFT.windowing(vRealCopy, SAMPLES, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(vRealCopy, vImag, SAMPLES, FFT_FORWARD);
  FFT.complexToMagnitude(vRealCopy, vImag, SAMPLES);

  double overallPeakHz = FFT.majorPeak(vRealCopy, SAMPLES, actualSampleRate);
  Serial.print("Overall dominant frequency, whole spectrum (Hz): ");
  Serial.println(overallPeakHz);

  // ---- STEP 5: Band-limited peak + energy, restricted to 16-24kHz ----
  // Only the first SAMPLES/2 bins are meaningful for a real-input FFT.
  int nyquistBin = SAMPLES / 2;
  double hzPerBin = actualSampleRate / SAMPLES;

  int loBin = (int)(BAND_LO_HZ / hzPerBin);
  int hiBin = (int)(BAND_HI_HZ / hzPerBin);
  if (hiBin > nyquistBin) hiBin = nyquistBin;  // clip to what this loop's rate can actually see

  if (loBin >= hiBin) {
    Serial.println("Band-limited (16-24kHz) check: SKIPPED -- achieved sample rate this loop is too low to reach 16kHz at all.");
  } else {
    double bandEnergy = 0;
    double bandPeakMag = 0;
    int bandPeakBin = loBin;
    for (int i = loBin; i < hiBin; i++) {
      bandEnergy += vRealCopy[i];
      if (vRealCopy[i] > bandPeakMag) {
        bandPeakMag = vRealCopy[i];
        bandPeakBin = i;
      }
    }
    double bandPeakHz = bandPeakBin * hzPerBin;

    Serial.print("16-24kHz band energy (sum of magnitude): ");
    Serial.println(bandEnergy);
    Serial.print("16-24kHz band peak frequency (Hz): ");
    Serial.print(bandPeakHz);
    Serial.print("   magnitude: ");
    Serial.println(bandPeakMag);
  }

  delay(1000);
}
