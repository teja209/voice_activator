/*
 * On-device MFCC extraction.
 *
 * IMPORTANT HONESTY NOTE: This is a straightforward, functional MFCC implementation
 * using arduinoFFT. It is NOT the fastest possible version. If your idle-CPU or
 * latency numbers come in tight against the 10% CPU budget, the standard next
 * optimization (documented but not done here, out of scope for a 1-2 day build) is
 * to move to Espressif's ESP-DSP library (fixed-point, SIMD-friendly, ESP-IDF only)
 * or to CMSIS-NN equivalents. Get this working first, measure, THEN optimize if needed.
 *
 * Requires: arduinoFFT library (install via Library Manager: "arduinoFFT" by kosme)
 */

#include "mfcc.h"
#include <arduinoFFT.h>
#include <math.h>

#define SAMPLE_RATE 16000
#define FRAME_LENGTH_SAMPLES 640   // 40ms @ 16kHz
#define FRAME_STEP_SAMPLES   320   // 20ms @ 16kHz
#define FFT_LENGTH 512             // next pow2 >= 640... actually need care, see note below
#define N_MEL_BINS 40
#define N_MFCC INPUT_MFCC_COEFFS

// NOTE ON FFT_LENGTH: training notebook used fft_length=512 in tf.signal.stft, which
// TRUNCATES/pads the 640-sample frame to 512 for the FFT. arduinoFFT needs a power-of-2
// length. We replicate that by taking the first 512 samples of each 640-sample frame
// before windowing+FFT, matching TF's behavior (tf.signal.stft pads/truncates to fft_length
// when fft_length < frame_length... actually TF errors if fft_length < frame_length).
//
// SAFER FIX APPLIED HERE: use FFT_LENGTH=1024 (next pow2 >= 640) and zero-pad the 640
// frame to 1024 -- this matches librosa/most MFCC pipelines' standard behavior.
// >>> IF your notebook's actual STFT output bin count differs, TEST END-TO-END and
// >>> adjust N_MEL_BINS' spectrogram-bin mapping below to match. This is the #1 place
// >>> where "works in Colab, garbage on device" bugs come from. Budget time on Day 5/6
// >>> to sanity check: feed the SAME wav file through both the notebook's compute_mfcc
// >>> and this on-device version, print both outputs, confirm they're close.
#undef FFT_LENGTH
#define FFT_LENGTH 1024

static double vReal[FFT_LENGTH];
static double vImag[FFT_LENGTH];
static ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, FFT_LENGTH, SAMPLE_RATE);

static float mel_filterbank[N_MEL_BINS][FFT_LENGTH / 2 + 1];
static bool filterbank_initialized = false;

static float hz_to_mel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}
static float mel_to_hz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Build a standard triangular mel filterbank once, matching tf.signal.linear_to_mel_weight_matrix
// with lower_edge_hertz=20, upper_edge_hertz=4000 (must match training notebook Cell 5)
static void init_mel_filterbank() {
  float low_mel = hz_to_mel(20.0f);
  float high_mel = hz_to_mel(4000.0f);
  float mel_points[N_MEL_BINS + 2];
  for (int i = 0; i < N_MEL_BINS + 2; i++) {
    mel_points[i] = low_mel + (high_mel - low_mel) * i / (N_MEL_BINS + 1);
  }
  float hz_points[N_MEL_BINS + 2];
  for (int i = 0; i < N_MEL_BINS + 2; i++) hz_points[i] = mel_to_hz(mel_points[i]);

  int bin_points[N_MEL_BINS + 2];
  for (int i = 0; i < N_MEL_BINS + 2; i++) {
    bin_points[i] = (int)floorf((FFT_LENGTH + 1) * hz_points[i] / SAMPLE_RATE);
  }

  memset(mel_filterbank, 0, sizeof(mel_filterbank));
  for (int m = 1; m <= N_MEL_BINS; m++) {
    for (int k = bin_points[m - 1]; k < bin_points[m]; k++) {
      if (k >= 0 && k < FFT_LENGTH / 2 + 1 && bin_points[m] != bin_points[m-1])
        mel_filterbank[m - 1][k] = (float)(k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1]);
    }
    for (int k = bin_points[m]; k < bin_points[m + 1]; k++) {
      if (k >= 0 && k < FFT_LENGTH / 2 + 1 && bin_points[m+1] != bin_points[m])
        mel_filterbank[m - 1][k] = (float)(bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m]);
    }
  }
  filterbank_initialized = true;
}

// Simple DCT-II for MFCC from log-mel energies
static void dct(const float* mel_energies, float* mfcc_out, int n_mel, int n_mfcc) {
  for (int k = 0; k < n_mfcc; k++) {
    double sum = 0;
    for (int n = 0; n < n_mel; n++) {
      sum += mel_energies[n] * cos(M_PI * k * (2 * n + 1) / (2.0 * n_mel));
    }
    mfcc_out[k] = (float)sum;
  }
}

void compute_mfcc(const int16_t* audio, int audio_len, float output[][INPUT_MFCC_COEFFS]) {
  if (!filterbank_initialized) init_mel_filterbank();

  int n_frames = (audio_len - FRAME_LENGTH_SAMPLES) / FRAME_STEP_SAMPLES + 1;
  if (n_frames > INPUT_TIME_FRAMES) n_frames = INPUT_TIME_FRAMES;

  for (int f = 0; f < n_frames; f++) {
    int start = f * FRAME_STEP_SAMPLES;

    // Load frame, zero-pad to FFT_LENGTH, apply Hamming window
    for (int i = 0; i < FFT_LENGTH; i++) {
      if (i < FRAME_LENGTH_SAMPLES && (start + i) < audio_len) {
        float w = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (FRAME_LENGTH_SAMPLES - 1));
        vReal[i] = (audio[start + i] / 32768.0f) * w;
      } else {
        vReal[i] = 0.0;
      }
      vImag[i] = 0.0;
    }

    FFT.compute(FFTDirection::Forward);
    FFT.complexToMagnitude();

    // Apply mel filterbank
    float mel_energies[N_MEL_BINS];
    for (int m = 0; m < N_MEL_BINS; m++) {
      double energy = 0;
      for (int k = 0; k < FFT_LENGTH / 2 + 1; k++) {
        energy += vReal[k] * mel_filterbank[m][k];
      }
      mel_energies[m] = log(energy + 1e-6);
    }

    // DCT -> MFCC coefficients
    float mfcc_frame[N_MFCC];
    dct(mel_energies, mfcc_frame, N_MEL_BINS, N_MFCC);

    for (int c = 0; c < N_MFCC; c++) {
      output[f][c] = mfcc_frame[c];
    }
  }

  // Pad remaining frames with zeros if audio was shorter than expected
  for (int f = n_frames; f < INPUT_TIME_FRAMES; f++) {
    for (int c = 0; c < N_MFCC; c++) output[f][c] = 0.0f;
  }
}
