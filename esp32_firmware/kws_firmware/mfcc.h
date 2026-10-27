#pragma once
#include <stdint.h>
#include "model_params.h"

// Computes MFCC features from a 1-second int16 audio buffer.
// MUST match the training notebook's Cell 5 params exactly:
//   frame_length=40ms, frame_step=20ms, fft_length=512, n_mel_bins=40, n_mfcc=10
//
// Output shape: [INPUT_TIME_FRAMES][INPUT_MFCC_COEFFS]
void compute_mfcc(const int16_t* audio, int audio_len, float output[][INPUT_MFCC_COEFFS]);
