# Voice Activator

### Privacy-first, low-latency keyword spotting on an ESP32

Voice Activator is an end-to-end embedded speech pipeline that detects a custom
wake word locally on an ESP32 and streams audio to a server only after the
keyword is detected. The project combines a small neural network, on-device
MFCC feature extraction, TensorFlow Lite Micro inference, and chunked
WebSocket audio transport.

The result is a practical edge-AI prototype: the device does not continuously
stream microphone audio, reducing unnecessary network traffic and keeping the
first-stage voice trigger close to the user.

> **Project status:** Working prototype. The training and export pipeline,
> quantized model artifact, ESP32 firmware, and WebSocket receiver are included.
> Hardware-level accuracy and end-to-end latency should be measured on the
> target setup before production use.

## Why this project is interesting

- **Edge inference:** keyword detection happens on the microcontroller rather
  than in the cloud.
- **Efficient model design:** a compact depthwise-separable CNN is trained for
  three classes: keyword, unknown speech, and silence.
- **Embedded deployment:** the model is converted to fully int8 TensorFlow Lite
  and exported as a C array for TensorFlow Lite Micro.
- **Streaming architecture:** audio is sent in small binary chunks immediately
  after detection instead of waiting for a complete recording.
- **Measurable engineering:** the firmware exposes tensor-arena usage and the
  receiver records first-chunk timing and saves received audio for inspection.

## System architecture

```text
INMP441 microphone
        |
        v
ESP32 I2S capture -> 1-second rolling buffer -> MFCC features
                                                   |
                                                   v
                                      int8 DS-CNN / TFLite Micro
                                                   |
                          keyword confidence >= threshold?
                              /                         \
                            no                           yes
                            |                             |
                    continue listening        WebSocket audio stream
                                                          |
                                                          v
                                               Python receiver / ASR
```

The firmware classifies a rolling 1-second window every 300 ms. When the
keyword class reaches the configured confidence threshold, it sends an
`utterance_start` marker followed by 16 kHz, mono, signed 16-bit PCM chunks.
After the configured streaming interval, it sends `utterance_end`.

## Technology stack

| Area | Technologies |
| --- | --- |
| Embedded hardware | ESP32, INMP441 I2S microphone |
| ML training | Python, TensorFlow/Keras, NumPy, SciPy, scikit-learn |
| Audio features | 16 kHz mono audio, 40 ms frames, 20 ms hop, 40 mel bins, 10 MFCC coefficients |
| Model | Compact DS-CNN with depthwise-separable convolution blocks |
| Deployment | Full-int8 TensorFlow Lite, TensorFlow Lite Micro |
| Transport | Wi-Fi and WebSockets |
| Receiver | Python `asyncio` WebSocket server |

## Repository layout

```text
.
├── train_kws_dscnn1.ipynb              # Train, evaluate, quantize, and export the model
├── record_samples.py                   # Record custom keyword WAV samples
├── convert_phone_recordings.py         # Convert phone recordings for dataset preparation
├── merge_recordings.py                 # Merge/preprocess recording sets
├── esp32_firmware/
│   └── kws_firmware/
│       ├── kws_firmware.ino            # ESP32 capture, inference, and streaming
│       ├── mfcc.cpp                    # On-device MFCC implementation
│       └── mfcc.h
├── stub_server/
│   └── server.py                       # WebSocket receiver and WAV exporter
├── 31-8-26_v4_esp/
│   ├── kws_model_int8.tflite           # Quantized model artifact
│   ├── kws_model_data.h                # Model converted to a C array
│   └── model_params.h                  # Normalization and quantization parameters
├── data/                               # Keyword recordings and prepared samples
└── LICENSE
```

## Quick start

### 1. Create a Python environment

Python 3.10 or 3.11 is recommended for the TensorFlow 2.15 training notebook.

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install numpy scipy sounddevice websockets
```

For model training, open [`train_kws_dscnn1.ipynb`](./train_kws_dscnn1.ipynb)
in Google Colab or a compatible Jupyter environment. The notebook installs
TensorFlow 2.15 and downloads Speech Commands background/negative samples.

### 2. Record a custom keyword

Record one-second, 16 kHz, mono WAV files with a laptop microphone:

```bash
python record_samples.py --label sonic --count 200 --outdir data/keyword
```

For a more robust model, vary speaker, distance, volume, pronunciation, and
background noise. The recorder can resume from a specific index:

```bash
python record_samples.py --label sonic --count 100 --start_index 200 --outdir data/keyword
```

### 3. Train and export the model

Run the notebook from top to bottom:

1. Load Speech Commands negatives and the custom keyword recordings.
2. Generate MFCC features and normalize them.
3. Train the regularized three-class DS-CNN.
4. Evaluate accuracy, keyword recall, false activation rate, and the confusion
   matrix.
5. Convert the model to full int8 TensorFlow Lite.
6. Verify quantized accuracy against the float model.
7. Export the `.tflite` model, C array, and firmware parameters.

The export step produces:

- `kws_model_int8.tflite` — portable quantized model
- `kws_model_data.h` — model bytes embedded as a C array
- `model_params.h` — MFCC normalization and int8 quantization constants
- `confusion_matrix.png` — evaluation artifact

The committed `model_params.h` documents the current model input shape:
`49 x 10` MFCC features. If you retrain the model, replace the generated
artifacts together so that the firmware and model remain compatible.

### 4. Run the receiver

Start the local WebSocket receiver before flashing the ESP32:

```bash
cd stub_server
python server.py
```

The server listens on `ws://0.0.0.0:8765`, logs connection and first-chunk
timing, and saves each received utterance under `stub_server/received_audio/`.
The saved WAV files make it easy to verify that the audio is correctly ordered,
aligned, and audible before integrating a real ASR backend.

### 5. Flash the ESP32

1. Copy the generated `kws_model_data.h` and `model_params.h` into
   `esp32_firmware/kws_firmware/`.
2. Install the Arduino libraries:
   - TensorFlowLite ESP32 (or another compatible TensorFlow Lite Micro port)
   - WebSockets by Markus Sattler
3. Configure `WIFI_SSID`, `WIFI_PASSWORD`, `WS_SERVER_HOST`, and
   `WS_SERVER_PORT` in `kws_firmware.ino`.
4. Wire the INMP441:

   | INMP441 | ESP32 |
   | --- | --- |
   | VDD | 3.3V |
   | GND | GND |
   | SD | GPIO32 |
   | SCK | GPIO26 |
   | WS | GPIO25 |
   | L/R | GND |

5. Select the ESP32 board in Arduino IDE, upload the sketch, and open the
   Serial Monitor at 115200 baud.

On boot, the firmware reports the TensorFlow Lite tensor-arena usage. Say the
keyword and confirm that the receiver logs an utterance and writes a WAV file.

## Configuration points

The main runtime settings are in
[`kws_firmware.ino`](./esp32_firmware/kws_firmware/kws_firmware.ino):

| Setting | Current value | Purpose |
| --- | ---: | --- |
| `SAMPLE_RATE` | 16000 Hz | Audio sampling rate |
| `AUDIO_BUFFER_MS` | 1000 ms | Rolling inference window |
| `CLASSIFY_EVERY_MS` | 300 ms | Classification interval |
| `STREAM_CHUNK_MS` | 40 ms | WebSocket PCM chunk duration |
| `DETECTION_THRESHOLD` | 0.85 | Keyword trigger threshold |
| `STREAM_DURATION_MS` | 3000 ms | Post-trigger streaming duration |
| Tensor arena | 60 KB reserved | TFLite Micro working memory |

These are starting points rather than universal guarantees. Threshold and
streaming duration should be tuned using recordings from the target environment.

## Model and embedded design

The notebook uses a compact DS-CNN with:

- an initial strided convolution,
- three depthwise + pointwise convolution blocks,
- batch normalization and ReLU activations,
- spatial dropout and L2 regularization,
- global average pooling,
- a three-class softmax output.

This architecture keeps computation and memory suitable for a microcontroller
while reducing overfitting on a small custom-keyword dataset. The deployment
path uses representative-data calibration and int8 input/output tensors, then
embeds the model directly into the firmware.

The device-side feature pipeline mirrors training: MFCC extraction, mean/std
normalization, and int8 quantization. Keeping those parameters in
`model_params.h` prevents the firmware from silently using stale calibration
values after retraining.

## Measuring quality and latency

The notebook reports:

- test accuracy,
- keyword recall / true-positive rate,
- false activation rate,
- confusion matrix,
- float-versus-quantized accuracy.

For an end-to-end measurement, compare the ESP32 detection timestamp with the
receiver's `utterance_start` and first-binary-chunk timestamps. Also listen to
the saved WAV files; a low numeric latency is not useful if samples are clipped,
misaligned, or corrupted.

## Current limitations and next steps

- The included Python receiver is a validation stub, not a production ASR
  service. It can be connected to a streaming Whisper, Vosk, or other ASR
  backend at the marked integration point.
- MFCC extraction is functional and intentionally straightforward. It should be
  compared against the notebook output on representative WAV files before
  relying on hardware accuracy numbers.
- The default detection threshold and stream duration need tuning with real
  users and background conditions.
- A production version should add secure transport/authentication, reconnect
  handling, watchdog/error telemetry, and a larger speaker-diverse dataset.
- Future optimization can replace the floating-point MFCC path with an
  ESP-DSP or other SIMD/fixed-point implementation if CPU or latency budgets
  require it.

## License

Released under the [MIT License](./LICENSE).
