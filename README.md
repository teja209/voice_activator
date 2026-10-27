# SIH26172 ZeroLatency — KWS Build Guide

Fastest path to a real, working model + real numbers. Follow in order.

## What you're getting

1. `record_samples.py` — records your custom keyword on your laptop mic
2. `train_kws_dscnn.ipynb` — Colab notebook: trains DS-CNN, quantizes to int8, exports
   `.tflite` + C header for ESP32, and prints/saves every metric you need for the PPT
3. `esp32_firmware/` — Arduino sketch: I2S mic capture, on-device MFCC, TFLite Micro
   inference, WiFi/WebSocket streaming on detection
4. `stub_server/server.py` — a receiving server so you can measure real latency and
   verify streamed audio is correct, without needing full ASR wired up yet

## Order of operations (maps to your 8-day plan, compressed)

### Step 1 — Record keyword samples (~30-45 min)
```bash
pip install sounddevice scipy numpy
python record_samples.py --label marvin --count 200 --outdir data
```
Pick a 2-3 syllable custom word, not a real English word already in Speech Commands
(avoids the unknown-class overlapping with your keyword). Say it naturally, vary
distance/tone/background per the docstring tips. Get a teammate or two to record some
too if time allows — meaningfully improves generalization.

Zip it when done:
```bash
zip -r keyword.zip data/marvin/
```

### Step 2 — Train (Colab, ~20-40 min hands-on + wait time)
Open `train_kws_dscnn.ipynb` in Colab (upload it, or File > Upload notebook).
Run cells top to bottom. Cell 3 will prompt you to upload `keyword.zip`.

**Change `KEYWORD_LABEL = 'marvin'`** in Cell 3 to your actual word.

Cell 10 gives you: test accuracy, keyword recall (TPR), false activation rate,
confusion matrix (saved as PNG — drop straight into your PPT).

Cell 11-13 quantize and export. You'll download 3 files at the end:
- `kws_model_data.h` — the model as a C array
- `model_params.h` — normalization + quantization constants the firmware needs
- `kws_model_int8.tflite` — the raw quantized model (keep for reference/re-export)
- `confusion_matrix.png` — for the PPT

### Step 3 — Flash to ESP32 (~1-2 hrs first time, mostly library setup)
1. Copy `kws_model_data.h` and `model_params.h` into `esp32_firmware/kws_firmware/`,
   replacing the placeholders.
2. In Arduino IDE, install libraries: **TensorFlowLite_ESP32** (or Chirale's TFLite
   Micro port — search both, use whichever installs cleanly), **WebSockets** (Markus
   Sattler), **arduinoFFT** (kosme).
3. Edit `kws_firmware.ino`: fill in `WIFI_SSID`, `WIFI_PASSWORD`, `WS_SERVER_HOST`
   (your laptop's LAN IP), pin numbers if your wiring differs from the comment block.
4. Wire INMP441: VDD→3.3V, GND→GND, SD→GPIO32, SCK→GPIO26, WS→GPIO25, L/R→GND.
5. Flash. Open Serial Monitor at 115200 baud.

**On boot it prints your real tensor arena (RAM) usage** — this is your Efficiency
metric for the PS. If it's over budget, first thing to try: reduce `kTensorArenaSize`
in the .ino down to just above what `arena_used_bytes()` reports, then reduce model
width (fewer filters in Cell 8) if still too big.

### Step 4 — Verify MFCC correctness (DO NOT SKIP — see mfcc.cpp header comment)
The single most likely failure mode: MFCC computed on-device doesn't match what the
model was trained on, so accuracy craters even though everything "runs." Before
trusting any accuracy number from hardware:
- Say your keyword near the mic, check Serial Monitor shows `class=0` (keyword)
  with reasonable confidence when you say it, and `class=1/2` otherwise.
- If it's flaky/wrong, the MFCC implementation in `mfcc.cpp` is the first suspect —
  it's a functional straightforward port, not verified bit-exact against the
  notebook's `tf.signal` calls. Budget time here on Day 5.

### Step 5 — Measure real latency (stub server)
```bash
cd stub_server
pip install websockets
python server.py
```
Say the keyword to the ESP32. The server logs timestamp of `utterance_start` receipt
and first audio chunk receipt — that delta plus your own keyword-end timestamp (logged
on the ESP32 side via Serial) gives you the PS's actual required metric: time from
keyword-end to cloud receiving audio. Saved WAV files let you confirm the streamed
audio is intact and correctly ordered.

### Step 6 — Swap in real ASR (optional, only if time remains)
The PS's graded scope ends at "cloud ASR receives the audio stream" — the stub server
already satisfies this. If you want a live demo with actual transcription, wire
`server.py`'s `# TODO: feed to ASR here` to whisper.cpp's server mode or a local Vosk
recognizer. Not required for Round 1 credibility, since it's explicitly out of graded
scope — prioritize Steps 1-5 numbers being solid first.

## Honest scope notes

- The MFCC implementation on-device is functional but not optimized or bit-verified
  against the training pipeline — flagged clearly in the code, budget real time to
  sanity-check it (Step 4).
- Detection threshold (`DETECTION_THRESHOLD = 0.85` in the .ino) is a starting guess.
  Tune it after you have real hardware behavior — lower if too many missed detections,
  raise if too many false triggers.
- CPU idle utilization isn't directly printed; if you need that number, add a rough
  duty-cycle measurement (time spent in `run_inference()` vs `CLASSIFY_EVERY_MS`
  interval) — inference time is already loggable via the commented-out line in the
  `.ino` loop.
- Everything here is a first working pass sized for your timeline, not a
  fully-optimized production system — matches what your PS explicitly wants for
  Round 1 (real numbers + credible "room for improvement" framing), not a polished
  final product.
