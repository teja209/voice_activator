/*
 * SIH26172 - ZeroLatency
 * ESP32 Keyword Spotting -> WiFi stream to cloud ASR
 *
 * Pipeline:
 *   INMP441 (I2S) -> 1s rolling audio buffer -> MFCC -> TFLite Micro DS-CNN
 *   -> if keyword detected -> stream subsequent audio chunks via WebSocket
 *
 * BEFORE FLASHING:
 *   1. Replace kws_model_data.h and model_params.h with the ones from the
 *      training notebook (Cell 13 downloads).
 *   2. Fill in WIFI_SSID, WIFI_PASSWORD, WS_SERVER_HOST, WS_SERVER_PORT below.
 *   3. Wire INMP441: VDD->3.3V, GND->GND, SD->GPIO32, SCK->GPIO26, WS->GPIO25, L/R->GND
 *      (pins are configurable below if your wiring differs)
 *
 * Required Arduino libraries (install via Library Manager):
 *   - TensorFlowLite_ESP32  (or "Chirale_TensorFlowLite" — either TFLite Micro port works)
 *   - WebSockets by Markus Sattler (for streaming to server)
 *   - ArduinoFFT is NOT used; we hand-roll MFCC below to keep control over exact params
 *     matching the training notebook. If you'd rather offload MFCC to a library, ESP-DSP
 *     (Espressif's own, ESP-IDF only) is faster but requires ESP-IDF not Arduino.
 */

#include <WiFi.h>
#include <WebSocketsClient.h>
#include <math.h>

#include "kws_model_data.h"   // <-- from notebook Cell 13
#include "model_params.h"     // <-- from notebook Cell 13
#include "mfcc.h"             // on-device MFCC extraction, see mfcc.h/mfcc.cpp

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ---------------- CONFIG ----------------
#define WIFI_SSID       "YOUR_WIFI_SSID"
#define WIFI_PASSWORD   "YOUR_WIFI_PASSWORD"
#define WS_SERVER_HOST  "192.168.1.100"   // your stub/ASR server IP
#define WS_SERVER_PORT  8765

// I2S pins for INMP441 (adjust to your wiring)
#define I2S_WS_PIN   25
#define I2S_SCK_PIN  26
#define I2S_SD_PIN   32
#define I2S_PORT     I2S_NUM_0

#define SAMPLE_RATE       16000
#define AUDIO_BUFFER_MS   1000          // 1s sliding window for classification
#define AUDIO_BUFFER_LEN  (SAMPLE_RATE * AUDIO_BUFFER_MS / 1000)
#define CLASSIFY_EVERY_MS 300           // re-classify every 300ms (slide the window)
#define STREAM_CHUNK_MS   40            // chunk size when streaming post-detection

#define DETECTION_THRESHOLD 0.85f       // softmax confidence to trigger (tune after real testing)
#define CLASS_KEYWORD  0
#define CLASS_UNKNOWN  1
#define CLASS_SILENCE  2

// ---------------- Globals ----------------
int16_t audio_ring_buffer[AUDIO_BUFFER_LEN];
volatile int ring_write_pos = 0;
volatile bool ring_filled_once = false;

WebSocketsClient webSocket;
bool ws_connected = false;
bool streaming_audio = false;
unsigned long stream_start_time = 0;

// TFLite Micro globals
namespace {
  const tflite::Model* tf_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Tensor arena size: start conservative, tune down after checking actual usage
  // (arena_used_bytes() printed at startup tells you the real number for your PPT)
  constexpr int kTensorArenaSize = 60 * 1024;  // 60KB, adjust based on model
  uint8_t tensor_arena[kTensorArenaSize];
}

// MFCC output buffer, sized to match training: (INPUT_TIME_FRAMES x INPUT_MFCC_COEFFS)
float mfcc_output[INPUT_TIME_FRAMES][INPUT_MFCC_COEFFS];

// ---------------- I2S setup ----------------
#include <driver/i2s.h>

void setup_i2s() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,  // INMP441 outputs 32-bit frames (24-bit data)
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD_PIN
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
}

// Reads available I2S samples into the ring buffer. Call frequently from loop().
void read_i2s_into_ring_buffer() {
  int32_t raw_samples[256];
  size_t bytes_read = 0;
  i2s_read(I2S_PORT, raw_samples, sizeof(raw_samples), &bytes_read, 0);  // non-blocking (timeout=0)

  int n_samples = bytes_read / sizeof(int32_t);
  for (int i = 0; i < n_samples; i++) {
    // INMP441: 32-bit frame, top 24 bits are data -> shift down to get usable 16-bit-ish range
    int16_t sample = (int16_t)(raw_samples[i] >> 14);
    audio_ring_buffer[ring_write_pos] = sample;
    ring_write_pos = (ring_write_pos + 1) % AUDIO_BUFFER_LEN;
    if (ring_write_pos == 0) ring_filled_once = true;

    // If we're mid-stream (post-detection), forward raw samples to the WS chunk buffer too
    if (streaming_audio) {
      extern void stream_sample(int16_t);
      stream_sample(sample);
    }
  }
}

// Copies ring buffer into a linear array, oldest-to-newest
void get_linear_audio(int16_t* out, int len) {
  int start = ring_write_pos;  // oldest sample is at write_pos when buffer has wrapped
  for (int i = 0; i < len; i++) {
    out[i] = audio_ring_buffer[(start + i) % AUDIO_BUFFER_LEN];
  }
}

// ---------------- TFLite Micro setup ----------------
void setup_model() {
  tf_model = tflite::GetModel(g_kws_model);
  if (tf_model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema mismatch!");
    while (1) delay(1000);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      tf_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    while (1) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // *** THIS IS YOUR REAL RAM FOOTPRINT NUMBER FOR THE PPT ***
  size_t arena_used = interpreter->arena_used_bytes();
  Serial.printf("Tensor arena used: %u bytes (%.1f KB)\n", arena_used, arena_used / 1024.0);
}

// Run inference on the current 1s window, return (predicted_class, confidence)
void run_inference(int* pred_class, float* confidence) {
  int16_t linear_audio[AUDIO_BUFFER_LEN];
  get_linear_audio(linear_audio, AUDIO_BUFFER_LEN);

  // Compute MFCC (see mfcc.cpp) -- must match training notebook's params exactly
  compute_mfcc(linear_audio, AUDIO_BUFFER_LEN, mfcc_output);

  // Normalize with training mean/std, then quantize to int8
  int8_t* input_data = input->data.int8;
  int idx = 0;
  for (int t = 0; t < INPUT_TIME_FRAMES; t++) {
    for (int c = 0; c < INPUT_MFCC_COEFFS; c++) {
      float normalized = (mfcc_output[t][c] - MFCC_MEAN) / MFCC_STD;
      int32_t quantized = (int32_t)roundf(normalized / INPUT_SCALE) + INPUT_ZERO_POINT;
      if (quantized > 127) quantized = 127;
      if (quantized < -128) quantized = -128;
      input_data[idx++] = (int8_t)quantized;
    }
  }

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    *pred_class = -1;
    *confidence = 0;
    return;
  }

  // Dequantize output, softmax already applied in model so just compare
  int8_t* out_data = output->data.int8;
  float scores[3];
  float max_score = -1e9;
  int best = 0;
  for (int i = 0; i < 3; i++) {
    scores[i] = (out_data[i] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
    if (scores[i] > max_score) { max_score = scores[i]; best = i; }
  }
  *pred_class = best;
  *confidence = max_score;
}

// ---------------- WebSocket streaming ----------------
void webSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      ws_connected = true;
      Serial.println("WebSocket connected to ASR server");
      break;
    case WStype_DISCONNECTED:
      ws_connected = false;
      Serial.println("WebSocket disconnected");
      break;
    default:
      break;
  }
}

int16_t stream_chunk_buffer[SAMPLE_RATE * STREAM_CHUNK_MS / 1000];
int stream_chunk_pos = 0;

void stream_sample(int16_t sample) {
  stream_chunk_buffer[stream_chunk_pos++] = sample;
  if (stream_chunk_pos >= (SAMPLE_RATE * STREAM_CHUNK_MS / 1000)) {
    // Send this chunk immediately -- this is the "chunked not batch-then-send" requirement
    webSocket.sendBIN((uint8_t*)stream_chunk_buffer, stream_chunk_pos * sizeof(int16_t));
    stream_chunk_pos = 0;
  }
}

#define STREAM_DURATION_MS 3000  // how long to stream after detection (tune as needed)

void start_streaming() {
  streaming_audio = true;
  stream_start_time = millis();
  stream_chunk_pos = 0;
  Serial.println(">>> KEYWORD DETECTED -- streaming started <<<");
  // Send a small header/marker so the server knows a new utterance is starting
  webSocket.sendTXT("{\"event\":\"utterance_start\"}");
}

void check_stop_streaming() {
  if (streaming_audio && (millis() - stream_start_time > STREAM_DURATION_MS)) {
    streaming_audio = false;
    webSocket.sendTXT("{\"event\":\"utterance_end\"}");
    Serial.println("Streaming stopped.");
  }
}

// ---------------- Main ----------------
unsigned long last_classify_time = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("SIH26172 ZeroLatency - KWS boot");

  setup_i2s();
  setup_model();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected: " + WiFi.localIP().toString());

  webSocket.begin(WS_SERVER_HOST, WS_SERVER_PORT, "/");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(3000);

  Serial.println("Setup complete. Listening...");
}

void loop() {
  webSocket.loop();
  read_i2s_into_ring_buffer();
  check_stop_streaming();

  unsigned long now = millis();
  if (!streaming_audio && ring_filled_once && (now - last_classify_time >= CLASSIFY_EVERY_MS)) {
    last_classify_time = now;

    unsigned long t0 = micros();
    int pred_class;
    float confidence;
    run_inference(&pred_class, &confidence);
    unsigned long inference_us = micros() - t0;

    // Uncomment for latency debugging (this IS your "inference time" number, but
    // remember the PS's real latency metric is keyword-end -> server-receipt, measured
    // via timestamps on both ESP32 and server, not just this on-device inference time)
    // Serial.printf("Inference: %lu us, class=%d, conf=%.3f\n", inference_us, pred_class, confidence);

    if (pred_class == CLASS_KEYWORD && confidence >= DETECTION_THRESHOLD) {
      unsigned long detect_timestamp_ms = millis();  // log this against server-side receipt timestamp
      Serial.printf("Detected at t=%lu ms, confidence=%.3f\n", detect_timestamp_ms, confidence);
      start_streaming();
    }
  }
}
