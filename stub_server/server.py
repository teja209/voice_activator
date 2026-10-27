"""
Stub WebSocket server for testing ESP32 -> cloud streaming and measuring latency.

Not a real ASR server yet -- just receives audio chunks, logs timing, and saves
the received audio so you can sanity-check it's actually your keyword's audio
(and not garbage/misaligned bytes, which is the #1 thing to verify before wiring
up real Whisper/Vosk).

Usage:
    pip install websockets
    python server.py

Then point WS_SERVER_HOST/PORT in the ESP32 firmware at this machine's IP:8765.

To swap in real ASR later: replace the `# TODO: feed to ASR here` section with a
call to whisper.cpp's server API or a local Vosk recognizer. Keep the chunk-by-chunk
receipt pattern -- that's what gives you low latency vs record-then-send.
"""
import asyncio
import json
import time
import wave
import os

import websockets

OUTPUT_DIR = "received_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 16000


async def handle_connection(websocket):
    print(f"[{time.strftime('%H:%M:%S')}] Client connected: {websocket.remote_address}")
    audio_chunks = []
    utterance_start_time = None
    first_chunk_time = None

    async for message in websocket:
        recv_time = time.time()

        if isinstance(message, str):
            data = json.loads(message)
            if data.get("event") == "utterance_start":
                utterance_start_time = recv_time
                first_chunk_time = None
                audio_chunks = []
                print(f"[{time.strftime('%H:%M:%S.%f')}] Utterance START received")
            elif data.get("event") == "utterance_end":
                print(f"[{time.strftime('%H:%M:%S.%f')}] Utterance END received. "
                      f"Total chunks: {len(audio_chunks)}, "
                      f"first-chunk latency from start marker: "
                      f"{(first_chunk_time - utterance_start_time)*1000:.1f}ms" if first_chunk_time else "")
                # Save the utterance to a WAV file for manual inspection
                if audio_chunks:
                    fname = os.path.join(OUTPUT_DIR, f"utterance_{int(time.time())}.wav")
                    with wave.open(fname, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # int16
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(b''.join(audio_chunks))
                    print(f"  Saved {fname} -- LISTEN TO THIS to verify audio quality/alignment")
        else:
            # Binary audio chunk
            if first_chunk_time is None:
                first_chunk_time = recv_time
                if utterance_start_time:
                    delta_ms = (first_chunk_time - utterance_start_time) * 1000
                    print(f"  >>> First audio chunk received {delta_ms:.1f}ms after start marker")
            audio_chunks.append(message)
            # TODO: feed to ASR here (chunk-by-chunk if using a streaming ASR like
            # whisper.cpp's streaming mode, or buffer + feed on utterance_end for
            # non-streaming Whisper)


async def main():
    print("Stub ASR server listening on ws://0.0.0.0:8765")
    print("Point your ESP32's WS_SERVER_HOST at this machine's LAN IP.")
    async with websockets.serve(handle_connection, "0.0.0.0", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
