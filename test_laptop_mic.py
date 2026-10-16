import time
import wave
import sys
from pathlib import Path
import sounddevice as sd
import numpy as np

# Ensure workspace is in sys.path
workspace_dir = Path(__file__).resolve().parent
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

from backend.stt import STTWrapper

def record_and_test_laptop_mic():
    sample_rate = 16000
    duration_sec = 4
    output_path = "laptop_mic_test.wav"

    print("\n========================================================")
    print("LAPTOP MICROPHONE DIAGNOSTIC TEST")
    print("========================================================")
    print(f"Recording from laptop microphone for {duration_sec} seconds...")
    print(">>> SPEAK NOW into your laptop microphone! (e.g. 'What time is it right now?') <<<\n")

    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    print("\n>>> RECORDING STARTED! Speak clearly into your laptop mic now... <<<")
    audio_data = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print(">>> RECORDING COMPLETE! <<<\n")

    # Save to WAV file
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    # Inspect audio properties
    flat_data = audio_data.flatten()
    max_val = np.max(np.abs(flat_data))
    mean_val = np.mean(flat_data)
    file_size = Path(output_path).stat().st_size

    print(f"Recorded file: {output_path} ({file_size} bytes)")
    print(f"Laptop Mic Properties: DC offset = {mean_val:.1f}, Peak Amplitude = {max_val} / 32767")

    # Pass through STTWrapper
    stt = STTWrapper()
    print("\nTranscribing via STTWrapper...")
    transcript = stt.transcribe_file(output_path)

    print("\n========================================================")
    print("DIAGNOSTIC TEST RESULT")
    print("========================================================")
    print(f"Transcribed text from Laptop Mic: '{transcript}'")

    if transcript and len(transcript.strip()) > 1:
        from backend.main import ask_service, tts_wrapper
        from backend.config import settings
        print(f"\nQuerying Second Mind (model: {settings.OLLAMA_MODEL})...")
        ask_resp, audio_path = ask_service.process_ask_text(transcript, device_id="laptop_test")
        print(f"Intent:      {ask_resp.intent.value.upper()}")
        print(f"Confidence:  {ask_resp.confidence.value.upper()}")
        if ask_resp.formatted_time:
            print(f"Observed At: {ask_resp.formatted_time}")
        print(f"Answer:      '{ask_resp.answer_text}'")
        print("\n>>> Playing voice answer through LAPTOP SPEAKERS now... <<<")
        tts_wrapper.play_audio_file(audio_path)
        print(">>> Speaker playback completed! <<<")
    else:
        print("\n(No speech detected or mic audio was silent. Try speaking louder or closer to the microphone.)")
    print("========================================================\n")

if __name__ == "__main__":
    record_and_test_laptop_mic()
