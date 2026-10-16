#!/usr/bin/env python3
"""Interactive Laptop Voice Assistant for Second Mind.

Provides complete hands-free or push-to-talk voice interaction:
1. Captures audio directly from the laptop microphone.
2. Transcribes voice input via STTWrapper (faster-whisper / Google STT).
3. Dispatches query to Second Mind agents powered by local Ollama llama3.1:latest.
4. Plays synthesized voice answers out loud directly through laptop speakers.
"""

import argparse
import os
import sys
import time
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Ensure workspace root is in path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.config import settings
from backend.stt import STTWrapper
from backend.tts import TTSWrapper


def print_banner():
    banner = r"""
========================================================================
   ____  ____ ____ ___  _   _ ____  __  __ ___ _   _ ____  
  / ___|| ____/ ___/ _ \| \ | |  _ \|  \/  |_ _| \ | |  _ \ 
  \___ \|  _|| |  | | | |  \| | | | | |\/| || ||  \| | | | |
   ___) | |__| |__| |_| | |\  | |_| | |  | || || |\  | |_| |
  |____/|_____\____\___/|_| \_|____/|_|  |_|___|_| \_|____/ 
                                                            
       Local AI Ambient Assistant for Dementia Care
========================================================================
"""
    print(banner)


def check_audio_devices():
    """Verify laptop microphone and speaker accessibility."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        default_out = sd.default.device[1]
        in_name = devices[default_in]["name"] if default_in is not None and default_in < len(devices) else "Unknown"
        out_name = devices[default_out]["name"] if default_out is not None and default_out < len(devices) else "Unknown"
        return in_name, out_name
    except Exception as e:
        return f"Error: {e}", "System Default"


def record_laptop_mic(duration_sec: float = 5.0, sample_rate: int = 16000, output_file: str = "laptop_input.wav") -> Optional[str]:
    """Record audio from the laptop microphone and save to WAV."""
    import sounddevice as sd

    print(f"\n[MIC] Preparing to record from laptop microphone ({duration_sec}s)...")
    print("      >> Speak clearly now! (e.g. 'Where are my reading glasses?') <<")

    for i in range(3, 0, -1):
        print(f"      Recording starts in {i}...", end="\r", flush=True)
        time.sleep(1)
    print("\n   [● REC] RECORDING NOW... SPEAK!                   ")

    try:
        total_samples = int(duration_sec * sample_rate)
        audio_data = sd.rec(total_samples, samplerate=sample_rate, channels=1, dtype="int16")
        
        # Simple progress bar
        step = duration_sec / 10.0
        for _ in range(10):
            time.sleep(step)
            print("▓", end="", flush=True)
        sd.wait()
        print("\n   [✓] RECORDING COMPLETE!\n")

        # Check audio energy
        flat = audio_data.flatten()
        peak = int(np.max(np.abs(flat))) if len(flat) > 0 else 0
        mean = float(np.mean(flat)) if len(flat) > 0 else 0.0

        if peak < 150:
            print(f"   [!] WARNING: Very low audio signal detected (Peak: {peak}/32767).")
            print("       Make sure your laptop microphone is not muted or volume too low.")

        out_path = Path(output_file).resolve()
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        return str(out_path)
    except Exception as e:
        print(f"\n   [ERROR] Failed to record from microphone: {e}")
        return None


def ask_second_mind(transcript_or_file: str, is_file: bool = False, device_id: str = "laptop_user") -> Tuple[dict, str]:
    """Send voice or text query to Second Mind via HTTP or direct in-process service."""
    # 1. Try local HTTP server first if running
    import requests
    http_url = "http://localhost:8000/ask"
    
    try:
        if is_file:
            with open(transcript_or_file, "rb") as f:
                files = {"file": ("query.wav", f, "audio/wav")}
                resp = requests.post(
                    http_url,
                    files=files,
                    headers={"X-Device-ID": device_id},
                    params={"return_json": "true"},
                    timeout=25,
                )
        else:
            resp = requests.post(
                http_url,
                json={"transcript": transcript_or_file, "device_id": device_id},
                params={"return_json": "true"},
                timeout=25,
            )

        if resp.status_code == 200:
            data = resp.json()
            audio_path = resp.headers.get("X-Audio-Path") or ""
            return data, audio_path
    except Exception:
        pass  # Fallback to direct in-process execution

    # 2. In-process fallback: Always works without needing uvicorn running!
    from backend.main import ask_service, tts_wrapper, stt_wrapper

    if is_file:
        with open(transcript_or_file, "rb") as f:
            audio_bytes = f.read()
        ask_resp, audio_path = ask_service.process_ask_audio(audio_bytes, device_id=device_id)
    else:
        ask_resp, audio_path = ask_service.process_ask_text(transcript_or_file, device_id=device_id)

    data = ask_resp.model_dump(mode="json")
    return data, audio_path


def capture_and_register_object_opencv():
    """Capture reference photo from laptop camera and register into OpenCV Object Registry."""
    import cv2
    from backend.main import visual_service, tts_wrapper

    print("\n========================================================")
    print("  OPENCV OBJECT REGISTRATION (LAPTOP CAMERA)")
    print("========================================================")
    name = input("Enter object name (e.g. Reading Glasses, Keys, Pill Organizer): ").strip()
    if not name:
        print("[!] Object name cannot be empty.")
        return

    category = input("Enter category [personal/glasses/keys/medication/wallet] (default personal): ").strip() or "personal"
    description = input("Enter brief description (optional): ").strip()

    print("\nOpening laptop webcam...")
    print(">> Position the object clearly in front of your camera <<")
    input(">> Press ENTER when ready to snap photo... <<")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access laptop webcam.")
        return

    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[ERROR] Failed to capture frame from webcam.")
        return

    _, buf = cv2.imencode(".jpg", frame)
    print(f"\n[OPENCV] Processing and encoding '{name}' with MobileNetV3 + ORB features...")
    obj = visual_service.register_object(
        name=name,
        category=category,
        description=description,
        image_bytes=buf.tobytes(),
        filename=f"{name.lower().replace(' ', '_')}.jpg",
    )
    print(f"[✓] Successfully registered '{name}' (ID: {obj['id']}) in OpenCV Object Registry!")
    speak_msg = f"Successfully registered {name} in your OpenCV Object Registry."
    tts_wrapper.speak(speak_msg)


def scan_room_opencv():
    """Capture a live frame from laptop camera and search against OpenCV Object Registry."""
    import cv2
    from backend.main import visual_service, tts_wrapper

    print("\n========================================================")
    print("  OPENCV OBJECT DETECTION & MEMORY SCAN")
    print("========================================================")
    print("Scanning laptop camera for registered objects...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access laptop webcam.")
        return

    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[ERROR] Failed to capture frame from webcam.")
        return

    result = visual_service.search_frame(frame=frame, save_observation=True)
    print(f"Status:          {result.status.value.upper()}")
    print(f"Top Similarity:  {result.top_similarity:.3f}")

    if result.is_confident and result.best_match:
        bm = result.best_match
        print(f"Detected Object: {bm.object_name}")
        print(f"Location:        {bm.spatial_location}")
        print(f"Bounding Box:    {bm.bounding_box}")
        print(f"Local Matches:   {bm.local_matches}")
        print(f"Observed Time:   {result.formatted_time}")
        print(f"\n[✓] Saved to MemoryStore & Reality Checksum verified.")
        msg = f"I detected your {bm.object_name} on the {bm.spatial_location} {result.formatted_time}."
        print(f"[SPEAKER] Speaking: \"{msg}\"")
        tts_wrapper.speak(msg)
    else:
        msg = "I scanned the room, but did not detect any registered objects in this view."
        print(f"\n{msg}")
        tts_wrapper.speak(msg)


def list_registered_objects_opencv():
    """List all registered objects in the OpenCV Object Registry."""
    from backend.main import visual_service

    print("\n========================================================")
    print("  OPENCV OBJECT REGISTRY (Active Registered Objects)")
    print("========================================================")
    objs = visual_service.object_store.list_objects()
    if not objs:
        print("No objects currently registered.")
        return

    print(f"{'OBJECT NAME':<24} {'CATEGORY':<14} {'REFERENCES':<12} {'REGISTERED AT'}")
    print("-" * 65)
    for o in objs:
        print(f"{o['name']:<24} {o.get('category',''):<14} {o.get('reference_count',0):<12} {o.get('created_at','')[:16]}")
    print("-" * 65 + "\n")


def run_voice_loop():
    """Main interactive loop for laptop voice input and speaker output."""
    print_banner()

    in_dev, out_dev = check_audio_devices()
    print(f"  Laptop Mic:     {in_dev}")
    print(f"  Laptop Speaker: {out_dev}")
    print(f"  Target LLM:     {settings.OLLAMA_MODEL} (via Ollama)")
    print("------------------------------------------------------------------------")

    stt = STTWrapper()
    tts = TTSWrapper()

    while True:
        print("\nCHOOSE AN OPTION:")
        print("  [1] Push-to-Talk (Record from Laptop Mic & Speak Answer)")
        print("  [2] Quick 5-Second Voice Query")
        print("  [3] Type Question (Keyboard In -> Laptop Speaker Out)")
        print("  [4] Register Object via Laptop Camera (OpenCV Object Registry)")
        print("  [5] Scan Room with Laptop Camera (OpenCV Detection)")
        print("  [6] Launch Live OpenCV Camera Window & Tactical HUD")
        print("  [7] View All Registered Objects in OpenCV Registry")
        print("  [8] Test Laptop Speaker Sound")
        print("  [9] Exit")

        choice = input("\nEnter choice [1-9] (default 1): ").strip()
        if choice in ["9", "exit", "quit", "q"]:
            print("\nExiting Second Mind Laptop Voice Assistant. Take care!\n")
            break

        if choice == "4":
            capture_and_register_object_opencv()
            continue
        elif choice == "5":
            scan_room_opencv()
            continue
        elif choice == "6":
            from opencv_live_tracker import OpenCVLiveTrackerHUD
            print("\n[OpenCV] Launching Live Camera Window & Tactical HUD...")
            tracker = OpenCVLiveTrackerHUD()
            tracker.run()
            continue
        elif choice == "7":
            list_registered_objects_opencv()
            continue
        elif choice == "8":
            print("\n   [SPEAKER] Testing laptop speaker output...")
            tts.speak("Hello! This is Second Mind speaking through your laptop speakers. Everything is working perfectly.")
            print("   [✓] Test sound finished!\n")
            continue

        transcript = ""
        audio_file = None

        if choice in ["", "1", "2"]:
            dur = 5.0
            if choice == "1":
                custom = input("Recording duration in seconds [default 5]: ").strip()
                if custom.isdigit() and int(custom) > 0:
                    dur = float(custom)
            
            input("\n>>> Press ENTER when you are ready to speak... <<<")
            audio_file = record_laptop_mic(duration_sec=dur)
            if not audio_file:
                continue

            print("   [STT] Transcribing your voice via Whisper / SpeechRecognition...")
            transcript = stt.transcribe_file(audio_file)
            print(f"   [TRANSCRIPT] You said: \"{transcript}\"")

            if not transcript or len(transcript.strip()) < 2:
                print("   [!] Could not detect clear words. Please try speaking again closer to the mic.")
                continue

        elif choice == "3":
            transcript = input("\nEnter your question: ").strip()
            if not transcript:
                continue
        else:
            print("Invalid choice, please select 1 to 8.")
            continue

        # Process through Second Mind & LLaMA 3.1
        print(f"\n   [AI] Second Mind ({settings.OLLAMA_MODEL}) is thinking...")
        t0 = time.time()
        result, audio_path = ask_second_mind(transcript, is_file=False)
        elapsed = time.time() - t0

        intent = result.get("intent", "UNKNOWN")
        confidence = result.get("confidence", "UNKNOWN")
        reasoning = result.get("reasoning", "")
        answer = result.get("answer_text", "")
        formatted_time = result.get("formatted_time") or ""

        print("\n" + "=" * 60)
        print("  SECOND MIND RESPONSE")
        print("=" * 60)
        print(f"  Question:     {transcript}")
        print(f"  Intent:       {intent.upper()}")
        print(f"  Confidence:   {confidence.upper()}")
        if formatted_time:
            print(f"  Observed At:  {formatted_time}")
        print(f"  Reasoning:    {reasoning}")
        print(f"  Latency:      {elapsed:.2f}s")
        print("-" * 60)
        print(f"  Answer:       \"{answer}\"")
        print("=" * 60 + "\n")

        # Output voice through laptop speakers
        print("   [SPEAKER] Speaking answer through laptop speakers...")
        tts.speak(answer)
        print("   [✓] Voice playback complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Second Mind Laptop Voice Assistant")
    parser.add_argument("--test", action="store_true", help="Run speaker test and exit")
    parser.add_argument("--ask", type=str, default=None, help="Directly ask a question from command line")
    parser.add_argument("--opencv", "--hud", action="store_true", help="Launch live OpenCV camera window & tactical HUD")
    args = parser.parse_args()

    if args.test:
        tts = TTSWrapper()
        print("Testing laptop speakers...")
        tts.speak("Hello! Second Mind laptop audio output is ready.")
        sys.exit(0)

    if args.opencv:
        from opencv_live_tracker import OpenCVLiveTrackerHUD
        print("Launching Second Mind Live OpenCV Camera HUD...")
        tracker = OpenCVLiveTrackerHUD()
        tracker.run()
        sys.exit(0)

    if args.ask:
        stt = STTWrapper()
        tts = TTSWrapper()
        print(f"Asking: '{args.ask}'")
        res, _ = ask_second_mind(args.ask, is_file=False)
        print("Answer:", res.get("answer_text"))
        tts.speak(res.get("answer_text", ""))
        sys.exit(0)

    run_voice_loop()
