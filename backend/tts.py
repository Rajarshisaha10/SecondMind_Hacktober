"""Text-to-Speech (TTS) wrapper module.

Provides speech synthesis interface for Track A using pyttsx3 offline TTS engine
with safe fallback to clean WAV audio file generation.
"""

import os
import struct
import subprocess
import sys
import uuid
import wave
from pathlib import Path
from typing import Optional
from backend.config import settings
from backend.logger import logger


class TTSWrapper:
    """Wrapper for offline speech synthesis."""

    def __init__(self, output_dir: Optional[Path] = None, speech_rate: int = 150) -> None:
        """Initialize TTSWrapper.

        Args:
            output_dir: Optional directory path to store generated audio files.
            speech_rate: Words per minute (default 150, calm and measured for dementia care).
        """
        self.output_dir = output_dir or settings.AUDIO_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.speech_rate = speech_rate

    def synthesize_to_file(
        self, text: str, output_path: Optional[str | Path] = None
    ) -> str:
        """Synthesize text answer into a WAV audio file.

        Args:
            text: Text string to synthesize.
            output_path: Optional output file path.

        Returns:
            Absolute string path to synthesized WAV audio file.
        """
        if output_path:
            file_path = Path(output_path)
        else:
            filename = f"answer_{uuid.uuid4().hex[:8]}.wav"
            file_path = self.output_dir / filename

        file_path.parent.mkdir(parents=True, exist_ok=True)

        success = self._synthesize_pyttsx3(text, file_path)
        if not success:
            logger.warning("TTSWrapper: Falling back to dummy WAV generator for '%s'", text)
            self._create_wav_file(file_path)

        logger.info(
            "TTSWrapper: Synthesized audio for '%s' -> '%s' (size=%d bytes)",
            text,
            file_path,
            file_path.stat().st_size if file_path.exists() else 0,
        )
        return str(file_path)

    def play_audio_file(self, file_path: str | Path, async_play: bool = False) -> bool:
        """Play WAV audio file directly through laptop speakers using native Windows sound."""
        path_str = str(file_path)
        try:
            import winsound
            flags = winsound.SND_FILENAME
            if async_play:
                flags |= winsound.SND_ASYNC
            winsound.PlaySound(path_str, flags)
            logger.info("TTSWrapper: Played audio '%s' through laptop speakers", path_str)
            return True
        except Exception as e:
            logger.warning("TTSWrapper: winsound playback failed (%s), attempting fallback...", e)
            try:
                import sounddevice as sd
                import scipy.io.wavfile as wavfile
                fs, data = wavfile.read(path_str)
                sd.play(data, fs)
                if not async_play:
                    sd.wait()
                return True
            except Exception as e2:
                logger.error("TTSWrapper: Speaker playback error: %s", e2)
                return False

    def speak(self, text: str, play_audio: bool = True) -> str:
        """Synthesize text and play spoken audio out loud through laptop speakers."""
        file_path = self.synthesize_to_file(text)
        if play_audio:
            self.play_audio_file(file_path)
        return file_path

    def _synthesize_pyttsx3(self, text: str, file_path: Path) -> bool:
        """Synthesize speech using pyttsx3 in an isolated process to avoid COM deadlocks on Windows."""
        clean_text = text.strip()
        if not clean_text:
            return False

        script = (
            "import sys, pyttsx3\n"
            "text = sys.argv[1]\n"
            "out = sys.argv[2]\n"
            "rate = int(sys.argv[3])\n"
            "engine = pyttsx3.init()\n"
            "engine.setProperty('rate', rate)\n"
            "engine.save_to_file(text, out)\n"
            "engine.runAndWait()\n"
        )
        try:
            res = subprocess.run(
                [sys.executable, "-c", script, text, str(file_path), str(self.speech_rate)],
                capture_output=True,
                timeout=8,
            )
            return res.returncode == 0 and file_path.exists() and file_path.stat().st_size > 100
        except Exception as e:
            logger.debug("TTSWrapper: pyttsx3 synthesis exception: %s", e)
            return False

    def _create_wav_file(self, file_path: Path) -> None:
        """Generate a valid 1-second 16kHz mono WAV file fallback.

        Args:
            file_path: Target output path.
        """
        sample_rate = 16000
        duration = 1.0
        num_samples = int(sample_rate * duration)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(file_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            data = struct.pack("<" + "h" * num_samples, *([0] * num_samples))
            wav_file.writeframes(data)
