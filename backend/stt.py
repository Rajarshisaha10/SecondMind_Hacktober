"""Speech-to-Text (STT) wrapper module for Second Mind.

Uses faster-whisper for local real-time accurate audio transcription.
"""

import tempfile
import wave
from pathlib import Path
from typing import Optional
from backend.config import settings
from backend.logger import logger


class STTWrapper:
    """Wrapper for real-time audio transcription using faster-whisper."""

    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "cpu",
    ) -> None:
        """Initialize STTWrapper.

        Args:
            model_size: Whisper model size ('small.en', 'base.en', 'base', 'small').
            device: Compute device ('cpu' or 'cuda').
        """
        self.model_size = model_size
        self.device = device
        self._model = None

    def _get_model(self):
        """Lazy load the faster-whisper model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel

                logger.info(
                    "STTWrapper: Loading faster-whisper model '%s' on device '%s'...",
                    self.model_size,
                    self.device,
                )
                self._model = WhisperModel(
                    self.model_size, device=self.device, compute_type="float32"
                )
                logger.info(
                    "STTWrapper: faster-whisper model '%s' loaded successfully",
                    self.model_size,
                )
            except Exception as e:
                logger.error("STTWrapper: Failed to load faster-whisper model: %s", e)
                raise e
        return self._model

    def _is_valid_transcript(self, text: str) -> bool:
        """Check if transcribed text is valid speech and not repetitive garbage."""
        if not text or len(text.strip()) < 2:
            return False

        clean = text.strip().replace(" ", "").replace("-", "").replace(".", "")
        # Detect single repeated character artifacts like "M-M-M-M-M" or "A-A-A-A"
        if len(clean) > 6 and len(set(clean.lower())) <= 3:
            logger.warning("STTWrapper: Rejected repetitive character artifact '%s'", text)
            return False

        if "m-m-m" in text.lower() or "come on, let's go" in text.lower():
            logger.warning("STTWrapper: Rejected repetitive phrase artifact '%s'", text)
            return False

        return True

    def _transcribe_google(self, file_path: str) -> str:
        """Fallback to free Google Speech Recognition API."""
        try:
            import speech_recognition as sr

            r = sr.Recognizer()
            with sr.AudioFile(file_path) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data)
                if text and len(text.strip()) > 0:
                    logger.info("STTWrapper: Google STT API fallback transcribed -> '%s'", text.strip())
                    return text.strip()
        except Exception as e:
            logger.debug("STTWrapper: Google STT fallback attempt note: %s", e)
        return ""

    def transcribe_file(self, file_path: str | Path) -> str:
        """Transcribe an audio WAV file into text string.

        Args:
            file_path: Path to WAV audio file.

        Returns:
            Transcribed text.
        """
        path_obj = Path(file_path)
        logger.info("STTWrapper: Transcribing audio file '%s'", path_obj)

        if not path_obj.exists():
            logger.error("STTWrapper: Audio file '%s' does not exist", path_obj)
            return ""

        file_size = path_obj.stat().st_size
        logger.info("STTWrapper: Audio file size: %d bytes", file_size)

        duration = 0.0
        sample_rate = 0
        channels = 0
        bit_depth = 0

        try:
            with wave.open(str(path_obj), "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / float(sample_rate) if sample_rate > 0 else 0.0
                bit_depth = sample_width * 8
            logger.info(
                "STTWrapper: Audio properties - duration: %.2fs, sample_rate: %dHz, channels: %d, bit_depth: %d-bit",
                duration,
                sample_rate,
                channels,
                bit_depth,
            )
        except Exception as e:
            logger.warning(
                "STTWrapper: Could not parse WAV header properties for '%s': %s",
                path_obj,
                e,
            )

        # 1. Attempt local faster-whisper transcription
        whisper_text = ""
        try:
            model = self._get_model()
            effective_path = str(path_obj)
            normalized_tmp_path = None

            # Soft normalization without aggressive scaling
            try:
                with wave.open(str(path_obj), "rb") as wf:
                    params = wf.getparams()
                    frames = wf.readframes(wf.getnframes())

                if len(frames) > 0 and params.sampwidth == 2:
                    import numpy as np

                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    audio_float = audio_data.astype(np.float32)

                    # Subtract DC offset
                    dc_offset = np.mean(audio_float)
                    audio_float = audio_float - dc_offset
                    max_val = np.max(np.abs(audio_float))

                    if 500 < max_val < 10000:
                        scale = 22000.0 / max_val
                        logger.info("STTWrapper: Boosting low-amplitude mic audio (%.2fx)", scale)
                        norm_audio = np.clip(audio_float * scale, -32768, 32767).astype(np.int16)
                        with tempfile.NamedTemporaryFile(suffix="_norm.wav", delete=False) as tmp_norm:
                            with wave.open(tmp_norm.name, "wb") as wf_out:
                                wf_out.setparams(params)
                                wf_out.writeframes(norm_audio.tobytes())
                            normalized_tmp_path = tmp_norm.name
                            effective_path = normalized_tmp_path
            except Exception as norm_err:
                logger.debug("STTWrapper: Audio preprocessing note: %s", norm_err)

            segments, info = model.transcribe(
                effective_path,
                language="en",
                beam_size=5,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            raw_text_parts = [seg.text for seg in segments]
            whisper_text = " ".join(raw_text_parts).strip()

            if normalized_tmp_path and Path(normalized_tmp_path).exists():
                try:
                    Path(normalized_tmp_path).unlink()
                except Exception:
                    pass

            if self._is_valid_transcript(whisper_text):
                logger.info("STTWrapper: Whisper transcript -> '%s'", whisper_text)
                return whisper_text
            else:
                logger.warning("STTWrapper: Whisper produced unparseable transcript '%s'", whisper_text)
        except Exception as e:
            logger.error("STTWrapper: Whisper transcription error: %s", e)

        # 2. Hybrid fallback: Google STT API
        logger.info("STTWrapper: Attempting Google STT API fallback...")
        google_text = self._transcribe_google(str(path_obj))
        if self._is_valid_transcript(google_text):
            return google_text

        return ""

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """Transcribe raw audio WAV bytes into text string.

        Args:
            audio_bytes: Incoming audio WAV bytes.

        Returns:
            Transcribed text.
        """
        if not audio_bytes:
            logger.warning("STTWrapper: Received empty audio_bytes")
            return ""

        logger.info("STTWrapper: Transcribing audio bytes (%d bytes)", len(audio_bytes))

        tmp_path = None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.transcribe_file(tmp_path)
        finally:
            if tmp_path and Path(tmp_path).exists():
                try:
                    Path(tmp_path).unlink()
                    logger.debug("STTWrapper: Deleted temporary WAV file '%s'", tmp_path)
                except Exception as e:
                    logger.warning(
                        "STTWrapper: Failed to delete temp file '%s': %s", tmp_path, e
                    )

