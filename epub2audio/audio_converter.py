"""Text-to-speech conversion using Kokoro."""

import json
import os
from collections.abc import Generator
from dataclasses import asdict
from pathlib import Path
from typing import Union

import numpy as np
from kokoro import KModel, KPipeline  # Maybe I'll implement TextToSpeech, Voice
from loguru import logger
from soundfile import SoundFile

from .config import (
    KOKORO_PATHS,
    SAMPLE_RATE,
    SUPPORTED_AUDIO_FORMATS,
    ErrorCodes,
)
from .helpers import (
    CacheDirManager,
    ConversionError,
    StrPath,
    WordTiming,
)
from .voices import Voice


class AudioConverter:
    """Class for converting text to speech using Kokoro."""

    def __init__(
        self,
        epub_path: StrPath,
        voice: Union[str, Voice] = Voice.AF_HEART,
        speech_rate: float = 1.0,
        cache: bool = True,
        extension: str = ".flac",
    ):
        """Initialize the audio converter.

        Args:
            epub_path: Path to the EPUB file, used to generate a cache directory
            voice: Voice to use
            speech_rate: Speech rate multiplier
            cache: Whether to cache the generated audio
            extension: Extension to use for the output file

        Raises:
            ConversionError: If the voice is invalid or TTS initialization fails
        """
        try:
            self.voice = self._get_voice(voice)
            model = True
            if Path(KOKORO_PATHS["model_weight"]).exists():
                logger.debug(f"model weight found: {KOKORO_PATHS['model_weight']}")
                model = KModel(
                    config=KOKORO_PATHS["config"],
                    model=KOKORO_PATHS["model_weight"],
                    repo_id="hexgrad/Kokoro-82M",
                )
            self.tts = KPipeline(
                lang_code=self.voice.lang_code,
                repo_id="hexgrad/Kokoro-82M",
                model=model,
            )
            self.speech_rate = speech_rate
            self.cache_dir_manager = CacheDirManager(
                epub_path, extension=extension, voice=self.voice.name
            )
            self.cache = cache
            self.extension = extension
            self.format, self.subtype, _ = SUPPORTED_AUDIO_FORMATS[extension]
        except Exception as e:
            raise ConversionError(
                f"Failed to initialize TextToSpeech: {str(e)}", ErrorCodes.INVALID_VOICE
            ) from e

    def _get_voice(self, voice: Union[str, Voice]) -> Voice:
        """Get a voice by name.

        Args:
            voice: Voice to get

        Returns:
            Voice: The requested voice

        Raises:
            ConversionError: If the voice is invalid
        """
        if isinstance(voice, Voice):
            return voice
        try:
            return Voice.get_by_name(voice)
        except Exception as e:
            valid_voices = ", ".join([v.name for v in Voice.list_voices()])
            raise ConversionError(
                f"Invalid voice '{voice}'. Available voices: {valid_voices}",
                ErrorCodes.INVALID_VOICE,
            ) from e

    def _audio_data_generator(
        self, text: str
    ) -> Generator[KPipeline.Result, None, None]:
        """Generate audio data from text.

        Args:
            text: Text to convert

        Returns:
            Generator[KPipeline.Result, None, None]: Audio data generator
        """
        try:
            voice = self.voice.local_path if self.voice.local_path else self.voice.name
            yield from self.tts(
                text, voice=voice, speed=self.speech_rate, split_pattern=r"\n+"
            )
        except Exception as e:
            raise ConversionError(
                f"Failed to generate audio data: {str(e)}", ErrorCodes.UNKNOWN_ERROR
            ) from e

    def convert_text(self, text: str) -> SoundFile:
        """Convert text to speech.

        Args:
            text: Text to convert

        Returns:
            SoundFile: Converted audio
        """
        try:
            # Create a temporary file
            temp_file = self.cache_dir_manager.get_file(text)

            # If the file exists and caching is enabled, return the cached file
            if os.path.exists(temp_file) and self.cache:
                logger.trace(f"returning cached file: {temp_file}")
                return SoundFile(temp_file)

            if os.path.exists(f"{temp_file}.generating"):
                logger.trace(f"removing generating file: {temp_file}.generating")
                os.remove(f"{temp_file}.generating")

            audio_data = SoundFile(
                f"{temp_file}.generating",
                mode="w",
                samplerate=SAMPLE_RATE,
                channels=1,
                format=self.format,
                subtype=self.subtype,
            )
            int_size = np.int16
            max_int_size = np.iinfo(int_size).max
            # Generate speech
            for result in self._audio_data_generator(text):
                phonemes = result.phonemes
                audio = result.audio
                logger.trace(f"Phonemes: {phonemes}")
                if audio is None:
                    continue
                audio_bytes = (audio.numpy() * max_int_size).astype(int_size)
                audio_data.write(audio_bytes)

            # Close the audio data, and rename the file to the final file
            audio_data.close()
            os.rename(f"{temp_file}.generating", temp_file)
            return SoundFile(temp_file)

        except Exception as e:
            raise ConversionError(
                f"Failed to convert text to speech: {str(e)}", ErrorCodes.UNKNOWN_ERROR
            ) from e

    def convert_text_with_timing(
        self, text: str
    ) -> tuple[SoundFile, list[WordTiming]]:
        """Convert text to speech, also returning word-level timing.

        Timing comes from Kokoro's per-token alignment, which is already
        computed during inference but normally discarded. It is cached
        alongside the audio segment as a JSON sidecar (`<segment>.timing.json`)
        so cache hits still return timing without re-running TTS; a cache
        hit whose sidecar predates this feature returns an empty list.

        Args:
            text: Text to convert

        Returns:
            tuple[SoundFile, list[WordTiming]]: Converted audio and the
                word timings, relative to the start of this audio.
        """
        try:
            temp_file = self.cache_dir_manager.get_file(text)
            timing_file = f"{temp_file}.timing.json"

            if os.path.exists(temp_file) and self.cache:
                logger.trace(f"returning cached file: {temp_file}")
                words: list[WordTiming] = []
                if os.path.exists(timing_file):
                    with open(timing_file, encoding="utf-8") as f:
                        words = [WordTiming(**w) for w in json.load(f)]
                return SoundFile(temp_file), words

            if os.path.exists(f"{temp_file}.generating"):
                logger.trace(f"removing generating file: {temp_file}.generating")
                os.remove(f"{temp_file}.generating")

            audio_data = SoundFile(
                f"{temp_file}.generating",
                mode="w",
                samplerate=SAMPLE_RATE,
                channels=1,
                format=self.format,
                subtype=self.subtype,
            )
            int_size = np.int16
            max_int_size = np.iinfo(int_size).max
            words = []
            frames_written = 0
            for result in self._audio_data_generator(text):
                audio = result.audio
                if audio is None:
                    continue
                chunk_offset = frames_written / SAMPLE_RATE
                for token in result.tokens or []:
                    if token.start_ts is None or token.end_ts is None:
                        continue
                    word_text = token.text.strip()
                    if not word_text:
                        continue
                    words.append(
                        WordTiming(
                            text=word_text,
                            start=chunk_offset + token.start_ts,
                            end=chunk_offset + token.end_ts,
                            trailing_space=bool(token.whitespace),
                        )
                    )
                audio_bytes = (audio.numpy() * max_int_size).astype(int_size)
                audio_data.write(audio_bytes)
                frames_written += len(audio_bytes)

            audio_data.close()
            os.rename(f"{temp_file}.generating", temp_file)

            with open(timing_file, "w", encoding="utf-8") as f:
                json.dump([asdict(w) for w in words], f)

            return SoundFile(temp_file), words

        except Exception as e:
            raise ConversionError(
                f"Failed to convert text to speech: {str(e)}", ErrorCodes.UNKNOWN_ERROR
            ) from e
