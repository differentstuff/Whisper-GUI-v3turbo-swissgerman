# region Imports
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import numpy as np
from typing import Dict, Any
import os
import psutil
from scipy import signal
import src.print_color as pc
from src.logger_setup import model_logger as logger
from src.timestamp_handler import format_with_timestamps, format_without_timestamps

# endregion Imports


# region Config
# Audio Processing Configuration
# Adjusting these parameters can help optimize transcription quality and performance.
AUDIO_CONFIG = {
    # Target sample rate for audio processing
    # Whisper models expect 16000 Hz audio. Higher rates will be downsampled.
    "target_sample_rate": 16000,  # Hz
    # Audio normalization settings
    "audio": {
        "normalize": True,  # Whether to apply volume normalization
        "target_rms": -18,  # Target RMS level in dB (increased for better signal level)
        "target_peak": -2,  # Target peak level in dB (increased for better dynamics)
        "headroom": 0.1,  # Headroom to prevent clipping (0.1 = 10% below max)
        "min_gain": -24,  # Maximum reduction in dB (increased to match compression)
        "max_gain": 24,  # Maximum amplification in dB (balanced with reduction)
        "compression": {  # Dynamic range compression settings
            "enabled": False,  # Whether to apply compression
            "threshold": -24,  # Threshold in dB where compression begins
            # Lower threshold to catch more dynamic range
            "ratio": 2.5,  # Compression ratio (2.5:1)
            # Slightly stronger compression for better consistency
            "attack": 10,  # Attack time in milliseconds
            # Slower attack to preserve transients
            "release": 100,  # Release time in milliseconds
            # Longer release for smoother transitions
        },
    },
    # Chunk processing configuration for GPU and CPU modes
    "chunk": {
        "gpu": {
            "duration": 30,  # Length of each audio chunk in seconds
            # Standard chunk size for good context
            "overlap_ratio": 1
            / 5,  # How much chunks overlap (1/5 = 6 seconds for 30s chunks)
            # Balanced overlap for context preservation
            "step_ratio": 0.85,  # How far to advance between chunks (0.85 = 25.5s steps)
            # Conservative step size for stability
            "batch_size": 6,  # Number of chunks to process at once
            # Reduced for better memory efficiency with overlaps
        },
        "cpu": {
            "duration": 15,  # Shorter chunks for CPU to manage memory
            "overlap_ratio": 1 / 5,  # Same overlap ratio as GPU
            "step_ratio": 0.85,  # Same step ratio as GPU
            "batch_size": 1,  # Process one chunk at a time on CPU
        },
    },
    # Model generation parameters
    "model": {
        "max_length_multiplier": 1.75,  # Maximum output length relative to chunk duration
        # Optimized for 25% chunk overlap
        "min_gpu_memory": 4000,  # Minimum GPU memory required in MB
        # Below this, will fall back to CPU
        "generation": {  # Generation settings for transcription
            "do_sample": False,  # Disable sampling for more stable output
            "num_beams": 1,  # Use greedy search for faster, stable results
            "early_stopping": False,  # Disable early stopping with single beam
            "repetition_penalty": 1.2,  # Mild repetition penalty
            "no_repeat_ngram_size": 2,  # Conservative n-gram blocking
            "retry_on_repetition": True,  # Whether to retry chunks with repetition
            "words_per_second": 2,  # Approximate words per second for retry adjustment
        },
    },
    # Text processing settings
    "transcription": {
        "word_overlap": 12,  # Number of words to check when removing duplicates
        # Increased to better detect and remove duplicates
    },
}
# endregion Config


# region Initialization
pc.init()  # Initialize colorama
# endregion Initialization


# region Main
class WhisperModelHandler:
    def __init__(self, device=None, model_id=None):
        """Initialize the Whisper model handler"""

        logger.info(
            f"Initializing WhisperModelHandler with device={device}, model_id={model_id}"
        )
        self.model_id = model_id
        self.timestamps_enabled = False

        # Initialize CUDA if requested
        if device == "cuda" and torch.cuda.is_available():
            try:
                logger.info("Attempting CUDA initialization")
                torch.cuda.init()
                torch.cuda.empty_cache()
                gpu_info = {
                    "name": torch.cuda.get_device_name(0),
                    "total_memory": torch.cuda.get_device_properties(0).total_memory
                    / 1024**2,
                    "free_memory": (
                        torch.cuda.get_device_properties(0).total_memory
                        - torch.cuda.memory_allocated()
                    )
                    / 1024**2,
                }
                logger.info(
                    f"CUDA initialization successful. GPU: {gpu_info['name']}, Total VRAM: {gpu_info['total_memory']:.0f}MB, Free VRAM: {gpu_info['free_memory']:.0f}MB"
                )

                pc.print_info("GPU Configuration:")
                pc.print_info(f"• Device: {gpu_info['name']}")
                pc.print_info(f"• VRAM: {gpu_info['total_memory']:.0f}MB")
                pc.print_info(f"• Free VRAM: {gpu_info['free_memory']:.0f}MB")
                self.device = "cuda"
            except Exception as e:
                logger.error(f"CUDA initialization failed: {str(e)}", exc_info=True)
                pc.print_error(f"\nError initializing CUDA: {str(e)}")
                pc.print_warning("Falling back to CPU")
                self.device = "cpu"
        else:
            logger.info(
                "Using CPU mode" + (" (CUDA not available)" if device == "cuda" else "")
            )
            self.device = "cpu"
            pc.print_info("\nUsing CPU mode")

        # Initialize model settings
        self.processor = None
        self.model = None
        self.cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "model_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Cache directory set to: {self.cache_dir}")

        # Set processing parameters based on device configuration
        device_config = AUDIO_CONFIG["chunk"]["gpu" if self.device == "cuda" else "cpu"]
        self.chunk_length = device_config["duration"]
        self.batch_size = device_config["batch_size"]

        logger.info(
            f"Processing parameters set - Chunk Length: {self.chunk_length}s, Batch Size: {self.batch_size}"
        )
        pc.print_info("\nModel Settings:")
        pc.print_info(f"• Device: {self.device.upper()}")
        pc.print_info(f"• Cache Directory: {self.cache_dir}")
        pc.print_info(f"• Chunk Length: {self.chunk_length}s")
        pc.print_info(f"• Batch Size: {self.batch_size}")

    def load_model(self):
        """Load the model and processor"""
        if self.model is None:
            try:
                logger.info(f"Starting model load process for {self.model_id}")
                pc.print_header("Loading Model Components")

                # Step 1: Load processor
                logger.info("Starting processor download")
                pc.print_info("Step 1/4: Downloading processor")
                before_mem = (
                    torch.cuda.memory_allocated() / 1024**2
                    if self.device == "cuda"
                    else 0
                )

                self.processor = WhisperProcessor.from_pretrained(
                    self.model_id, cache_dir=self.cache_dir, local_files_only=False
                )

                after_mem = (
                    torch.cuda.memory_allocated() / 1024**2
                    if self.device == "cuda"
                    else 0
                )
                mem_change = after_mem - before_mem
                logger.info(
                    f"Processor downloaded successfully. Memory change: {mem_change:.1f}MB"
                )
                pc.print_success(f"Processor downloaded ({mem_change:.1f}MB)")

                # Step 2: Load model
                logger.info("Starting model download and initialization")
                pc.print_info("\nStep 2/4: Loading model (~3GB)")
                pc.print_info("This may take several minutes on first run")
                pc.print_info("Loading model: " + self.model_id)

                if self.device == "cuda":
                    free_memory = (
                        torch.cuda.get_device_properties(0).total_memory
                        - torch.cuda.memory_allocated()
                    ) / 1024**2
                    if free_memory < AUDIO_CONFIG["model"]["min_gpu_memory"]:
                        logger.warning(
                            f"Insufficient VRAM: {free_memory:.0f}MB available. Switching to CPU"
                        )
                        pc.print_warning(
                            f"Warning: Only {free_memory:.0f}MB VRAM available. Switching to CPU"
                        )
                        self.device = "cpu"

                before_mem = (
                    torch.cuda.memory_allocated() / 1024**2
                    if self.device == "cuda"
                    else 0
                )
                try:
                    # Clear GPU memory if needed
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        logger.info("Cleared CUDA cache before model load")

                    # Load model with optimized settings
                    logger.info(f"Loading model with device={self.device}")
                    self.model = WhisperForConditionalGeneration.from_pretrained(
                        self.model_id,
                        cache_dir=self.cache_dir,
                        local_files_only=False,  # Allow download if not present
                        torch_dtype=(
                            torch.float16 if self.device == "cuda" else torch.float32
                        ),  # Use float16 for GPU
                        device_map=self.device,  # Explicitly set device
                        low_cpu_mem_usage=True,  # More memory efficient
                        offload_folder=os.path.join(
                            self.cache_dir, "offload"
                        ),  # Enable model offloading
                    )

                    # Verify model loaded correctly
                    if self.model is None:
                        raise RuntimeError("Model failed to load")

                except Exception as e:
                    logger.error(
                        f"Error loading model on {self.device}: {str(e)}", exc_info=True
                    )
                    pc.print_error(f"\nError loading model on {self.device}: {str(e)}")

                    if self.device == "cuda":
                        logger.info("Attempting CPU fallback")
                        pc.print_info("Trying with CPU fallback")
                        self.device = "cpu"
                        torch.cuda.empty_cache()  # Clear GPU memory

                        self.model = WhisperForConditionalGeneration.from_pretrained(
                            self.model_id,
                            local_files_only=False,
                            torch_dtype=torch.float32,
                            cache_dir=self.cache_dir,
                            device_map="cpu",
                            low_cpu_mem_usage=True,
                            offload_folder=os.path.join(self.cache_dir, "offload"),
                        )
                    else:
                        raise  # Re-raise the error if already on CPU

                after_mem = (
                    torch.cuda.memory_allocated() / 1024**2
                    if self.device == "cuda"
                    else 0
                )
                mem_change = after_mem - before_mem
                logger.info(
                    f"Model loaded successfully. Memory change: {mem_change:.1f}MB"
                )
                pc.print_success(
                    f"Model loaded successfully (Memory change: {mem_change:.1f}MB)"
                )

                # Step 3: Verify model location
                logger.info("Verifying model device placement")
                pc.print_info("\nStep 3/4: Verifying model location")
                actual_device = next(self.model.parameters()).device
                # Check if model is on any CUDA device when CUDA is requested
                logger.info(f"Model is on device: {actual_device}")

                is_on_correct_device = (
                    self.device == "cuda" and actual_device.type == "cuda"
                ) or (self.device == "cpu" and actual_device.type == "cpu")

                if not is_on_correct_device:
                    logger.warning(
                        f"Model is on {actual_device}, expected {self.device}"
                    )
                    pc.print_warning(
                        f"Warning: Model is on {actual_device}, expected {self.device}"
                    )
                    if self.device == "cuda":
                        logger.info("Forcing model to GPU")
                        pc.print_info("Forcing model to GPU")
                        self.model.cuda()
                else:
                    logger.info(f"Model correctly loaded on {actual_device}")
                    pc.print_success(f"Model correctly loaded on {actual_device}")

                # Step 4: Verify model state
                logger.info("Verifying model state")
                pc.print_info("\nStep 4/4: Verifying model state")
                if self.model is None or self.processor is None:
                    logger.error("Model or processor not loaded properly")
                    raise RuntimeError("Model or processor not loaded properly")

                logger.info(f"Model initialization complete on {self.device}")
                pc.print_success(f"Model loaded successfully on {self.device}")

                if self.device == "cuda":
                    vram_usage = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"Final VRAM usage: {vram_usage:.1f}MB")
                    pc.print_info(f"VRAM usage: {vram_usage:.1f}MB")
                else:
                    ram_usage = psutil.Process().memory_info().rss / 1024**2
                    logger.info(f"Final RAM usage: {ram_usage:.1f}MB")
                    pc.print_info(f"RAM usage: {ram_usage:.1f}MB")

            except Exception as e:
                logger.error(f"Error loading model: {str(e)}", exc_info=True)
                pc.print_error(f"\nError loading model: {str(e)}")
                pc.print_warning("Please check your internet connection and try again")
                raise

    def transcribe(
        self,
        audio_path: str,
        language: str = "Auto",
        verbose: bool = True,
        timestamps_enabled: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using the Whisper model

        Args:
            audio_path: Path to audio file
            language: Language code (ignored as this is Swiss German specialized)
            verbose: Whether to show progress bar
            timestamps_enabled: Whether to include timestamps in output
            **kwargs: Additional arguments (for compatibility)
        """

        logger.info(f"Starting transcription of: {audio_path}")
        logger.info(
            f"Parameters: language={language}, verbose={verbose}, kwargs={kwargs}"
        )

        # Store timestamp setting for this transcription
        self.timestamps_enabled = timestamps_enabled
        logger.info(f"Timestamps {'enabled' if timestamps_enabled else 'disabled'}")

        from pydub import AudioSegment

        # Ensure model is loaded
        if self.model is None:
            logger.info("Model not loaded. Loading now")
            pc.print_info("\nModel not loaded. Loading now")
            self.load_model()
            if self.model is None:
                logger.error("Failed to load model")
                raise RuntimeError("Failed to load model")

            if self.device == "cuda":
                vram_usage = torch.cuda.memory_allocated() / 1024**2
                logger.info(
                    f"Model loaded successfully on GPU. VRAM usage: {vram_usage:.1f}MB"
                )
                pc.print_success(
                    f"Model loaded successfully on GPU. VRAM usage: {vram_usage:.1f}MB"
                )
            else:
                ram_usage = psutil.Process().memory_info().rss / 1024**2
                logger.info(
                    f"Model loaded successfully on CPU. RAM usage: {ram_usage:.1f}MB"
                )
                pc.print_success(
                    f"Model loaded successfully on CPU. RAM usage: {ram_usage:.1f}MB"
                )

        pc.print_header(f"Starting transcription of: {os.path.basename(audio_path)}")

        # Get file size and log it
        file_size = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"File size: {file_size:.2f} MB")
        pc.print_info(f"File size: {file_size:.2f} MB")
        pc.print_info(f"Timestamps {'enabled' if timestamps_enabled else 'disabled'}")

        try:
            # Load and normalize audio file
            logger.info("Loading audio file")
            audio = AudioSegment.from_file(audio_path)
            sr = audio.frame_rate
            logger.info(
                f"Audio loaded: {len(audio)/1000:.1f}s, {sr}Hz, {audio.channels} channels"
            )

            # Convert to mono if needed
            if audio.channels > 1:
                logger.info("Converting audio to mono")
                audio = audio.set_channels(1)

            # Apply compression if enabled
            if AUDIO_CONFIG["audio"]["compression"]["enabled"]:
                logger.info("Applying audio compression")
                audio = self._apply_compression(audio)

            # Normalize volume if enabled
            if AUDIO_CONFIG["audio"]["normalize"]:
                logger.info("Normalizing audio volume")
                audio = self._normalize_volume(audio)

            # Convert audio to numpy array and handle sample rate
            logger.info("Converting audio to numpy array")
            speech, sr = self._prepare_audio_samples(audio)

            duration = len(speech) / sr
            logger.info(f"Final audio duration: {duration:.1f} seconds")
            pc.print_info(f"Audio duration: {duration:.1f} seconds")

            # Create chunks for processing
            logger.info("Creating audio chunks")
            chunks, chunk_times, chunk_duration = self._create_chunks(
                speech, sr, duration
            )

            # Process chunks and get transcriptions
            logger.info("Starting chunk processing")
            pc.print_header(f"Starting processing chunks")
            all_transcriptions = self._process_chunks(
                chunks, chunk_times, chunk_duration, sr, language
            )

            # Combine transcriptions and create result
            logger.info("Combining transcriptions")
            combined_transcriptions, transcription = self._combine_transcriptions(
                all_transcriptions
            )

            if not combined_transcriptions:
                logger.warning("No transcriptions generated")
                return {"text": "", "segments": [], "language": language}

            logger.info("Transcription complete")
            logger.info(f"Final transcription length: {len(transcription)} characters")
            pc.print_success("Transcription complete")
            pc.print_info(
                f"Total transcription length: {len(transcription)} characters"
            )
            pc.print_info(
                f"Processed {len(chunks)} chunks, covering {duration:.2f} seconds of audio"
            )

            # Format output
            result = {
                "text": transcription,
                "segments": [
                    {"id": i, "start": start, "end": end, "text": text}
                    for i, (start, end, text) in enumerate(combined_transcriptions)
                ],
                "language": language,
            }

            # Format the output based on timestamp setting
            result = (
                format_with_timestamps(result)
                if self.timestamps_enabled
                else format_without_timestamps(result)
            )

            logger.info("Transcription result formatted successfully")
            return result

        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}", exc_info=True)
            raise

    def _process_chunks(
        self,
        chunks: list,
        chunk_times: list,
        chunk_duration: float,
        sr: int,
        language: str,
    ) -> list:
        """Process audio chunks and generate transcriptions

        Args:
            chunks: List of audio chunks
            chunk_times: List of (start_time, end_time) tuples
            chunk_duration: Duration of each chunk in seconds
            sr: Sample rate
            language: Language code for transcription

        Returns:
            list: List of (start_time, end_time, text) tuples
        """
        logger.info(
            f"Starting chunk processing: {len(chunks)} chunks, {chunk_duration}s each, {sr}Hz sample rate"
        )
        all_transcriptions = []

        for i, (chunk, (chunk_start_time, chunk_end_time)) in enumerate(
            zip(chunks, chunk_times), 1
        ):
            try:
                logger.info(
                    f"Processing chunk {i}/{len(chunks)} ({chunk_start_time:.2f}s - {chunk_end_time:.2f}s)"
                )
                pc.print_info(f"Processing chunk {i}/{len(chunks)}")
                pc.print_info(
                    f"Chunk time range: {chunk_start_time:.2f}s - {chunk_end_time:.2f}s"
                )

                # Process audio chunk with attention mask
                logger.debug("Processing audio chunk with processor")
                processed = self.processor(
                    chunk,
                    sampling_rate=sr,
                    return_tensors="pt",
                    return_attention_mask=True,  # Suppress warning for language detection
                )

                # Move to device and set correct data type
                logger.debug(f"Moving tensors to {self.device}")
                input_features = processed.input_features.to(self.device)
                attention_mask = (
                    processed.attention_mask.to(self.device)
                    if hasattr(processed, "attention_mask")
                    else None
                )

                if self.device == "cuda":
                    logger.debug("Converting tensors to float16 for GPU")
                    input_features = input_features.to(torch.float16)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(torch.float16)
                else:
                    logger.debug("Converting tensors to float32 for CPU")
                    input_features = input_features.to(torch.float32)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(torch.float32)

                # Clear memory before generation
                if self.device == "cuda":
                    logger.debug("Clearing CUDA cache before generation")
                    torch.cuda.empty_cache()

                # Generate transcription
                logger.info(f"Generating transcription for chunk {i}")
                pc.print_info(f"Generating transcription for chunk {i}")
                gen_config = AUDIO_CONFIG["model"]["generation"]

                logger.debug(f"Generation config: {gen_config}")
                predicted_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    do_sample=gen_config["do_sample"],
                    repetition_penalty=gen_config["repetition_penalty"],
                    no_repeat_ngram_size=gen_config["no_repeat_ngram_size"],
                    num_beams=gen_config["num_beams"],
                    early_stopping=gen_config["early_stopping"],
                    max_length=int(
                        chunk_duration * AUDIO_CONFIG["model"]["max_length_multiplier"]
                    ),
                    language=None if language == "Auto" else language.lower(),
                    return_timestamps=self.timestamps_enabled,  # Enable timestamp output
                )

                # Clear memory after generation
                logger.debug("Cleaning up memory after generation")
                del input_features
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                # Decode transcription
                logger.debug("Decoding transcription")
                chunk_text = self.processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0].strip()

                if chunk_text:
                    # Check for repetitions
                    if AUDIO_CONFIG["model"]["generation"][
                        "retry_on_repetition"
                    ] and self._detect_repetition(chunk_text):
                        logger.info(
                            "Repetition detected, retrying with adjusted boundaries"
                        )
                        pc.print_info(
                            "Detected repetition in chunk, retrying with adjusted boundaries"
                        )

                        # Adjust chunk boundaries
                        samples_per_second = sr
                        adjusted_chunk = (
                            chunk[samples_per_second:-samples_per_second]
                            if len(chunk) > 2 * samples_per_second
                            else chunk
                        )
                        logger.debug(
                            f"Adjusted chunk length: {len(adjusted_chunk)} samples"
                        )

                        # Reprocess with adjusted chunk
                        logger.debug("Reprocessing adjusted chunk")
                        processed = self.processor(
                            adjusted_chunk,
                            sampling_rate=sr,
                            return_tensors="pt",
                            return_attention_mask=True,
                        )

                        # Move to device and process
                        input_features = processed.input_features.to(self.device)
                        attention_mask = (
                            processed.attention_mask.to(self.device)
                            if hasattr(processed, "attention_mask")
                            else None
                        )

                        if self.device == "cuda":
                            input_features = input_features.to(torch.float16)
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(torch.float16)
                        else:
                            input_features = input_features.to(torch.float32)
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(torch.float32)

                        # Generate new transcription
                        logger.debug("Generating transcription for adjusted chunk")
                        predicted_ids = self.model.generate(
                            input_features,
                            attention_mask=attention_mask,
                            do_sample=gen_config["do_sample"],
                            repetition_penalty=gen_config["repetition_penalty"],
                            no_repeat_ngram_size=gen_config["no_repeat_ngram_size"],
                            num_beams=gen_config["num_beams"],
                            early_stopping=gen_config["early_stopping"],
                            max_length=int(
                                chunk_duration
                                * AUDIO_CONFIG["model"]["max_length_multiplier"]
                            ),
                            language=None if language == "Auto" else language.lower(),
                        )

                        # Update chunk text
                        chunk_text = self.processor.batch_decode(
                            predicted_ids, skip_special_tokens=True
                        )[0].strip()

                        logger.info("Retry complete")
                        pc.print_success("Retry complete")

                    all_transcriptions.append(
                        (chunk_start_time, chunk_end_time, chunk_text)
                    )
                    logger.info(f"Chunk {i} complete: {len(chunk_text)} chars")
                    logger.debug(f"Chunk {i} text: {chunk_text}")
                    pc.print_success(f"Chunk {i} complete ({len(chunk_text)} chars)")
                    pc.print_info(f"Text: {chunk_text}")
                else:
                    logger.warning(f"Chunk {i} produced no text")
                    pc.print_warning(f"Warning: Chunk {i} produced no text")

            except torch.cuda.OutOfMemoryError as e:
                error_msg = f"GPU out of memory on chunk {i}"
                logger.error(error_msg, exc_info=True)
                pc.print_error(f"\nError: {error_msg}. Try using CPU mode")
                raise
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}", exc_info=True)
                pc.print_warning(f"\nError processing chunk {i}: {str(e)}")
                raise

        logger.info(
            f"Chunk processing complete: {len(all_transcriptions)} transcriptions generated"
        )
        return all_transcriptions

    def _combine_transcriptions(self, transcriptions: list) -> tuple:
        """Combine chunk transcriptions and handle overlaps

        Args:
            transcriptions: List of (start_time, end_time, text) tuples

        Returns:
            tuple: (list of combined segments, full transcription text)
        """
        logger.info(
            f"Starting transcription combination: {len(transcriptions)} segments"
        )
        pc.print_info("\nCombining transcriptions")
        combined = []

        for i, item in enumerate(transcriptions):
            if item is None:
                logger.warning(f"Chunk {i+1} produced no transcription")
                pc.print_warning(f"Warning: Chunk {i+1} produced no transcription")
                continue

            start, end, text = item
            logger.debug(f"Processing segment {i+1}: {start:.2f}s - {end:.2f}s")

            if i == 0 or not combined:
                logger.debug(f"Adding first segment: {text[:50]}...")
                combined.append((start, end, text))
            else:
                prev_start, prev_end, prev_text = combined[-1]
                if start < prev_end:
                    logger.info(f"Overlap detected between segments {i} and {i+1}")
                    if not self.timestamps_enabled:
                        # Only merge if timestamps are disabled
                        merged_text = self._merge_transcriptions(prev_text, text)
                        logger.debug(f"Merged text: {merged_text[:50]}...")
                        combined[-1] = (prev_start, end, merged_text)
                        logger.debug("Merged overlapping segments")
                    else:
                        # When timestamps are enabled, use midpoint of overlap
                        overlap_mid = (prev_end + start) / 2
                        combined[-1] = (prev_start, overlap_mid, prev_text)
                        combined.append((overlap_mid, end, text))
                        logger.debug(f"Split at overlap midpoint: {overlap_mid:.2f}s")
                else:
                    logger.debug(f"Adding non-overlapping segment: {text[:50]}...")
                    combined.append((start, end, text))

        if not combined:
            logger.error("No valid transcriptions were produced")
            pc.print_error("Error: No valid transcriptions were produced")
            return [], ""

        # Create full transcription
        transcription = " ".join([text for _, _, text in combined])
        logger.info(
            f"Transcription combination complete: {len(combined)} segments, {len(transcription)} chars"
        )
        return combined, transcription

    def _create_chunks(self, speech: np.ndarray, sr: int, duration: float) -> tuple:
        """Create overlapping chunks for audio processing

        Args:
            speech: Audio samples as numpy array
            sr: Sample rate
            duration: Total audio duration in seconds

        Returns:
            tuple: (list of chunks, list of chunk times, chunk duration)
        """
        logger.info(f"Creating chunks for {duration:.1f}s audio at {sr}Hz")

        # Get chunk configuration
        device_config = AUDIO_CONFIG["chunk"]["gpu" if self.device == "cuda" else "cpu"]
        chunk_duration = device_config["duration"]
        samples_per_chunk = int(chunk_duration * sr)

        logger.debug(
            f"Chunk configuration: duration={chunk_duration}s, samples_per_chunk={samples_per_chunk}"
        )

        # Calculate overlap
        overlap_duration = chunk_duration * device_config["overlap_ratio"]
        overlap_samples = int(overlap_duration * sr)
        logger.debug(
            f"Overlap configuration: duration={overlap_duration:.1f}s, samples={overlap_samples}"
        )

        chunks = []
        chunk_times = []

        # Calculate step size
        step_size = int(samples_per_chunk * device_config["step_ratio"])
        logger.debug(f"Step size: {step_size/sr:.1f}s ({step_size} samples)")

        for i in range(0, len(speech), step_size):
            chunk_start = max(0, i)
            chunk_end = min(len(speech), i + samples_per_chunk)

            # Add overlap
            actual_start = max(0, chunk_start - overlap_samples)
            actual_end = min(len(speech), chunk_end + overlap_samples)

            chunks.append(speech[actual_start:actual_end])
            chunk_times.append((chunk_start / sr, min(chunk_end / sr, duration)))

            logger.debug(
                f"Created chunk {len(chunks)}: {chunk_start/sr:.2f}s - {chunk_end/sr:.2f}s"
            )

        # Log configuration
        logger.info("Chunk Processing Configuration:")
        logger.info(f"• Chunk Duration: {chunk_duration}s")
        logger.info(
            f"• Overlap: {overlap_duration:.1f}s ({device_config['overlap_ratio']*100:.0f}%)"
        )
        logger.info(
            f"• Step Size: {step_size/sr:.1f}s ({device_config['step_ratio']*100:.0f}% of chunk)"
        )
        logger.info(f"• Total Chunks: {len(chunks)}")
        logger.info(f"• Processing Mode: {self.device.upper()}")
        logger.info(f"• Batch Size: {self.batch_size}")

        # Print configuration
        pc.print_info("\nChunk Processing Configuration:")
        pc.print_info(f"• Chunk Duration: {chunk_duration}s")
        pc.print_info(
            f"• Overlap: {overlap_duration:.1f}s ({device_config['overlap_ratio']*100:.0f}%)"
        )
        pc.print_info(
            f"• Step Size: {step_size/sr:.1f}s ({device_config['step_ratio']*100:.0f}% of chunk)"
        )
        pc.print_info(f"• Total Chunks: {len(chunks)}")
        pc.print_info(f"• Processing Mode: {self.device.upper()}")
        pc.print_info(f"• Batch Size: {self.batch_size}")

        return chunks, chunk_times, chunk_duration

    def _prepare_audio_samples(self, audio):
        """Convert audio to numpy array and handle sample rate conversion

        Args:
            audio: pydub.AudioSegment to process

        Returns:
            tuple: (numpy array of audio samples, sample rate)
        """
        logger.info("Starting audio sample preparation")
        logger.debug(
            f"Input audio: {len(audio)}ms, {audio.frame_rate}Hz, {audio.channels} channels"
        )

        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        sr = audio.frame_rate
        logger.debug(f"Converted to numpy array: {len(samples)} samples")

        # Handle sample rate conversion
        target_sr = AUDIO_CONFIG["target_sample_rate"]
        if sr > target_sr:
            logger.info(f"Downsampling audio from {sr}Hz to {target_sr}Hz")
            pc.print_info(f"\nDownsampling audio from {sr} Hz to {target_sr} Hz")
            num_samples = round(len(samples) * target_sr / sr)
            samples = signal.resample(samples, num_samples)
            sr = target_sr
            logger.debug(f"Downsampled to {len(samples)} samples")
        elif sr < target_sr:
            logger.warning(
                f"Audio sample rate ({sr}Hz) is lower than model's expected rate ({target_sr}Hz)"
            )
            pc.print_warning(
                f"Warning: Audio sample rate ({sr} Hz) is lower than the model's expected rate ({target_sr} Hz). This may affect transcription quality"
            )

        logger.info(f"Audio preparation complete: {len(samples)} samples at {sr}Hz")
        return samples, sr

    def _normalize_volume(self, audio):
        """Normalize audio volume to target levels

        Args:
            audio: pydub.AudioSegment to normalize

        Returns:
            Normalized pydub.AudioSegment
        """
        logger.info("Starting volume normalization")
        pc.print_info("\nNormalizing audio volume:")

        # Get audio stats
        current_rms = audio.rms
        current_peak = float(np.max(np.abs(audio.get_array_of_samples())))
        logger.debug(f"Initial stats - RMS: {current_rms}, Peak: {current_peak}")

        if current_rms > 0 and current_peak > 0:  # Avoid division by zero
            # Calculate needed adjustments
            current_rms_db = 20 * np.log10(current_rms)
            current_peak_db = 20 * np.log10(
                current_peak / 32768.0
            )  # Normalize peak to 0 dB

            target_rms_db = AUDIO_CONFIG["audio"]["target_rms"]
            target_peak_db = AUDIO_CONFIG["audio"]["target_peak"]

            logger.debug(
                f"Current levels - RMS: {current_rms_db:.1f}dB, Peak: {current_peak_db:.1f}dB"
            )
            logger.debug(
                f"Target levels - RMS: {target_rms_db}dB, Peak: {target_peak_db}dB"
            )

            # Calculate required gain changes
            rms_gain = target_rms_db - current_rms_db
            peak_gain = target_peak_db - current_peak_db

            # Use the more conservative gain
            gain = min(rms_gain, peak_gain)

            # Apply gain limits
            gain = max(
                AUDIO_CONFIG["audio"]["min_gain"],
                min(AUDIO_CONFIG["audio"]["max_gain"], gain),
            )

            logger.debug(
                f"Calculated gain: {gain:.1f}dB (limited to [{AUDIO_CONFIG['audio']['min_gain']}, {AUDIO_CONFIG['audio']['max_gain']}])"
            )

            if abs(gain) > 0.1:  # Only adjust if change is significant
                logger.info(f"Applying gain adjustment of {gain:.1f}dB")
                pc.print_info(f"• Current RMS: {current_rms_db:.1f}dB")
                pc.print_info(f"• Current Peak: {current_peak_db:.1f}dB")
                pc.print_info(f"• Applying gain: {gain:.1f}dB")
                audio = audio.apply_gain(gain)
            else:
                logger.info("Gain adjustment too small, skipping normalization")
        else:
            logger.warning(
                "Invalid audio levels (zero RMS or peak), skipping normalization"
            )

        return audio

    def _apply_compression(self, audio):
        """Apply dynamic range compression to audio

        Args:
            audio: pydub.AudioSegment to compress

        Returns:
            Compressed pydub.AudioSegment
        """
        logger.info("Starting dynamic range compression")
        pc.print_info("Applying dynamic range compression")
        comp_settings = AUDIO_CONFIG["audio"]["compression"]
        logger.debug(f"Compression settings: {comp_settings}")

        # Convert audio to numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        logger.debug(f"Converted to numpy array: {len(samples)} samples")

        # Calculate volume in dB
        volume_db = 20 * np.log10(
            np.abs(samples) + 1e-10
        )  # Add small value to avoid log(0)

        # Set up compression parameters
        threshold_db = comp_settings["threshold"]
        ratio = comp_settings["ratio"]
        attack_time = comp_settings["attack"] / 1000.0  # Convert to seconds
        release_time = comp_settings["release"] / 1000.0

        attack_samples = int(attack_time * audio.frame_rate)
        release_samples = int(release_time * audio.frame_rate)
        logger.debug(
            f"Compression parameters - Attack: {attack_samples} samples, Release: {release_samples} samples"
        )

        # Apply compression
        gain_reduction = np.zeros_like(volume_db)
        above_threshold = volume_db > threshold_db
        gain_reduction[above_threshold] = (
            volume_db[above_threshold] - threshold_db
        ) * (1 - 1 / ratio)
        logger.debug(
            f"Applied initial gain reduction to {np.sum(above_threshold)} samples"
        )

        # Smooth gain reduction
        logger.debug("Applying smoothing to gain reduction")
        smoothed_reduction = np.zeros_like(gain_reduction)
        for i in range(1, len(gain_reduction)):
            if gain_reduction[i] > smoothed_reduction[i - 1]:
                smoothed_reduction[i] = (
                    smoothed_reduction[i - 1]
                    + (gain_reduction[i] - smoothed_reduction[i - 1]) / attack_samples
                )
            else:
                smoothed_reduction[i] = (
                    smoothed_reduction[i - 1]
                    + (gain_reduction[i] - smoothed_reduction[i - 1]) / release_samples
                )

        # Apply final gain reduction
        compressed_samples = samples * np.power(10, -smoothed_reduction / 20)
        logger.debug("Applied final gain reduction")

        # Create compressed audio
        compressed_audio = audio._spawn(compressed_samples.astype(np.int16))
        logger.info("Compression complete")

        pc.print_info(f"• Compression threshold: {threshold_db}dB")
        pc.print_info(f"• Ratio: {ratio}:1")
        pc.print_info(f"• Attack: {comp_settings['attack']}ms")
        pc.print_info(f"• Release: {comp_settings['release']}ms")

        return compressed_audio

    def _detect_repetition(self, text: str) -> bool:
        """Detect if text contains repetitive patterns

        Args:
            text: Text to check for repetitions

        Returns:
            bool: True if repetitive patterns found
        """
        logger.debug(f"Checking for repetitions in text of length {len(text)}")
        words = text.split()

        if len(words) < 4:
            logger.debug("Text too short for repetition check")
            return False

        # Check for immediate word repetition
        for i in range(len(words) - 1):
            if words[i].lower() == words[i + 1].lower():
                logger.info(f"Detected immediate word repetition: '{words[i]}'")
                return True

        # Check for phrase repetition
        for length in range(2, min(6, len(words) // 2)):  # Check phrases up to 5 words
            logger.debug(f"Checking for {length}-word phrase repetitions")
            for i in range(len(words) - length * 2):
                phrase1 = " ".join(words[i : i + length]).lower()
                phrase2 = " ".join(words[i + length : i + length * 2]).lower()
                if phrase1 == phrase2:
                    logger.info(f"Detected repeated phrase: '{phrase1}'")
                    return True

        logger.debug("No repetitions found")
        return False

    def _smart_merge_texts(self, text1: str, text2: str, overlap_words: int) -> str:
        """Advanced text merging with duplicate and phrase detection

        Args:
            text1: First text
            text2: Second text
            overlap_words: Number of words to check for overlap

        Returns:
            str: Merged text with duplicates removed
        """
        logger.info(f"Smart merging texts with {overlap_words} word overlap")
        logger.debug(f"Text1 ({len(text1)} chars), Text2 ({len(text2)} chars)")

        words1 = text1.split()
        words2 = text2.split()
        logger.debug(
            f"Split into words: Text1 ({len(words1)} words), Text2 ({len(words2)} words)"
        )

        # Check for repeated phrases
        for length in range(min(overlap_words, len(words1), len(words2)), 2, -1):
            logger.debug(f"Checking {length}-word phrases for overlap")
            for i in range(max(0, len(words1) - length), len(words1)):
                phrase1 = " ".join(words1[i:])
                for j in range(min(length, len(words2))):
                    phrase2 = " ".join(words2[: j + 1])
                    if phrase1.lower().endswith(phrase2.lower()):
                        logger.info(f"Found overlapping phrase: '{phrase2}'")
                        words2 = words2[j + 1 :]
                        break

        # Check for individual word duplicates
        filtered_words2 = []
        last_words = words1[-overlap_words:] if len(words1) >= overlap_words else words1
        logger.debug(f"Checking last {len(last_words)} words for duplicates")

        for w2 in words2:
            if not any(
                w2.lower() == w1.lower()
                or (
                    len(w2) > 3
                    and len(w1) > 3
                    and (
                        w2.lower().startswith(w1.lower())
                        or w1.lower().startswith(w2.lower())
                    )
                )
                for w1 in last_words
            ):
                filtered_words2.append(w2)
            else:
                logger.debug(f"Removed duplicate word: '{w2}'")

        merged_text = " ".join(words1 + filtered_words2)
        logger.info(f"Merge complete: {len(merged_text)} chars")
        return merged_text

    def _merge_transcriptions(self, text1, text2):
        """Merge two transcription texts with retry on repetition"""
        logger.info("Starting transcription merge")
        overlap_words = AUDIO_CONFIG["transcription"]["word_overlap"]
        logger.debug(f"Using {overlap_words} word overlap")

        merged = self._smart_merge_texts(text1, text2, overlap_words)
        logger.debug(f"Initial merge complete: {len(merged)} chars")

        # Check for repetitions and retry with smaller chunk if needed
        if self._detect_repetition(merged):
            logger.warning(
                "Detected repetition in merged text, retrying with adjustment"
            )
            pc.print_warning("Detected repetition, retrying with 1 second adjustment")

            # Trim words from the end of text1
            words_per_second = AUDIO_CONFIG["model"]["generation"]["words_per_second"]
            words1 = text1.split()
            if len(words1) > words_per_second:
                logger.info(f"Trimming {words_per_second} words from text1")
                text1_adjusted = " ".join(words1[:-words_per_second])
                merged = self._smart_merge_texts(text1_adjusted, text2, overlap_words)
                logger.debug(f"Adjusted merge complete: {len(merged)} chars")

        logger.info("Merge operation complete")
        return merged


# endregion Main
