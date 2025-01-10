from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import numpy as np
from typing import Dict, Any
import tqdm
import os
import psutil
from scipy import signal

# Audio Processing Configuration
# This configuration controls how audio is processed and transcribed.
# Adjusting these parameters can help optimize transcription quality and performance.
AUDIO_CONFIG = {
    # Target sample rate for audio processing
    # Whisper models expect 16000 Hz audio. Higher rates will be downsampled.
    'target_sample_rate': 16000,  # Hz
    
    # Audio normalization settings
    'audio': {
        'normalize': True,          # Whether to apply volume normalization
        'target_rms': -18,          # Target RMS level in dB (increased for better signal level)
        'target_peak': -2,          # Target peak level in dB (increased for better dynamics)
        'headroom': 0.1,            # Headroom to prevent clipping (0.1 = 10% below max)
        'min_gain': -24,            # Maximum reduction in dB (increased to match compression)
        'max_gain': 24,             # Maximum amplification in dB (balanced with reduction)
        'compression': {            # Dynamic range compression settings
            'enabled': False,        # Whether to apply compression
            'threshold': -24,       # Threshold in dB where compression begins
                                    # Lower threshold to catch more dynamic range
            'ratio': 2.5,           # Compression ratio (2.5:1)
                                    # Slightly stronger compression for better consistency
            'attack': 10,           # Attack time in milliseconds
                                    # Slower attack to preserve transients
            'release': 100          # Release time in milliseconds
                                    # Longer release for smoother transitions
        }
    },
    
    # Chunk processing configuration for GPU and CPU modes
    'chunk': {
        'gpu': {
            'duration': 30,         # Length of each audio chunk in seconds
                                    # Standard chunk size for good context
            'overlap_ratio': 1/5,   # How much chunks overlap (1/5 = 6 seconds for 30s chunks)
                                    # Balanced overlap for context preservation
            'step_ratio': 0.85,     # How far to advance between chunks (0.85 = 25.5s steps)
                                    # Conservative step size for stability
            'batch_size': 6         # Number of chunks to process at once
                                    # Reduced for better memory efficiency with overlaps
        },
        'cpu': {
            'duration': 15,         # Shorter chunks for CPU to manage memory
            'overlap_ratio': 1/5,   # Same overlap ratio as GPU
            'step_ratio': 0.85,     # Same step ratio as GPU
            'batch_size': 1         # Process one chunk at a time on CPU
        }
    },
    
    # Model generation parameters
    'model': {
        'max_length_multiplier': 1.75, # Maximum output length relative to chunk duration
                                      # Optimized for 25% chunk overlap
        'min_gpu_memory': 4000,     # Minimum GPU memory required in MB
                                    # Below this, will fall back to CPU
        'generation': {             # Generation settings for transcription
            'do_sample': False,     # Disable sampling for more stable output
            'num_beams': 1,         # Use greedy search for faster, stable results
            'early_stopping': False, # Disable early stopping with single beam
            'repetition_penalty': 1.2, # Mild repetition penalty
            'no_repeat_ngram_size': 2, # Conservative n-gram blocking
            'retry_on_repetition': True, # Whether to retry chunks with repetition
            'words_per_second': 2,  # Approximate words per second for retry adjustment
        }
    },
    
    # Text processing settings
    'transcription': {
    'word_overlap': 12,             # Number of words to check when removing duplicates
                                    # Increased to better detect and remove duplicates
    }
}

class WhisperModelHandler:
    def __init__(self, device=None, model_id=None):
        """Initialize the Whisper model handler
        
        Args:
            device: Device to use (cuda or cpu)
            model_id: Model ID to use (e.g., "openai/whisper-large-v3").
        """
        self.model_id = model_id
        
        # Initialize CUDA if requested
        if device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.init()
                torch.cuda.empty_cache()
                print("\nGPU Configuration:")
                print(f"• Device: {torch.cuda.get_device_name(0)}")
                print(f"• VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
                print(f"• Free VRAM: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2:.0f}MB")
                self.device = "cuda"
            except Exception as e:
                print(f"\nError initializing CUDA: {str(e)}")
                print("Falling back to CPU")
                self.device = "cpu"
        else:
            self.device = "cpu"
            print("\nUsing CPU mode")
        
        # Initialize model settings
        self.processor = None
        self.model = None
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache") # use os.path.dirname twice to go one level up instead of using '..'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set processing parameters based on device configuration
        device_config = AUDIO_CONFIG['chunk']['gpu' if self.device == "cuda" else 'cpu']
        self.chunk_length = device_config['duration']
        self.batch_size = device_config['batch_size']
        
        print("\nModel Settings:")
        print(f"• Device: {self.device.upper()}")
        print(f"• Cache Directory: {self.cache_dir}")
        print(f"• Chunk Length: {self.chunk_length}s")
        print(f"• Batch Size: {self.batch_size}")
        
    def load_model(self):
        """Load the model and processor"""
        if self.model is None:
            try:
                print("\n=== Loading Model Components ===")
                print("Step 1/4: Downloading processor...")
                before_mem = torch.cuda.memory_allocated() / 1024**2 if self.device == "cuda" else 0
                self.processor = WhisperProcessor.from_pretrained(
                    self.model_id,
                    cache_dir=self.cache_dir,
                    local_files_only=False  # Allow download if not present
                )
                after_mem = torch.cuda.memory_allocated() / 1024**2 if self.device == "cuda" else 0
                print(f"> Processor downloaded ({after_mem - before_mem:.1f}MB)")

                print("\nStep 2/4: Downloading model (~3GB)...")
                print("This may take several minutes on first run.")
                print("Downloading from: " + self.model_id)
                if self.device == "cuda":
                    free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2
                    if free_memory < AUDIO_CONFIG['model']['min_gpu_memory']:  # Check against configured minimum
                        print(f"Warning: Only {free_memory:.0f}MB VRAM available. Switching to CPU...")
                        self.device = "cpu"
                
                before_mem = torch.cuda.memory_allocated() / 1024**2 if self.device == "cuda" else 0
                try:
                    # Clear any existing cached memory
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        
                    # Load model with optimized settings
                    self.model = WhisperForConditionalGeneration.from_pretrained(
                        self.model_id,
                        cache_dir=self.cache_dir,
                        local_files_only=False,    # Allow download if not present
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,  # Use float16 for GPU
                        device_map=self.device,    # Explicitly set device
                        low_cpu_mem_usage=True,    # More memory efficient
                        offload_folder=os.path.join(self.cache_dir, "offload")  # Enable model offloading
                    )
                    
                    # Verify model loaded correctly
                    if self.model is None:
                        raise RuntimeError("Model failed to load")
                        
                except Exception as e:
                    print(f"\nError loading model on {self.device}: {str(e)}")
                    if self.device == "cuda":
                        print("Trying with CPU fallback...")
                        self.device = "cpu"
                        torch.cuda.empty_cache()  # Clear GPU memory
                        
                        self.model = WhisperForConditionalGeneration.from_pretrained(
                            self.model_id,
                            local_files_only=False,
                            torch_dtype=torch.float32,
                            cache_dir=self.cache_dir,
                            device_map="cpu",
                            low_cpu_mem_usage=True,
                            offload_folder=os.path.join(self.cache_dir, "offload")
                        )
                    else:
                        raise  # Re-raise the error if already on CPU
                after_mem = torch.cuda.memory_allocated() / 1024**2 if self.device == "cuda" else 0
                print(f"Model downloaded successfully (Memory change: {after_mem - before_mem:.1f}MB)")

                print("\nStep 3/4: Verifying model location...")
                actual_device = next(self.model.parameters()).device
                # Check if model is on any CUDA device when CUDA is requested
                is_on_correct_device = (
                    (self.device == "cuda" and actual_device.type == "cuda") or
                    (self.device == "cpu" and actual_device.type == "cpu")
                )
                if not is_on_correct_device:
                    print(f"Warning: Model is on {actual_device}, expected {self.device}")
                    if self.device == "cuda":
                        print("Forcing model to GPU...")
                        self.model.cuda()
                else:
                    print(f"Model correctly loaded on {actual_device}")

                print("\nStep 4/4: Verifying model state...")
                if self.model is None or self.processor is None:
                    raise RuntimeError("Model or processor not loaded properly")
                
                print(f"Model loaded successfully on {self.device}!")
                if self.device == "cuda":
                    print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
                else:
                    import psutil
                    print(f"RAM usage: {psutil.Process().memory_info().rss / 1024**2:.1f}MB")

            except Exception as e:
                print(f"\nError loading model: {str(e)}")
                print("Please check your internet connection and try again.")
                raise
    
    def transcribe(self, audio_path: str, language: str = "Auto", verbose: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio file using the Swiss German specialized model
        
        Args:
            audio_path: Path to audio file
            language: Language code (ignored as this is Swiss German specialized)
            verbose: Whether to show progress bar
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            Dictionary containing transcription results
        """
        from pydub import AudioSegment
        
        # Ensure model is loaded
        if self.model is None:
            print("\nModel not loaded. Loading now...")
            self.load_model()
            if self.model is None:
                raise RuntimeError("Failed to load model")
            if self.device == "cuda":
                print(f"Model loaded successfully on GPU. VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            else:
                print(f"Model loaded successfully on CPU. RAM usage: {psutil.Process().memory_info().rss / 1024**2:.1f}MB")
        
        print(f"\nStarting transcription of: {os.path.basename(audio_path)}")
        print("Loading audio file...")
        
        # Get file size
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")
        
        # Load and normalize audio file
        audio = AudioSegment.from_file(audio_path)
        sr = audio.frame_rate
        
        # Convert to mono if needed
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Apply compression if enabled
        if AUDIO_CONFIG['audio']['compression']['enabled']:
            audio = self._apply_compression(audio)
        
        # Normalize volume if enabled
        if AUDIO_CONFIG['audio']['normalize']:
            audio = self._normalize_volume(audio)
        
        # Convert audio to numpy array and handle sample rate
        speech, sr = self._prepare_audio_samples(audio)
        
        duration = len(speech) / sr
        print(f"Audio duration: {duration:.1f} seconds")
        
        # Report memory usage
        if self.device == "cuda":
            print(f"Current VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        
        # Create chunks for processing
        chunks, chunk_times, chunk_duration = self._create_chunks(speech, sr, duration)
        
        # Process chunks and get transcriptions
        all_transcriptions = self._process_chunks(chunks, chunk_times, chunk_duration, sr, language)
        
        # Combine transcriptions and create result
        combined_transcriptions, transcription = self._combine_transcriptions(all_transcriptions)
        if not combined_transcriptions:
            return {"text": "", "segments": [], "language": language}
        
        print("Transcription complete!")
        print(f"Total transcription length: {len(transcription)} characters")
        print(f"Processed {len(chunks)} chunks, covering {duration:.2f} seconds of audio")
        print(f"Full transcription: {transcription}")
        
        # Format output to match Whisper's format
        result = {
            "text": transcription,
            "segments": [{
                "id": i,
                "start": start,
                "end": end,
                "text": text
            } for i, (start, end, text) in enumerate(combined_transcriptions)],
            "language": language
        }
        
        return result
    
    def _process_chunks(self, chunks: list, chunk_times: list, chunk_duration: float, sr: int, language: str) -> list:
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
        all_transcriptions = []
        
        for i, (chunk, (chunk_start_time, chunk_end_time)) in enumerate(zip(chunks, chunk_times), 1):
            try:
                print(f"\nProcessing chunk {i}/{len(chunks)}")
                print(f"Chunk time range: {chunk_start_time:.2f}s - {chunk_end_time:.2f}s")
                
                # Process audio chunk with attention mask
                processed = self.processor(
                    chunk, 
                    sampling_rate=sr, 
                    return_tensors="pt",
                    return_attention_mask=True  # Suppress the warning for language detection
                )
                
                # Move to device and set correct data type
                input_features = processed.input_features.to(self.device)
                attention_mask = processed.attention_mask.to(self.device) if hasattr(processed, 'attention_mask') else None
                
                if self.device == "cuda":
                    input_features = input_features.to(torch.float16)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(torch.float16)
                else:
                    input_features = input_features.to(torch.float32)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(torch.float32)
                
                # Clear memory before generation
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Generate transcription
                print(f"Generating transcription for chunk {i}...")
                # Get generation settings
                gen_config = AUDIO_CONFIG['model']['generation']
                predicted_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    do_sample=gen_config['do_sample'],
                    repetition_penalty=gen_config['repetition_penalty'],
                    no_repeat_ngram_size=gen_config['no_repeat_ngram_size'],
                    num_beams=gen_config['num_beams'],
                    early_stopping=gen_config['early_stopping'],
                    max_length=int(chunk_duration * AUDIO_CONFIG['model']['max_length_multiplier']),
                    language=None if language == "Auto" else language.lower()
                )
                
                # Clear memory after generation
                del input_features
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Decode transcription and handle repetitions
                chunk_text = self.processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0].strip()
                
                if chunk_text:
                    # Check for repetitions in the current chunk
                    if AUDIO_CONFIG['model']['generation']['retry_on_repetition'] and self._detect_repetition(chunk_text):
                        print("Detected repetition in chunk, retrying with adjusted boundaries...")
                        # Adjust chunk boundaries by 1 second
                        samples_per_second = sr
                        adjusted_chunk = chunk[samples_per_second:-samples_per_second] if len(chunk) > 2*samples_per_second else chunk
                        
                        # Reprocess with adjusted chunk
                        processed = self.processor(
                            adjusted_chunk,
                            sampling_rate=sr,
                            return_tensors="pt",
                            return_attention_mask=True
                        )
                        
                        # Move to device and process
                        input_features = processed.input_features.to(self.device)
                        attention_mask = processed.attention_mask.to(self.device) if hasattr(processed, 'attention_mask') else None
                        
                        if self.device == "cuda":
                            input_features = input_features.to(torch.float16)
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(torch.float16)
                        else:
                            input_features = input_features.to(torch.float32)
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(torch.float32)
                        
                        # Generate new transcription
                        # Use same generation settings for retry
                        predicted_ids = self.model.generate(
                            input_features,
                            attention_mask=attention_mask,
                            do_sample=gen_config['do_sample'],
                            repetition_penalty=gen_config['repetition_penalty'],
                            no_repeat_ngram_size=gen_config['no_repeat_ngram_size'],
                            num_beams=gen_config['num_beams'],
                            early_stopping=gen_config['early_stopping'],
                            max_length=int(chunk_duration * AUDIO_CONFIG['model']['max_length_multiplier']),
                            language=None if language == "Auto" else language.lower()
                        )
                        
                        # Update chunk text with retry result
                        chunk_text = self.processor.batch_decode(
                            predicted_ids,
                            skip_special_tokens=True
                        )[0].strip()
                        
                        print("Retry complete")
                    
                    all_transcriptions.append((chunk_start_time, chunk_end_time, chunk_text))
                    print(f"Chunk {i} complete ({len(chunk_text)} chars)")
                    print(f"Text: {chunk_text}")
                else:
                    print(f"Warning: Chunk {i} produced no text")
                
            except torch.cuda.OutOfMemoryError:
                print(f"\nError: GPU out of memory on chunk {i}. Try using CPU mode.")
                raise
            except Exception as e:
                print(f"\nError processing chunk {i}: {str(e)}")
                raise
        
        return all_transcriptions

    def _combine_transcriptions(self, transcriptions: list) -> tuple:
        """Combine chunk transcriptions and handle overlaps
        
        Args:
            transcriptions: List of (start_time, end_time, text) tuples
            
        Returns:
            tuple: (list of combined segments, full transcription text)
        """
        print("\nCombining transcriptions...")
        combined = []
        
        for i, item in enumerate(transcriptions):
            if item is None:
                print(f"Warning: Chunk {i+1} produced no transcription")
                continue
                
            start, end, text = item
            if i == 0:
                combined.append((start, end, text))
            else:
                if not combined:
                    combined.append((start, end, text))
                else:
                    prev_start, prev_end, prev_text = combined[-1]
                    if start < prev_end:
                        # Overlap detected, merge the transcriptions
                        merged_text = self._merge_transcriptions(prev_text, text)
                        combined[-1] = (prev_start, end, merged_text)
                    else:
                        combined.append((start, end, text))
        
        if not combined:
            print("Error: No valid transcriptions were produced")
            return [], ""
            
        # Create full transcription from combined segments
        transcription = " ".join([text for _, _, text in combined])
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
        # Get chunk configuration based on device
        device_config = AUDIO_CONFIG['chunk']['gpu' if self.device == "cuda" else 'cpu']
        chunk_duration = device_config['duration']
        samples_per_chunk = int(chunk_duration * sr)
        
        # Calculate overlap based on configuration
        overlap_duration = chunk_duration * device_config['overlap_ratio']
        overlap_samples = int(overlap_duration * sr)
        chunks = []
        chunk_times = []
        
        # Calculate step size based on configuration
        step_size = int(samples_per_chunk * device_config['step_ratio'])
        
        for i in range(0, len(speech), step_size):
            chunk_start = max(0, i)
            chunk_end = min(len(speech), i + samples_per_chunk)
            
            # Add overlap to current chunk
            actual_start = max(0, chunk_start - overlap_samples)
            actual_end = min(len(speech), chunk_end + overlap_samples)
            
            chunks.append(speech[actual_start:actual_end])
            chunk_times.append((
                chunk_start / sr,  # Start time without overlap
                min(chunk_end / sr, duration)  # End time without overlap
            ))
        
        # Print detailed chunk processing configuration
        print("\nChunk Processing Configuration:")
        print(f"• Chunk Duration: {chunk_duration}s")
        print(f"• Overlap: {overlap_duration:.1f}s ({device_config['overlap_ratio']*100:.0f}%)")
        print(f"• Step Size: {step_size/sr:.1f}s ({device_config['step_ratio']*100:.0f}% of chunk)")
        print(f"• Total Chunks: {len(chunks)}")
        print(f"• Processing Mode: {self.device.upper()}")
        print(f"• Batch Size: {self.batch_size}")
        
        return chunks, chunk_times, chunk_duration

    def _prepare_audio_samples(self, audio):
        """Convert audio to numpy array and handle sample rate conversion
        
        Args:
            audio: pydub.AudioSegment to process
            
        Returns:
            tuple: (numpy array of audio samples, sample rate)
        """
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        sr = audio.frame_rate
        
        # Handle sample rate conversion
        target_sr = AUDIO_CONFIG['target_sample_rate']
        if sr > target_sr:
            print(f"Downsampling audio from {sr} Hz to {target_sr} Hz...")
            num_samples = round(len(samples) * target_sr / sr)
            samples = signal.resample(samples, num_samples)
            sr = target_sr
        elif sr < target_sr:
            print(f"Warning: Audio sample rate ({sr} Hz) is lower than the model's expected rate ({target_sr} Hz). This may affect transcription quality.")
        
        return samples, sr

    def _normalize_volume(self, audio):
        """Normalize audio volume to target levels
        
        Args:
            audio: pydub.AudioSegment to normalize
            
        Returns:
            Normalized pydub.AudioSegment
        """
        print("Normalizing audio volume...")
        
        # Get audio stats
        current_rms = audio.rms
        current_peak = float(np.max(np.abs(audio.get_array_of_samples())))
        
        if current_rms > 0 and current_peak > 0:  # Avoid division by zero
            # Calculate needed adjustments
            current_rms_db = 20 * np.log10(current_rms)
            current_peak_db = 20 * np.log10(current_peak / 32768.0)  # Normalize peak to 0 dB
            
            target_rms_db = AUDIO_CONFIG['audio']['target_rms']
            target_peak_db = AUDIO_CONFIG['audio']['target_peak']
            
            # Calculate required gain changes
            rms_gain = target_rms_db - current_rms_db
            peak_gain = target_peak_db - current_peak_db
            
            # Use the more conservative gain to avoid clipping
            gain = min(rms_gain, peak_gain)
            
            # Apply gain limits
            gain = max(AUDIO_CONFIG['audio']['min_gain'], 
                     min(AUDIO_CONFIG['audio']['max_gain'], gain))
            
            if abs(gain) > 0.1:  # Only adjust if change is significant
                print(f"• Current RMS: {current_rms_db:.1f}dB")
                print(f"• Current Peak: {current_peak_db:.1f}dB")
                print(f"• Applying gain: {gain:.1f}dB")
                audio = audio.apply_gain(gain)
        
        return audio

    def _apply_compression(self, audio):
        """Apply dynamic range compression to audio
        
        Args:
            audio: pydub.AudioSegment to compress
            
        Returns:
            Compressed pydub.AudioSegment
        """
        print("Applying dynamic range compression...")
        comp_settings = AUDIO_CONFIG['audio']['compression']
        
        # Convert audio to numpy array for compression
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        
        # Calculate volume in dB
        volume_db = 20 * np.log10(np.abs(samples) + 1e-10)  # Add small value to avoid log(0)
        
        # Calculate gain reduction
        threshold_db = comp_settings['threshold']
        ratio = comp_settings['ratio']
        attack_time = comp_settings['attack'] / 1000.0  # Convert to seconds
        release_time = comp_settings['release'] / 1000.0
        
        # Calculate time constants
        attack_samples = int(attack_time * audio.frame_rate)
        release_samples = int(release_time * audio.frame_rate)
        
        # Initialize gain reduction array
        gain_reduction = np.zeros_like(volume_db)
        
        # Apply compression
        above_threshold = volume_db > threshold_db
        gain_reduction[above_threshold] = (volume_db[above_threshold] - threshold_db) * (1 - 1/ratio)
        
        # Smooth gain reduction with attack/release
        smoothed_reduction = np.zeros_like(gain_reduction)
        for i in range(1, len(gain_reduction)):
            if gain_reduction[i] > smoothed_reduction[i-1]:  # Attack
                smoothed_reduction[i] = smoothed_reduction[i-1] + (gain_reduction[i] - smoothed_reduction[i-1]) / attack_samples
            else:  # Release
                smoothed_reduction[i] = smoothed_reduction[i-1] + (gain_reduction[i] - smoothed_reduction[i-1]) / release_samples
        
        # Apply smoothed gain reduction
        compressed_samples = samples * np.power(10, -smoothed_reduction/20)
        
        # Create new AudioSegment with compressed audio
        compressed_audio = audio._spawn(compressed_samples.astype(np.int16))
        
        print(f"• Compression threshold: {threshold_db}dB")
        print(f"• Ratio: {ratio}:1")
        print(f"• Attack: {comp_settings['attack']}ms")
        print(f"• Release: {comp_settings['release']}ms")
        
        return compressed_audio

    def _detect_repetition(self, text: str) -> bool:
        """Detect if text contains repetitive patterns
        
        Args:
            text: Text to check for repetitions
            
        Returns:
            bool: True if repetitive patterns found
        """
        words = text.split()
        if len(words) < 4:  # Too short to check
            return False
            
        # Check for immediate word repetition
        for i in range(len(words)-1):
            if words[i].lower() == words[i+1].lower():
                return True
        
        # Check for phrase repetition
        for length in range(2, min(6, len(words)//2)):  # Check phrases up to 5 words
            for i in range(len(words)-length*2):
                phrase1 = ' '.join(words[i:i+length]).lower()
                phrase2 = ' '.join(words[i+length:i+length*2]).lower()
                if phrase1 == phrase2:
                    return True
        
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
        words1 = text1.split()
        words2 = text2.split()
        
        # Check for repeated phrases
        for length in range(min(overlap_words, len(words1), len(words2)), 2, -1):
            for i in range(max(0, len(words1) - length), len(words1)):
                phrase1 = ' '.join(words1[i:])
                for j in range(min(length, len(words2))):
                    phrase2 = ' '.join(words2[:j+1])
                    if phrase1.lower().endswith(phrase2.lower()):
                        words2 = words2[j+1:]
                        break
        
        # Check for individual word duplicates
        filtered_words2 = []
        last_words = words1[-overlap_words:] if len(words1) >= overlap_words else words1
        
        for w2 in words2:
            if not any(w2.lower() == w1.lower() or 
                      (len(w2) > 3 and len(w1) > 3 and 
                       (w2.lower().startswith(w1.lower()) or 
                        w1.lower().startswith(w2.lower())))
                      for w1 in last_words):
                filtered_words2.append(w2)
        
        return ' '.join(words1 + filtered_words2)

    def _merge_transcriptions(self, text1, text2):
        """Merge two transcription texts with retry on repetition"""
        overlap_words = AUDIO_CONFIG['transcription']['word_overlap']
        merged = self._smart_merge_texts(text1, text2, overlap_words)
        
        # Check for repetitions and retry with smaller chunk if needed
        if self._detect_repetition(merged):
            print("Detected repetition, retrying with 1 second adjustment...")
            # Trim words from the end of text1 based on configured words per second
            words_per_second = AUDIO_CONFIG['model']['generation']['words_per_second']
            words1 = text1.split()
            if len(words1) > words_per_second:
                text1_adjusted = ' '.join(words1[:-words_per_second])
                merged = self._smart_merge_texts(text1_adjusted, text2, overlap_words)
        
        return merged
