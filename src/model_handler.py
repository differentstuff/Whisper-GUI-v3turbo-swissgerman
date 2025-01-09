from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import numpy as np
from typing import Dict, Any
import tqdm
import os
import psutil

class WhisperModelHandler:
    def __init__(self, device=None):
        """Initialize the Swiss German Whisper model handler"""
        self.model_id = "nizarmichaud/whisper-large-v3-turbo-swissgerman"
        
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
        self.cache_dir = os.path.join(os.path.dirname(__file__), "..", "model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set processing parameters based on device
        self.chunk_length = 30 if self.device == "cuda" else 15
        self.batch_size = 8 if self.device == "cuda" else 1
        
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
                print(f"✓ Processor downloaded ({after_mem - before_mem:.1f}MB)")

                print("\nStep 2/4: Downloading model (~3GB)...")
                print("This may take several minutes on first run.")
                print("Downloading from: " + self.model_id)
                if self.device == "cuda":
                    print("Checking available VRAM...")
                    free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2
                    if free_memory < 4000:  # Need at least 4GB free
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
                if str(actual_device) != self.device:
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
    
    def transcribe(self, audio_path: str, language: str = None, verbose: bool = True, **kwargs) -> Dict[str, Any]:
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
        import numpy as np
        from scipy import signal
        
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
        
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        sr = audio.frame_rate
        speech = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        if audio.channels > 1:
            speech = speech.reshape((-1, audio.channels)).mean(axis=1)
        
        # Downsample to 16000 Hz if necessary
        if sr > 16000:
            print(f"Downsampling audio from {sr} Hz to 16000 Hz...")
            num_samples = round(len(speech) * 16000 / sr)
            speech = signal.resample(speech, num_samples)
            sr = 16000
        elif sr < 16000:
            print(f"Warning: Audio sample rate ({sr} Hz) is lower than the model's expected rate (16000 Hz). This may affect transcription quality.")
        
        duration = len(speech) / sr
        print(f"Audio duration: {duration:.1f} seconds")
        
        # Report memory usage
        if self.device == "cuda":
            print(f"Current VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        
        # Determine chunk size based on time
        chunk_duration = 30  # 30 seconds per chunk
        samples_per_chunk = int(chunk_duration * sr)
        
        # Split audio into chunks with overlap
        overlap = int(10 * sr)  # 10 seconds overlap
        chunks = [speech[max(0, i-overlap):i+samples_per_chunk] for i in range(0, len(speech), samples_per_chunk-overlap)]
        print(f"Processing audio in {len(chunks)} chunks of {chunk_duration} seconds each (with 10s overlap)")
        
        all_transcriptions = []
        
        for i, chunk in enumerate(chunks, 1):
            try:
                print(f"\nProcessing chunk {i}/{len(chunks)}")
                chunk_start_time = max(0, (i-1) * (chunk_duration - 10))
                chunk_end_time = min(i * chunk_duration, duration)
                print(f"Chunk time range: {chunk_start_time:.2f}s - {chunk_end_time:.2f}s")
                
                # Process audio chunk
                input_features = self.processor(
                    chunk, 
                    sampling_rate=sr, 
                    return_tensors="pt"
                ).input_features
                
                # Move to device and set correct data type
                input_features = input_features.to(self.device)
                if self.device == "cuda":
                    input_features = input_features.to(torch.float16)
                else:
                    input_features = input_features.to(torch.float32)
                
                # Clear memory before generation
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Generate transcription
                print(f"Generating transcription for chunk {i}...")
                predicted_ids = self.model.generate(
                    input_features,
                    do_sample=False,
                    max_length=int(chunk_duration * 3),  # Adjust max_length based on chunk duration
                    language="de",
                    task="transcribe"
                )
                
                # Clear memory after generation
                del input_features
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Decode transcription
                chunk_text = self.processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0].strip()
                
                if chunk_text:
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
        
        # Combine all transcriptions
        print("\nCombining transcriptions...")
        combined_transcriptions = []
        for i, item in enumerate(all_transcriptions):
            if item is None:
                print(f"Warning: Chunk {i+1} produced no transcription")
                continue
            start, end, text = item
            if i == 0:
                combined_transcriptions.append((start, end, text))
            else:
                if not combined_transcriptions:
                    combined_transcriptions.append((start, end, text))
                else:
                    prev_start, prev_end, prev_text = combined_transcriptions[-1]
                    if start < prev_end:
                        # Overlap detected, merge the transcriptions
                        merged_text = self._merge_transcriptions(prev_text, text)
                        combined_transcriptions[-1] = (prev_start, end, merged_text)
                    else:
                        combined_transcriptions.append((start, end, text))
        
        if not combined_transcriptions:
            print("Error: No valid transcriptions were produced")
            return {"text": "", "segments": [], "language": "Swiss German"}
        
        transcription = " ".join([text for _, _, text in combined_transcriptions])
        
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
            "language": "Swiss German"
        }
        
        return result
    
    def _merge_transcriptions(self, text1, text2):
        # Simple merging strategy: split into words and remove duplicates
        words1 = text1.split()
        words2 = text2.split()
        merged = words1 + [w for w in words2 if w not in words1[-5:]]  # Check last 5 words for duplicates
        return " ".join(merged)
