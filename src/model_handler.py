from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import numpy as np
from typing import Dict, Any
import tqdm
import os

class WhisperModelHandler:
    def __init__(self, device=None):
        """Initialize the Swiss German Whisper model handler"""
        self.model_id = "nizarmichaud/whisper-large-v3-turbo-swissgerman"
        
        # Initialize CUDA if available
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
            free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2
            print(f"\nCUDA initialized:")
            print(f"- Total VRAM: {total_memory:.0f}MB")
            print(f"- Free VRAM: {free_memory:.0f}MB")
            print(f"- GPU: {torch.cuda.get_device_name(0)}")
            
            # Check if we have enough VRAM for GPU mode
            if free_memory < 4000:  # Need at least 4GB
                print(f"Warning: Limited GPU memory ({free_memory:.0f}MB). Using CPU for better reliability.")
                self.device = "cpu"
            else:
                print(f"Sufficient GPU memory ({free_memory:.0f}MB). Using CUDA.")
                self.device = "cuda"
        else:
            print("\nNo CUDA device found. Using CPU.")
            self.device = "cpu"
            
        if device:  # Override automatic selection if device explicitly specified
            self.device = device
            print(f"Device explicitly set to: {self.device}")
        
        # Initialize model settings
        self.processor = None
        self.model = None
        self.cache_dir = os.path.join(os.path.dirname(__file__), "..", "model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set lower memory usage for CPU
        if self.device == "cpu":
            self.chunk_length = 15  # Process in 15-second chunks
            self.batch_size = 1
        else:
            self.chunk_length = 30  # Larger chunks for GPU
            self.batch_size = 8
            
        print(f"Model will be cached in: {self.cache_dir}")
        
    def load_model(self):
        """Load the model and processor"""
        if self.model is None:
            try:
                print("\nStep 1/4: Downloading processor...")
                before_mem = torch.cuda.memory_allocated() / 1024**2 if self.device == "cuda" else 0
                self.processor = WhisperProcessor.from_pretrained(
                    self.model_id,
                    cache_dir=self.cache_dir,
                    local_files_only=False  # Allow download if not present
                )
                after_mem = torch.cuda.memory_allocated() / 1024**2 if self.device == "cuda" else 0
                print(f"Processor downloaded successfully (Memory change: {after_mem - before_mem:.1f}MB)")

                print("\nStep 2/4: Downloading model (this will take a while, ~3GB)...")
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
                        use_legacy_cache=False,    # Avoid deprecation warning
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
                            use_legacy_cache=False,
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
        import librosa
        
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
        
        # Load audio with progress indicator
        speech, sr = librosa.load(audio_path, sr=16000)
        duration = len(speech) / sr
        print(f"Audio duration: {duration:.1f} seconds")
        
        # Report memory usage
        if self.device == "cuda":
            print(f"Current VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        
        print("Processing audio in chunks...")
        
        # Split audio into chunks
        chunk_length_samples = int(self.chunk_length * sr)
        chunks = [speech[i:i + chunk_length_samples] for i in range(0, len(speech), chunk_length_samples)]
        
        all_transcriptions = []
        total_chunks = len(chunks)
        print(f"Split into {total_chunks} chunks of {self.chunk_length} seconds each")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\nProcessing chunk {i}/{total_chunks}")
            chunk_start_time = i * self.chunk_length - self.chunk_length
            print(f"Time range: {chunk_start_time:.1f}s - {min(chunk_start_time + self.chunk_length, duration):.1f}s")
            
            try:
                # Process audio chunk with memory tracking
                before_mem = torch.cuda.memory_allocated() / 1024**2 if self.device == "cuda" else 0
                
                print(f"Processing audio features for chunk {i}...")
                input_features = self.processor(
                    chunk, 
                    sampling_rate=sr, 
                    return_tensors="pt"
                ).input_features
                
                # Move to device and create attention mask
                input_features = input_features.to(self.device)
                attention_mask = torch.ones_like(input_features)
                
                # Clear memory before generation
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Generate tokens with progress indicator
                print(f"Generating transcription for chunk {i}...")
                predicted_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    num_beams=5,
                    temperature=0.0,
                    do_sample=False,
                    max_length=int(self.chunk_length * 3),  # ~3 tokens per second
                    task="transcribe",
                    language="gsw",
                    return_timestamps=True
                )
                
                # Clear memory after generation
                del input_features
                del attention_mask
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                after_mem = torch.cuda.memory_allocated() / 1024**2 if self.device == "cuda" else 0
                if self.device == "cuda":
                    print(f"Memory usage for chunk {i}: {after_mem - before_mem:.1f}MB")
                    torch.cuda.empty_cache()  # Clear unused memory
                
                # Decode transcription
                chunk_text = self.processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0].strip()
                
                if chunk_text:
                    all_transcriptions.append(chunk_text)
                    print(f"Chunk {i} complete ({len(chunk_text)} chars)")
                    print(f"Text: {chunk_text[:100]}...")
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
        transcription = " ".join(all_transcriptions)
        
        print("Transcription complete!")
        
        # Format output to match Whisper's format
        duration = len(speech) / sr
        result = {
            "text": transcription,
            "segments": [{
                "id": 0,
                "start": 0,
                "end": duration,
                "text": transcription
            }],
            "language": "Swiss German"
        }
        
        return result
