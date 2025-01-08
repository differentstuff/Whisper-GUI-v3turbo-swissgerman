from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import numpy as np
from typing import Dict, Any
import tqdm
import os

class WhisperModelHandler:
    def __init__(self):
        """Initialize the Swiss German Whisper model handler"""
        self.model_id = "nizarmichaud/whisper-large-v3-turbo-swissgerman"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        
    def load_model(self):
        """Load the model and processor"""
        if self.model is None:
            print("\nLoading Swiss German Whisper model...")
            print("Step 1/3: Loading processor...")
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            print("Step 2/3: Loading model...")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id,
                use_legacy_cache=False  # Avoid past_key_values deprecation warning
            )
            print(f"Step 3/3: Moving model to {self.device}...")
            self.model.to(self.device)
            print("Model loaded successfully!")
    
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
        
        print(f"\nStarting transcription of: {os.path.basename(audio_path)}")
        print("Loading audio file...")
        
        # Load audio with progress indicator
        speech, sr = librosa.load(audio_path, sr=16000)
        duration = len(speech) / sr
        print(f"Audio duration: {duration:.1f} seconds")
        
        print("Processing audio...")
        # Process audio
        input_features = self.processor(
            speech, 
            sampling_rate=sr, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        print("Generating transcription...")
        # Create attention mask
        attention_mask = torch.ones_like(input_features)
        
        # Generate tokens with progress indicator
        predicted_ids = self.model.generate(
            input_features,
            attention_mask=attention_mask,
            num_beams=5,
            temperature=0.0,
            do_sample=False,
            max_length=int(duration * 3),  # Roughly 3 tokens per second
            task="transcribe",  # Explicitly set to transcribe, not translate
            language="gsw",  # Swiss German language code
            return_timestamps=True
        )
        
        print("Decoding transcription...")
        # Decode transcription
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
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
