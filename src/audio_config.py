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