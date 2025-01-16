"""
Handles timestamp formatting for transcription output.
"""

from typing import List, Tuple, Dict, Any


def format_with_timestamps(transcription_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format transcription output with timestamps.

    Args:
        transcription_data: Dictionary containing transcription results with segments

    Returns:
        Dictionary with formatted text including timestamps
    """
    segments = transcription_data["segments"]

    # Format each segment with timestamps
    formatted_text = "\n".join(
        f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}"
        for segment in segments
    )

    # Update the text while preserving segments
    result = transcription_data.copy()
    result["text"] = formatted_text

    return result


def format_without_timestamps(transcription_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format transcription output without timestamps, merging segments.

    Args:
        transcription_data: Dictionary containing transcription results with segments

    Returns:
        Dictionary with merged text without timestamps
    """
    segments = transcription_data["segments"]

    # Merge all segment texts
    merged_text = " ".join(segment["text"] for segment in segments)

    # Update the text while preserving segments
    result = transcription_data.copy()
    result["text"] = merged_text

    return result
