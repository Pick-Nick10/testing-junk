#!/usr/bin/env python3
"""
Wake Word Clip Trimmer

Reads review_results.json from the reviewer, takes KEEP/MAYBE clips,
and trims leading/trailing silence to produce tight training samples.

Uses energy-based voice activity detection to find where speech starts
and ends, then pads with a small margin.

Requires:
    pip install numpy soundfile

Usage:
    python trim_clips.py --input-dir ./clips/training_ready --output-dir ./clips/trimmed
    python trim_clips.py --input-dir ./clips/training_ready --output-dir ./clips/trimmed --margin-ms 200
"""

import argparse
import json
import wave
import struct
from datetime import datetime
from pathlib import Path

import numpy as np


def load_wav(path):
    """Load a WAV file and return (samples_int16[], sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32)
        samples = (samples >> 16).astype(np.int16)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        samples = samples[::n_channels]  # take first channel

    return samples, rate


def save_wav(path, samples, rate):
    """Save int16 samples as a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.astype(np.int16).tobytes())


def find_speech_bounds(samples, rate, frame_ms=20, energy_threshold_db=-35, min_speech_ms=100):
    """
    Find the start and end sample indices of speech in the audio.

    Uses short-time energy in dB. Speech frames are those above the threshold.
    Returns (start_sample, end_sample) of the speech region.
    """
    frame_size = int(rate * frame_ms / 1000)
    n_frames = len(samples) // frame_size

    if n_frames == 0:
        return 0, len(samples)

    # Compute per-frame energy in dB
    frame_energies = []
    for i in range(n_frames):
        frame = samples[i * frame_size : (i + 1) * frame_size].astype(np.float64)
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > 0:
            db = 20 * np.log10(rms / 32768.0)
        else:
            db = -100.0
        frame_energies.append(db)

    frame_energies = np.array(frame_energies)

    # Find frames above threshold
    speech_frames = frame_energies > energy_threshold_db

    if not np.any(speech_frames):
        # No speech found — return the whole clip
        return 0, len(samples)

    # Find first and last speech frame
    first_speech = np.argmax(speech_frames)
    last_speech = len(speech_frames) - 1 - np.argmax(speech_frames[::-1])

    # Require minimum speech duration
    min_frames = int(min_speech_ms / frame_ms)
    speech_count = np.sum(speech_frames[first_speech:last_speech + 1])
    if speech_count < min_frames:
        # Not enough speech — return whole clip
        return 0, len(samples)

    start_sample = first_speech * frame_size
    end_sample = min((last_speech + 1) * frame_size, len(samples))

    return start_sample, end_sample


def trim_clip(samples, rate, margin_ms=150, energy_threshold_db=-35):
    """
    Trim silence from the beginning and end of a clip.
    Keeps a margin around the detected speech region.
    """
    start, end = find_speech_bounds(samples, rate, energy_threshold_db=energy_threshold_db)

    margin_samples = int(rate * margin_ms / 1000)

    trim_start = max(0, start - margin_samples)
    trim_end = min(len(samples), end + margin_samples)

    return samples[trim_start:trim_end]


def main():
    parser = argparse.ArgumentParser(description="Trim silence from wake word clips")
    parser.add_argument("--input-dir", type=str, default="./clips/training_ready",
                        help="Directory with KEEP/MAYBE clips from the reviewer")
    parser.add_argument("--output-dir", type=str, default="./clips/trimmed",
                        help="Directory to write trimmed clips")
    parser.add_argument("--margin-ms", type=int, default=150,
                        help="Margin in ms to keep around detected speech (default: 150)")
    parser.add_argument("--threshold-db", type=float, default=-35,
                        help="Energy threshold in dB for speech detection (default: -35)")
    parser.add_argument("--min-duration-ms", type=int, default=300,
                        help="Discard clips shorter than this after trimming (default: 300)")
    parser.add_argument("--max-duration-ms", type=int, default=3000,
                        help="Cap clips at this duration (default: 3000)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return

    print(f"Trimming {len(wav_files)} clips from {input_dir}")
    print(f"  Margin: {args.margin_ms}ms")
    print(f"  Threshold: {args.threshold_db}dB")
    print(f"  Min duration: {args.min_duration_ms}ms")
    print(f"  Max duration: {args.max_duration_ms}ms")
    print()

    kept = 0
    discarded = 0
    results = []

    for wav_path in wav_files:
        samples, rate = load_wav(wav_path)
        original_ms = len(samples) / rate * 1000

        trimmed = trim_clip(samples, rate,
                           margin_ms=args.margin_ms,
                           energy_threshold_db=args.threshold_db)
        trimmed_ms = len(trimmed) / rate * 1000

        # Cap duration
        max_samples = int(rate * args.max_duration_ms / 1000)
        if len(trimmed) > max_samples:
            trimmed = trimmed[:max_samples]
            trimmed_ms = len(trimmed) / rate * 1000

        # Check minimum duration
        if trimmed_ms < args.min_duration_ms:
            print(f"  SKIP {wav_path.name}: trimmed to {trimmed_ms:.0f}ms (below {args.min_duration_ms}ms minimum)")
            discarded += 1
            results.append({
                "file": wav_path.name,
                "original_ms": round(original_ms),
                "trimmed_ms": round(trimmed_ms),
                "status": "discarded_too_short",
            })
            continue

        # Generate unique filename: batch timestamp + original name
        batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{batch_ts}_{wav_path.stem}.wav"
        out_path = output_dir / out_name
        save_wav(out_path, trimmed, rate)
        kept += 1

        reduction = (1 - trimmed_ms / original_ms) * 100
        print(f"  OK {wav_path.name}: {original_ms:.0f}ms → {trimmed_ms:.0f}ms ({reduction:.0f}% trimmed)")

        results.append({
            "file": wav_path.name,
            "original_ms": round(original_ms),
            "trimmed_ms": round(trimmed_ms),
            "status": "trimmed",
        })

    # Save results
    results_path = output_dir / "trim_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Move processed raw clips to an archive folder so they don't get re-processed
    archive_dir = Path(args.input_dir).parent / "processed"
    archive_dir.mkdir(parents=True, exist_ok=True)

    archived = 0
    raw_clips_dir = Path(args.input_dir).parent
    for wav_path in sorted(raw_clips_dir.glob("*.wav")):
        dest = archive_dir / wav_path.name
        wav_path.rename(dest)
        # Move matching JSON too
        json_path = wav_path.with_suffix(".json")
        if json_path.exists():
            json_path.rename(archive_dir / json_path.name)
        archived += 1

    # Also move the training_ready clips to archive
    for wav_path in sorted(input_dir.glob("*.wav")):
        dest = archive_dir / wav_path.name
        wav_path.rename(dest)
        json_path = wav_path.with_suffix(".json")
        if json_path.exists():
            json_path.rename(archive_dir / json_path.name)
    # Move review_results.json too
    review_results = input_dir / "review_results.json"
    if review_results.exists():
        review_results.rename(archive_dir / f"review_results_{batch_ts}.json")

    print()
    print(f"Done: {kept} trimmed, {discarded} discarded, {archived} raw clips archived")
    print(f"Trimmed clips: {output_dir}")
    print(f"Archived originals: {archive_dir}")
    print(f"Results: {results_path}")
    print()
    print(f"Next step: copy trimmed clips to your trainer:")
    print(f"  cp {output_dir}/*.wav ~/microwakeword-trainer/personal_samples/")


if __name__ == "__main__":
    main()
