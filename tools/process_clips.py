#!/usr/bin/env python3
"""
Wake Word Clip Processor

End-to-end pipeline: review clips with Gemma 4, trim silence, archive originals.

1. Scans clips directory for new WAV files
2. Uses Gemma 4 to listen and classify each clip (KEEP/MAYBE/DISCARD)
3. Trims silence from kept clips using energy-based VAD
4. Saves trimmed clips with unique timestamped names
5. Archives all processed originals so they aren't re-processed

Requires:
    pip install transformers torch accelerate soundfile librosa numpy

Usage:
    python process_clips.py
    python process_clips.py --clips-dir ./clips --output-dir ./clips/trimmed
    python process_clips.py --model google/gemma-4-E2B-it   # smaller model
    python process_clips.py --threshold-db -40 --margin-ms 250  # looser trim
"""

import argparse
import json
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


# ─── Model ───────────────────────────────────────────────────────────────────

def load_model(model_id):
    print(f"Loading model: {model_id}")
    print("  (this may take a few minutes on first run to download weights)")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"  Model loaded on: {model.device}")
    return model, processor


# ─── Review ──────────────────────────────────────────────────────────────────

def review_clip(model, processor, wav_path, wake_word):
    prompt = f"""Listen to this audio clip carefully. I need you to classify it for wake word training data.

The wake word is: "{wake_word}"

Answer these questions:
1. SPEECH: Does someone say "{wake_word}" (or something very similar) in this clip? (yes/no/unclear)
2. COUNT: How many times is the wake word spoken in this clip? (0, 1, 2, or more)
3. QUALITY: Is the audio quality usable? Not corrupted, not pure silence, not pure noise? (good/acceptable/bad)
4. CLASSIFICATION: Based on the above, should this clip be used for training?
   - KEEP: Someone clearly says the wake word EXACTLY ONCE, audio quality is acceptable or better
   - MAYBE: Wake word might be present but unclear, or quality is borderline
   - DISCARD: No wake word present, wake word is said MORE THAN ONCE, or audio is unusable

IMPORTANT: A good training sample must contain the wake word spoken exactly ONE time. If the wake word is said twice or more in the clip, classify as DISCARD.

Respond in this exact JSON format:
{{"speech": "yes/no/unclear", "count": 0, "quality": "good/acceptable/bad", "classification": "KEEP/MAYBE/DISCARD", "reason": "brief explanation"}}"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": str(wav_path)},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def parse_response(response_text):
    import re
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    text_lower = response_text.lower()
    if "keep" in text_lower:
        return {"classification": "KEEP", "reason": response_text, "speech": "unknown", "quality": "unknown"}
    elif "discard" in text_lower:
        return {"classification": "DISCARD", "reason": response_text, "speech": "unknown", "quality": "unknown"}
    return {"classification": "MAYBE", "reason": response_text, "speech": "unknown", "quality": "unknown"}


# ─── Trim ────────────────────────────────────────────────────────────────────

def load_wav(path):
    with wave.open(str(path), "rb") as wf:
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32)
        samples = (samples >> 16).astype(np.int16)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        samples = samples[::n_channels]

    return samples, rate


def save_wav(path, samples, rate):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.astype(np.int16).tobytes())


def find_speech_bounds_adaptive(samples, rate, frame_ms=20):
    """
    Adaptive spike detection: finds speech by looking for energy rises above
    a rolling local baseline rather than using a fixed dB threshold.
    """
    frame_size = int(rate * frame_ms / 1000)
    n_frames = len(samples) // frame_size

    if n_frames == 0:
        return 0, len(samples)

    energies = []
    for i in range(n_frames):
        frame = samples[i * frame_size : (i + 1) * frame_size].astype(np.float64)
        rms = np.sqrt(np.mean(frame ** 2))
        db = 20 * np.log10(rms / 32768.0) if rms > 0 else -100.0
        energies.append(db)
    energies = np.array(energies)

    # Rolling median baseline (500ms window = 25 frames)
    baseline = np.array([
        np.median(energies[max(0, i - 12):min(n_frames, i + 13)])
        for i in range(n_frames)
    ])

    rise_above_baseline = energies - baseline
    top_30_threshold = np.percentile(energies, 70)

    # Speech = frames that spike >6dB above local baseline AND are in the top 30%
    speech_mask = (rise_above_baseline > 6) & (energies > top_30_threshold)

    # Fill 1-frame gaps between speech regions
    for i in range(1, n_frames - 1):
        if speech_mask[i - 1] and speech_mask[i + 1]:
            speech_mask[i] = True

    if not np.any(speech_mask):
        return 0, len(samples)

    # Find contiguous speech regions, discard tiny ones (< 3 frames / 60ms)
    MIN_REGION_FRAMES = 3
    regions = []
    in_region = False
    region_start = 0
    for i in range(n_frames):
        if speech_mask[i] and not in_region:
            region_start = i
            in_region = True
        elif not speech_mask[i] and in_region:
            if (i - 1) - region_start + 1 >= MIN_REGION_FRAMES:
                regions.append((region_start, i - 1))
            in_region = False
    if in_region:
        if (n_frames - 1) - region_start + 1 >= MIN_REGION_FRAMES:
            regions.append((region_start, n_frames - 1))

    if not regions:
        # No substantial speech regions found — return whole clip untrimmed
        return 0, len(samples)

    # If multiple regions separated by significant silence (>800ms = 40 frames),
    # keep only the LAST contiguous group (the one that triggered detection).
    # 800ms threshold preserves the natural "Hey ... Glitch" pause (~100-200ms)
    # while splitting truly separate utterances (700ms+).
    SILENCE_GAP_FRAMES = 40  # 800ms at 20ms/frame
    if len(regions) > 1:
        # Find the last major silence gap and keep everything after it
        last_big_gap_idx = -1
        for i in range(len(regions) - 1):
            gap = regions[i + 1][0] - regions[i][1]
            if gap >= SILENCE_GAP_FRAMES:
                last_big_gap_idx = i

        if last_big_gap_idx >= 0:
            # Keep all regions after the last big gap
            kept_regions = regions[last_big_gap_idx + 1:]
        else:
            # No big gaps — all regions are one utterance
            kept_regions = regions

        first = kept_regions[0][0]
        last = kept_regions[-1][1]
    else:
        first = regions[0][0]
        last = regions[0][1]

    return first * frame_size, min((last + 1) * frame_size, len(samples))


def find_speech_bounds_probability(prob_timeline, prob_step_ms, rate, threshold=0.1):
    """
    Find speech bounds using the model's probability timeline.
    The wake word is where probabilities rise above the threshold.
    Returns (start_sample, end_sample) in audio sample indices.
    """
    if not prob_timeline:
        return None, None

    # Find first and last frame where probability exceeds threshold
    first_idx = None
    last_idx = None
    for i, p in enumerate(prob_timeline):
        if p > threshold:
            if first_idx is None:
                first_idx = i
            last_idx = i

    if first_idx is None:
        return None, None

    # Convert probability frame indices to audio sample indices
    # Each probability frame corresponds to prob_step_ms of audio
    samples_per_step = int(rate * prob_step_ms / 1000)
    start_sample = first_idx * samples_per_step
    end_sample = (last_idx + 1) * samples_per_step

    return start_sample, end_sample


def trim_clip(samples, rate, margin_before_ms=300, margin_after_ms=500,
              max_duration_ms=3000, prob_timeline=None, prob_step_ms=10):
    """
    Trim a clip. Uses probability timeline if available (most accurate),
    otherwise falls back to energy-based spike detection.
    """
    start_sample = None
    end_sample = None

    # Try probability-based trimming first
    if prob_timeline:
        start_sample, end_sample = find_speech_bounds_probability(
            prob_timeline, prob_step_ms, rate, threshold=0.1)

    # Fallback to energy-based spike detection
    if start_sample is None or end_sample is None:
        start_sample, end_sample = find_speech_bounds_adaptive(samples, rate)

    margin_before = int(rate * margin_before_ms / 1000)
    margin_after = int(rate * margin_after_ms / 1000)
    trim_start = max(0, start_sample - margin_before)
    trim_end = min(len(samples), end_sample + margin_after)

    trimmed = samples[trim_start:trim_end]

    max_samples = int(rate * max_duration_ms / 1000)
    if len(trimmed) > max_samples:
        trimmed = trimmed[:max_samples]

    return trimmed


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Process wake word clips: review → trim → archive")
    parser.add_argument("--clips-dir", type=str, default="./clips",
                        help="Directory with raw clips from clip_receiver.py")
    parser.add_argument("--output-dir", type=str, default="./clips/trimmed",
                        help="Directory for final trimmed clips")
    parser.add_argument("--wake-word", type=str, default="Hey Glitch")
    parser.add_argument("--model", type=str, default="google/gemma-4-E4B-it")
    parser.add_argument("--min-size", type=int, default=50000,
                        help="Skip clips smaller than this (bytes)")
    parser.add_argument("--margin-before-ms", type=int, default=300,
                        help="Padding before detected speech (ms)")
    parser.add_argument("--margin-after-ms", type=int, default=500,
                        help="Padding after detected speech (ms)")
    parser.add_argument("--min-duration-ms", type=int, default=300,
                        help="Discard trimmed clips shorter than this (ms)")
    parser.add_argument("--max-duration-ms", type=int, default=3000,
                        help="Cap trimmed clip duration (ms)")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    output_dir = Path(args.output_dir)
    archive_dir = clips_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Find new clips ──
    wav_files = sorted(clips_dir.glob("*.wav"))
    valid_clips = []
    skipped = 0

    for wav in wav_files:
        if wav.stat().st_size < args.min_size:
            print(f"  SKIP (too small, {wav.stat().st_size}B): {wav.name}")
            skipped += 1
            # Archive the runt
            wav.rename(archive_dir / wav.name)
            json_path = wav.with_suffix(".json")
            if json_path.exists():
                json_path.rename(archive_dir / json_path.name)
        else:
            valid_clips.append(wav)

    if not valid_clips:
        print(f"No new clips to process in {clips_dir}")
        return

    print(f"\nFound {len(wav_files)} clips, {skipped} skipped as too small, {len(valid_clips)} to review\n")

    # ── Load model ──
    model, processor = load_model(args.model)
    print()

    # ── Review + Trim ──
    results = []
    output_count = 0

    for i, wav_path in enumerate(valid_clips):
        # Load metadata
        json_path = wav_path.with_suffix(".json")
        meta = {}
        if json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)

        det_type = meta.get("detection_type", "unknown")
        avg_prob = meta.get("avg_probability", 0)
        max_prob = meta.get("max_probability", 0)

        print(f"[{i+1}/{len(valid_clips)}] {wav_path.name}")
        print(f"  Type: {det_type}  Prob: avg={avg_prob:.2f} max={max_prob:.2f}")

        # Review with Gemma
        try:
            raw_response = review_clip(model, processor, wav_path, args.wake_word)
            parsed = parse_response(raw_response)
        except Exception as e:
            print(f"  ERROR reviewing: {e}")
            parsed = {"classification": "MAYBE", "reason": f"Error: {e}", "speech": "error", "quality": "error"}

        classification = parsed.get("classification", "MAYBE").upper()
        reason = parsed.get("reason", "")
        speech = parsed.get("speech", "?")
        quality = parsed.get("quality", "?")
        count = parsed.get("count", "?")

        print(f"  Speech: {speech}  Count: {count}  Quality: {quality}")
        print(f"  Review: {classification} — {reason}")

        result = {
            "file": wav_path.name,
            "detection_type": det_type,
            "avg_probability": avg_prob,
            "max_probability": max_prob,
            "classification": classification,
            "speech": speech,
            "count": count,
            "quality": quality,
            "reason": reason,
        }

        if classification in ("KEEP", "MAYBE"):
            # Load probability timeline from metadata if available
            prob_timeline = meta.get("probability_timeline")
            prob_step_ms = meta.get("prob_step_ms", 10)
            trim_method = "probability curve" if prob_timeline else "energy spike detection"

            try:
                samples, rate = load_wav(wav_path)
                original_ms = len(samples) / rate * 1000

                trimmed = trim_clip(samples, rate,
                                    margin_before_ms=args.margin_before_ms,
                                    margin_after_ms=args.margin_after_ms,
                                    max_duration_ms=args.max_duration_ms,
                                    prob_timeline=prob_timeline,
                                    prob_step_ms=prob_step_ms)
                trimmed_ms = len(trimmed) / rate * 1000

                if trimmed_ms < args.min_duration_ms:
                    print(f"  DISCARD after trim: {trimmed_ms:.0f}ms (below {args.min_duration_ms}ms)")
                    result["trim_status"] = "too_short"
                else:
                    output_count += 1
                    suffix = "_maybe" if classification == "MAYBE" else ""
                    out_name = f"{batch_ts}_clip_{output_count:02d}{suffix}.wav"
                    out_path = output_dir / out_name
                    save_wav(out_path, trimmed, rate)

                    reduction = (1 - trimmed_ms / original_ms) * 100
                    print(f"  Trimmed: {original_ms:.0f}ms → {trimmed_ms:.0f}ms ({reduction:.0f}% removed) [{trim_method}]")
                    print(f"  ✅ Saved: {out_name}")
                    result["trim_status"] = "saved"
                    result["output_file"] = out_name
                    result["original_ms"] = round(original_ms)
                    result["trimmed_ms"] = round(trimmed_ms)

            except Exception as e:
                print(f"  ERROR trimming: {e}")
                result["trim_status"] = f"error: {e}"
        else:
            result["trim_status"] = "skipped"

        results.append(result)
        print()

    # ── Archive originals ──
    archived = 0
    for wav_path in valid_clips:
        dest = archive_dir / wav_path.name
        if wav_path.exists():
            wav_path.rename(dest)
            archived += 1
        json_path = wav_path.with_suffix(".json")
        if json_path.exists():
            json_path.rename(archive_dir / json_path.name)

    # ── Save results ──
    results_path = output_dir / f"results_{batch_ts}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ──
    n_keep = sum(1 for r in results if r["classification"] == "KEEP")
    n_maybe = sum(1 for r in results if r["classification"] == "MAYBE")
    n_discard = sum(1 for r in results if r["classification"] == "DISCARD")

    print("=" * 60)
    print("SUMMARY")
    print(f"  Reviewed:  {len(valid_clips)}")
    print(f"  KEEP:      {n_keep}")
    print(f"  MAYBE:     {n_maybe}")
    print(f"  DISCARD:   {n_discard}")
    print(f"  Trimmed:   {output_count} clips saved to {output_dir}")
    print(f"  Archived:  {archived} originals moved to {archive_dir}")
    print(f"  Results:   {results_path}")
    print()
    print("Next step — copy trimmed clips to your trainer (WSL):")
    print(f"  cp /mnt/c/Users/Nick/Homelab_CoWork/Satellite1-ESPHome/{output_dir}/*.wav ~/microwakeword-trainer/personal_samples/")


if __name__ == "__main__":
    main()
