#!/usr/bin/env python3
"""
Wake Word Clip Reviewer

Uses Gemma 4 (local, via Hugging Face Transformers) to listen to captured
wake word clips and classify them for training.

Requires:
    pip install transformers torch accelerate soundfile

Usage:
    python review_clips.py --clips-dir ./clips --output-dir ./clips/training_ready
    python review_clips.py --clips-dir ./clips --output-dir ./clips/training_ready --model google/gemma-4-E4B-it
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


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


def review_clip(model, processor, wav_path, wake_word):
    """Ask Gemma to listen to a clip and classify it."""

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

    # Decode only the new tokens (skip the input)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(new_tokens, skip_special_tokens=True).strip()

    return response


def parse_response(response_text):
    """Try to extract JSON from the model's response."""
    # Find JSON in the response
    try:
        # Try direct parse
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in the response
    import re
    match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract classification from text
    text_lower = response_text.lower()
    if "keep" in text_lower:
        return {"classification": "KEEP", "reason": response_text, "speech": "unknown", "quality": "unknown"}
    elif "discard" in text_lower:
        return {"classification": "DISCARD", "reason": response_text, "speech": "unknown", "quality": "unknown"}
    else:
        return {"classification": "MAYBE", "reason": response_text, "speech": "unknown", "quality": "unknown"}


def main():
    parser = argparse.ArgumentParser(description="Review wake word clips using Gemma 4")
    parser.add_argument("--clips-dir", type=str, default="./clips")
    parser.add_argument("--output-dir", type=str, default="./clips/training_ready")
    parser.add_argument("--wake-word", type=str, default="Hey Glitch")
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--min-size", type=int, default=50000,
                        help="Skip clips smaller than this (bytes) — likely corrupted partials")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all WAV files
    wav_files = sorted(clips_dir.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {clips_dir}")
        return

    # Filter out tiny/corrupted clips
    valid_clips = []
    skipped = 0
    for wav in wav_files:
        if wav.stat().st_size < args.min_size:
            print(f"  SKIP (too small, {wav.stat().st_size}B): {wav.name}")
            skipped += 1
        else:
            valid_clips.append(wav)

    print(f"\nFound {len(wav_files)} clips, {skipped} skipped as too small, {len(valid_clips)} to review\n")

    if not valid_clips:
        print("No clips to review.")
        return

    # Load model
    model, processor = load_model(args.model)

    # Review each clip
    results = []
    kept = 0

    for i, wav_path in enumerate(valid_clips):
        # Load metadata if available
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

        # Ask Gemma to review
        try:
            raw_response = review_clip(model, processor, wav_path, args.wake_word)
            parsed = parse_response(raw_response)
        except Exception as e:
            print(f"  ERROR: {e}")
            parsed = {"classification": "MAYBE", "reason": f"Error: {e}", "speech": "error", "quality": "error"}

        classification = parsed.get("classification", "MAYBE").upper()
        reason = parsed.get("reason", "")
        speech = parsed.get("speech", "?")
        quality = parsed.get("quality", "?")

        print(f"  Speech: {speech}  Quality: {quality}")
        print(f"  → {classification}: {reason}")

        if classification == "KEEP":
            kept += 1
            dest = output_dir / f"device_clip_{kept:02d}.wav"
            shutil.copy2(wav_path, dest)
            print(f"  ✅ Copied to {dest.name}")
        elif classification == "MAYBE":
            kept += 1
            dest = output_dir / f"device_clip_{kept:02d}_maybe.wav"
            shutil.copy2(wav_path, dest)
            print(f"  ⚠️  Copied (review recommended) to {dest.name}")

        results.append({
            "file": wav_path.name,
            "detection_type": det_type,
            "avg_probability": avg_prob,
            "max_probability": max_prob,
            "classification": classification,
            "speech": speech,
            "quality": quality,
            "reason": reason,
        })
        print()

    # Save review results
    results_path = output_dir / "review_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("=" * 60)
    print(f"SUMMARY")
    print(f"  Total reviewed: {len(valid_clips)}")
    print(f"  KEEP:    {sum(1 for r in results if r['classification'] == 'KEEP')}")
    print(f"  MAYBE:   {sum(1 for r in results if r['classification'] == 'MAYBE')}")
    print(f"  DISCARD: {sum(1 for r in results if r['classification'] == 'DISCARD')}")
    print(f"  Copied:  {kept} clips to {output_dir}")
    print(f"  Results: {results_path}")
    print()
    print(f"Next step: copy good clips to your trainer:")
    print(f"  cp {output_dir}/*.wav ~/microwakeword-trainer/personal_samples/")


if __name__ == "__main__":
    main()
