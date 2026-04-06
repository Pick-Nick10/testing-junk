#!/usr/bin/env python3
"""
Wake Word Clip Receiver

Listens for UDP packets from the Satellite1 micro_wake_word clip capture
feature and saves them as WAV files with JSON metadata.

Usage:
    python clip_receiver.py [--port 12345] [--output-dir ./clips]

Protocol:
    First packet (chunk_index=0) has a 56-byte header:
      Bytes 0-3:    Magic "MWWC"
      Byte  4:      Protocol version (1)
      Byte  5:      Detection type: 0=detected, 1=near_miss, 2=vad_blocked
      Byte  6:      Max probability (uint8, 0-255)
      Byte  7:      Average probability (uint8, 0-255)
      Bytes 8-9:    Sample rate (uint16 LE)
      Bytes 10-11:  Bits per sample (uint16 LE)
      Bytes 12-15:  Total samples (uint32 LE)
      Bytes 16-17:  Total chunks (uint16 LE)
      Bytes 18-19:  Chunk index (uint16 LE) = 0
      Bytes 20-23:  Clip ID (uint32 LE)
      Bytes 24-55:  Wake word name (32 bytes, null-padded)
      Bytes 56+:    First chunk of PCM audio data

    Subsequent packets have a 10-byte header:
      Bytes 0-3:    Magic "MWWC"
      Bytes 4-7:    Clip ID (uint32 LE)
      Bytes 8-9:    Chunk index (uint16 LE)
      Bytes 10+:    PCM audio data
"""

import argparse
import json
import socket
import struct
import time
import wave
from datetime import datetime
from pathlib import Path

MAGIC = b"MWWC"
DETECTION_TYPES = {0: "detected", 1: "near_miss", 2: "vad_blocked"}

FIRST_HEADER_SIZE = 56
DATA_HEADER_SIZE = 10
CLIP_TIMEOUT_SECONDS = 5.0


class ClipAssembler:
    def __init__(self, clip_id, metadata):
        self.clip_id = clip_id
        self.metadata = metadata
        self.chunks = {}
        self.total_chunks = metadata["total_chunks"]
        self.created_at = time.time()

    def add_chunk(self, chunk_index, audio_data):
        self.chunks[chunk_index] = audio_data

    def is_complete(self):
        return len(self.chunks) >= self.total_chunks

    def is_expired(self):
        return (time.time() - self.created_at) > CLIP_TIMEOUT_SECONDS

    def assemble(self):
        audio = bytearray()
        for i in range(self.total_chunks):
            if i in self.chunks:
                audio.extend(self.chunks[i])
        return bytes(audio)


def parse_first_packet(data):
    if len(data) < FIRST_HEADER_SIZE or data[:4] != MAGIC:
        return None, None, None

    version = data[4]
    if version != 1:
        return None, None, None

    det_type = data[5]
    max_prob = data[6]
    avg_prob = data[7]
    sample_rate = struct.unpack_from("<H", data, 8)[0]
    bits_per_sample = struct.unpack_from("<H", data, 10)[0]
    total_samples = struct.unpack_from("<I", data, 12)[0]
    total_chunks = struct.unpack_from("<H", data, 16)[0]
    clip_id = struct.unpack_from("<I", data, 20)[0]
    wake_word = data[24:56].split(b"\x00")[0].decode("utf-8", errors="replace")

    metadata = {
        "detection_type": DETECTION_TYPES.get(det_type, f"unknown_{det_type}"),
        "max_probability": round(max_prob / 255.0, 3),
        "avg_probability": round(avg_prob / 255.0, 3),
        "sample_rate": sample_rate,
        "bits_per_sample": bits_per_sample,
        "total_samples": total_samples,
        "total_chunks": total_chunks,
        "clip_id": clip_id,
        "wake_word": wake_word,
    }

    audio_data = data[FIRST_HEADER_SIZE:]
    return clip_id, metadata, audio_data


def parse_data_packet(data):
    if len(data) < DATA_HEADER_SIZE or data[:4] != MAGIC:
        return None, None, None

    clip_id = struct.unpack_from("<I", data, 4)[0]
    chunk_index = struct.unpack_from("<H", data, 8)[0]
    audio_data = data[DATA_HEADER_SIZE:]

    return clip_id, chunk_index, audio_data


def save_clip(assembler, output_dir):
    meta = assembler.metadata
    audio = assembler.assemble()

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = (
        f"{timestamp}_{meta['wake_word']}_{meta['detection_type']}"
        f"_avg{meta['avg_probability']:.2f}_max{meta['max_probability']:.2f}.wav"
    )

    filepath = output_dir / filename

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(meta["bits_per_sample"] // 8)
        wf.setframerate(meta["sample_rate"])
        wf.writeframes(audio)

    meta_path = filepath.with_suffix(".json")
    meta_copy = dict(meta)
    meta_copy["filename"] = filename
    meta_copy["timestamp"] = timestamp
    meta_copy["received_chunks"] = len(assembler.chunks)
    meta_copy["audio_bytes"] = len(audio)
    with open(meta_path, "w") as f:
        json.dump(meta_copy, f, indent=2)

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Wake Word Clip Receiver")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--output-dir", type=str, default="./clips")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1.0)

    print(f"Listening for wake word clips on UDP port {args.port}")
    print(f"Saving clips to: {output_dir.resolve()}")

    active_clips = {}

    try:
        while True:
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                data = None

            if data and len(data) >= FIRST_HEADER_SIZE and data[:4] == MAGIC:
                chunk_index_at_18 = struct.unpack_from("<H", data, 18)[0]
                if chunk_index_at_18 == 0 and data[4] == 1:
                    clip_id, metadata, audio = parse_first_packet(data)
                    if clip_id is not None and clip_id not in active_clips:
                        assembler = ClipAssembler(clip_id, metadata)
                        assembler.add_chunk(0, audio)
                        active_clips[clip_id] = assembler
                        print(
                            f"  [{addr[0]}] New clip {clip_id:#010x}: "
                            f"{metadata['wake_word']} "
                            f"({metadata['detection_type']}) "
                            f"prob={metadata['avg_probability']:.2f} "
                            f"chunks=0/{metadata['total_chunks']}"
                        )
                        data = None  # consumed

            if data and len(data) >= DATA_HEADER_SIZE and data[:4] == MAGIC:
                clip_id, chunk_index, audio = parse_data_packet(data)
                if clip_id is not None and clip_id in active_clips:
                    active_clips[clip_id].add_chunk(chunk_index, audio)

            # Check for complete or expired clips
            completed = []
            for cid, assembler in list(active_clips.items()):
                if assembler.is_complete():
                    filepath = save_clip(assembler, output_dir)
                    print(f"  Saved complete clip: {filepath.name}")
                    completed.append(cid)
                elif assembler.is_expired():
                    filepath = save_clip(assembler, output_dir)
                    pct = len(assembler.chunks) / max(assembler.total_chunks, 1) * 100
                    print(
                        f"  Saved partial clip ({pct:.0f}% received): {filepath.name}"
                    )
                    completed.append(cid)

            for cid in completed:
                del active_clips[cid]

    except KeyboardInterrupt:
        print("\nShutting down...")
        # Save any in-progress clips
        for cid, assembler in active_clips.items():
            filepath = save_clip(assembler, output_dir)
            pct = len(assembler.chunks) / max(assembler.total_chunks, 1) * 100
            print(f"  Saved in-progress clip ({pct:.0f}%): {filepath.name}")
        print(f"Done. {len(active_clips)} clip(s) saved on exit.")


if __name__ == "__main__":
    main()
