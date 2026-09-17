#!/usr/bin/env python3
"""Diarize a 16 kHz mono WAV and stamp speaker labels onto whisper segments.

Runs under the repo's .venv (needs sherpa-onnx + numpy). Invoked by bin/ytx.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

SEG_MODEL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB_MODEL = "nemo_en_titanet_large.onnx"


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def read_wav16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as fh:
        if fh.getframerate() != 16000 or fh.getnchannels() != 1 or fh.getsampwidth() != 2:
            raise SystemExit(
                f"speakers.py: expected 16 kHz mono 16-bit WAV, got "
                f"{fh.getframerate()} Hz, {fh.getnchannels()} ch, {fh.getsampwidth() * 8} bit"
            )
        raw = fh.readframes(fh.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def diarize(samples: np.ndarray, models: Path, clusters: int, threshold: float):
    seg_model = models / SEG_MODEL
    emb_model = models / EMB_MODEL
    for path in (seg_model, emb_model):
        if not path.is_file():
            raise SystemExit(f"speakers.py: missing model {path} (run ./bootstrap.sh)")

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg_model)
            ),
            num_threads=4,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_model), num_threads=4
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=clusters if clusters > 1 else -1, threshold=threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    return list(engine.process(samples).sort_by_start_time())


def load_whisper_segments(path: Path) -> list[tuple[float, float, str]]:
    with path.open() as fh:
        data = json.load(fh)
    segments = []
    for entry in data.get("transcription", []):
        text = entry.get("text", "").strip()
        if not text:
            continue
        offsets = entry["offsets"]
        segments.append((offsets["from"] / 1000.0, offsets["to"] / 1000.0, text))
    return segments


def label_segments(segments, turns, max_gap=3.0):
    """Pick the speaker with the most overlap; fall back to the nearest turn."""
    labelled = []
    for start, end, text in segments:
        overlap: dict[int, float] = {}
        for turn in turns:
            shared = min(end, turn.end) - max(start, turn.start)
            if shared > 0:
                overlap[turn.speaker] = overlap.get(turn.speaker, 0.0) + shared
        if overlap:
            speaker = max(overlap, key=overlap.get)
        else:
            mid = (start + end) / 2
            nearest, distance = None, None
            for turn in turns:
                gap = min(abs(mid - turn.start), abs(mid - turn.end))
                if distance is None or gap < distance:
                    nearest, distance = turn.speaker, gap
            speaker = nearest if distance is not None and distance <= max_gap else None
        labelled.append((start, end, text, speaker))
    return labelled


def render(labelled):
    """Return (block_text, srt_text). Speakers are numbered by first appearance."""
    numbering: dict[int, int] = {}
    for _, _, _, speaker in labelled:
        if speaker is not None and speaker not in numbering:
            numbering[speaker] = len(numbering) + 1

    blocks, current, lines = [], None, []
    for _, _, text, speaker in labelled:
        name = f"Speaker {numbering[speaker]}" if speaker is not None else "Unknown"
        if name != current:
            if lines:
                blocks.append((current, lines))
            current, lines = name, []
        lines.append(text)
    if lines:
        blocks.append((current, lines))

    out = [f"{name}: {join_segments(words)}\n" for name, words in blocks]

    def stamp(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    srt = []
    for index, (start, end, text, speaker) in enumerate(labelled, start=1):
        name = f"Speaker {numbering[speaker]}" if speaker is not None else "Unknown"
        srt.append(f"{index}\n{stamp(start)} --> {stamp(end)}\n{name}: {text}\n")
    return "".join(out), "\n".join(srt)


def join_segments(segments) -> str:
    """Glue whisper's short segments into readable prose.

    Whisper splits on pauses, so segments often continue the previous sentence.
    Where the previous segment did not end in terminal punctuation, the next
    segment's leading capital is lowered so the result reads as one sentence.
    All-caps openers such as "AI" are left alone.
    """
    merged = ""
    for raw in segments:
        text = raw.strip()
        if not text:
            continue
        if merged:
            first_word = text.split(maxsplit=1)[0]
            is_acronym = len(first_word) > 1 and first_word.isupper()
            if merged.rstrip()[-1:] not in ".!?" and not is_acronym:
                text = text[0].lower() + text[1:]
            merged += " "
        merged += text
    if merged and merged[-1:] not in ".!?":
        merged += "."
    return merged


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out-txt", required=True)
    parser.add_argument("--out-srt", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--clusters", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args(argv)

    turns = diarize(
        read_wav16k(Path(args.wav)), Path(args.models), args.clusters, args.threshold
    )
    speakers = {turn.speaker for turn in turns}
    log(f"diarization: {len(turns)} turns, {len(speakers)} speakers detected")

    labelled = label_segments(load_whisper_segments(Path(args.json)), turns)
    transcript, srt = render(labelled)

    Path(args.out_txt).write_text(transcript)
    Path(args.out_srt).write_text(srt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
