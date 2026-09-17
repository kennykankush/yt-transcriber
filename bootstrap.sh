#!/usr/bin/env bash
# Sets up yt-transcriber: the speaker-labelling environment and the model files.
# Safe to re-run; it skips anything already present.
#
#   ./bootstrap.sh                      # everything except the 1.6 GB whisper model
#   ./bootstrap.sh --with-whisper-model # also download ggml-large-v3-turbo.bin
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="$REPO/models"
SHERPA="$MODELS/sherpa"

WITH_WHISPER=no
if [[ "${1:-}" == "--with-whisper-model" ]]; then
  WITH_WHISPER=yes
fi

ok()   { printf '  ok    %s\n' "$1"; }
miss() { printf '  MISS  %s\n' "$1"; }
note() { printf '  ..    %s\n' "$1"; }

echo
echo "yt-transcriber bootstrap"
echo "========================"

echo
echo "Command-line tools"
for tool in ffmpeg yt-dlp; do
  if command -v "$tool" >/dev/null 2>&1; then
    ok "$tool ($(command -v "$tool"))"
  else
    miss "$tool - install with: brew install $tool"
  fi
done

WHISPER_CLI="${YTX_WHISPER_CLI:-$HOME/dev/whisperccp/build/bin/whisper-cli}"
if [[ -x "$WHISPER_CLI" ]]; then
  ok "whisper-cli ($WHISPER_CLI)"
elif command -v whisper-cli >/dev/null 2>&1; then
  ok "whisper-cli ($(command -v whisper-cli))"
else
  miss "whisper-cli - build https://github.com/ggml-org/whisper.cpp, then set YTX_WHISPER_CLI"
fi

echo
echo "Whisper models  (searched: \$YTX_MODELS, $MODELS, ~/dev/whisperccp/models)"
found_model=""
for candidate in \
  "${YTX_MODELS:-/nonexistent}/ggml-large-v3-turbo.bin" \
  "$MODELS/ggml-large-v3-turbo.bin" \
  "$HOME/dev/whisperccp/models/ggml-large-v3-turbo.bin" \
  "$MODELS/ggml-medium.bin" \
  "$HOME/dev/whisperccp/models/ggml-medium.bin"; do
  if [[ -f "$candidate" ]]; then
    found_model="$candidate"
    break
  fi
done

if [[ -n "$found_model" ]]; then
  ok "found $(basename "$found_model")"
elif [[ "$WITH_WHISPER" == "yes" ]]; then
  mkdir -p "$MODELS"
  note "downloading ggml-large-v3-turbo.bin (1.6 GB) to $MODELS"
  curl -L --fail --progress-bar \
    -o "$MODELS/ggml-large-v3-turbo.bin" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
  ok "downloaded ggml-large-v3-turbo.bin"
else
  miss "no whisper model found"
  note "re-run with --with-whisper-model, or copy an existing .bin into $MODELS"
fi

VAD="$MODELS/ggml-silero-v6.2.0.bin"
if [[ -f "$VAD" || -f "$HOME/dev/whisperccp/models/ggml-silero-v6.2.0.bin" ]]; then
  ok "silero VAD model present"
else
  mkdir -p "$MODELS"
  note "downloading silero VAD model (~1 MB)"
  curl -L --fail -s \
    -o "$VAD" \
    "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin"
  ok "downloaded ggml-silero-v6.2.0.bin"
fi

echo
echo "Speaker-labelling models  ($SHERPA)"
SEG="$SHERPA/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB="$SHERPA/nemo_en_titanet_large.onnx"

if [[ -f "$SEG" ]]; then
  ok "pyannote segmentation model present"
else
  mkdir -p "$SHERPA"
  note "downloading pyannote segmentation model (~6 MB)"
  curl -L --fail -s -o "$SHERPA/seg.tar.bz2" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  tar xjf "$SHERPA/seg.tar.bz2" -C "$SHERPA"
  rm -f "$SHERPA/seg.tar.bz2"
  ok "downloaded segmentation model"
fi

if [[ -f "$EMB" ]]; then
  ok "speaker embedding model present"
else
  mkdir -p "$SHERPA"
  note "downloading speaker embedding model (~101 MB)"
  curl -L --fail -s -o "$EMB" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_large.onnx"
  ok "downloaded speaker embedding model"
fi

echo
echo "Python environment  ($REPO/.venv)"
if [[ -x "$REPO/.venv/bin/python" ]]; then
  ok "venv already exists"
else
  if command -v uv >/dev/null 2>&1; then
    uv venv "$REPO/.venv" --python 3.12 >/dev/null
  else
    python3 -m venv "$REPO/.venv"
  fi
  ok "created venv"
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$REPO/.venv/bin/python" -q -r "$REPO/requirements-speakers.txt"
else
  "$REPO/.venv/bin/pip" install -q -r "$REPO/requirements-speakers.txt"
fi
ok "sherpa-onnx installed"

echo
echo "Ready. Try:  bin/ytx --speakers <file-or-url>"
echo
