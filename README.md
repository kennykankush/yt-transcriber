# yt-transcriber

Transcribe a local media file or a URL, optionally labelling who spoke.

Two passes under the hood: whisper.cpp does the words, a separate diarization
pass finds who spoke when, and the two are merged by timestamp. The speaker pass
never touches the transcript text, so labels cost you nothing in accuracy.

## Install

```sh
./bootstrap.sh                  # tools + speaker models + python venv
./bootstrap.sh --with-whisper-model   # ...and the 1.6 GB whisper model
```

Requirements: `ffmpeg`, `yt-dlp`, and a built
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) (`whisper-cli`). Models are
never committed to this repo - `bootstrap.sh` fetches them into `models/`, which
is gitignored. Existing models in `~/dev/whisperccp/models` are reused as-is.

## Usage

```sh
bin/ytx <file|url>                    # raw transcript, .txt + .srt
bin/ytx --speakers <file|url>         # ...plus .speakers.txt and .speakers.srt
bin/ytx --speakers 2 <file|url>       # hint the speaker count (or --speaker-count 2)
bin/ytx --lang en <file|url>          # force a language instead of auto-detect
bin/ytx --vad <file|url>              # skip silence (recommended for long recordings)
bin/ytx --keep-audio <file|url>       # also keep the normalised 16 kHz WAV
bin/ytx --cookies chrome <url>        # uses your logged-in YouTube session
bin/ytx --help
```

Results land in `~/Desktop/transcriptions` unless you pass an output directory.

## Environment variables

| Variable | Effect |
| --- | --- |
| `YTX_MODELS` | extra directory to search for models, checked first |
| `YTX_WHISPER_CLI` | path to `whisper-cli` |
| `YTX_PYTHON` | python interpreter that has sherpa-onnx (default `.venv/bin/python`) |
| `YTX_OUT` | default output directory |
| `TRANSCRIBE_LANG` | default spoken language (same as `--lang`) |
| `TRANSCRIBE_VAD` | set to 1 to default `--vad` on |
| `TRANSCRIBE_PROMPT` | nudge spelling of names and jargon |
| `TRANSCRIBE_COOKIES` | browser to pull YouTube cookies from, e.g. `chrome` |

## How it works

1. A URL goes through `yt-dlp`; a local file is used directly.
2. `ffmpeg` normalises everything to 16 kHz mono 16-bit WAV, the only shape
   Whisper accepts.
3. `whisper-cli` transcribes with `ggml-large-v3-turbo` on the GPU.
4. With `--speakers`, sherpa-onnx runs pyannote segmentation plus a Titanet
   speaker embedding model, clusters the segments into voices, and each
   transcript segment is assigned the speaker it overlaps most. Consecutive
   segments from one voice are merged into a single block.

## Tuning

Clustering ships with a threshold of 0.85 on purpose. The library default of 0.5
over-splits badly: a two-person interview came out as nine speakers. At 0.85 the
same audio resolves to exactly two, matching what you hear. If a recording with
a known headcount comes out wrong, pass the count explicitly (`--speakers 3`)
and clustering is bypassed entirely.

## YouTube 403 errors

YouTube intermittently refuses the media request from yt-dlp's default client
with `HTTP Error 403: Forbidden`, even though metadata extraction succeeds. The
tool tries three things in order:

1. Your normal request (plus cookies, if configured).
2. The same request with the `android` player client.
3. Gives up and tells you to pass cookies.

Passing cookies is the better fix, not just a last resort. Measured on the same
video: without cookies the default client 403s, and the android fallback
downloads a 3.5 MB video+audio file; with `--cookies chrome` it succeeds on the
first try and pulls a 1.2 MB audio-only stream. Cookies also make bot checks far
less likely on longer videos.

```sh
bin/ytx --cookies chrome <url>          # or export TRANSCRIBE_COOKIES=chrome
```

macOS may ask for Keychain access the first time a browser's cookies are read.

## What has actually been measured

On a 3-minute slice of a two-person interview recording (OBS, dual-mono AAC):

| | whisper alone | + this tool's labels |
| --- | --- | --- |
| Text | `Hi Fahmy`, `Mohamad Fahmy`, `Sendian Berhad` | identical |
| Speakers | none | 2 detected, correct turns |
| Runtime | ~7 s | ~17 s total |

The alternative `--tinydiarize` route built into whisper.cpp was also tested and
rejected: it tracks turns well, but it forces the `small.en` model, which turned
`Mohamad Fahmy` into `Mama Fami`, and its markers are stripped from `.txt`/`.srt`
output anyway. It is English-only as well.

## Compatibility with the old shell function

This tool replaces a `transcribe` shell function, and was checked against it
rather than assumed equivalent. Same input through both, outputs byte-compared:

| Scenario | Result |
| --- | --- |
| Local file, explicit output dir | `.txt` and `.srt` byte-identical |
| Local file, no output dir | same destination, `~/Desktop/transcriptions` |
| Name with spaces and dots (`parity test.v1.mov`) | same output name, byte-identical |
| YouTube URL with cookies | byte-identical, same derived filename |
| Scheme-less URL (`youtu.be/...`) | works in both |
| `TRANSCRIBE_PROMPT` | byte-identical |
| Missing file | both exit 1 |
| No arguments | both exit 1 |

Deliberate differences, all additive or cosmetic:

- Usage errors exit 1 rather than argparse's default 2, matching the shell
  function, but the message is argparse's fuller one.
- Messages are prefixed `ytx:` rather than `transcribe:`.
- A title that sanitises to nothing now falls back to `transcript-<timestamp>`.
  The shell function checked for an empty title *before* sanitising, so an
  all-punctuation title could write to a bare `.txt`.

## Limits

- Diarization separates voices by acoustic similarity, not identity. Two people
  with similar voices, heavy crosstalk, or a noisy recording will produce
  imperfect boundaries. Labels are `Speaker 1`, `Speaker 2`, assigned by first
  appearance - they are not names.
- Diarization is language-agnostic, so non-English audio is fine. Whisper
  transcription quality depends on the model you point it at.
- Segments with no overlapping speech and no nearby turn are labelled `Unknown`.
