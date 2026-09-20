import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, MagicMock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


ytx = load_module("ytx", ROOT / "bin" / "ytx")
speakers = load_module("speakers", ROOT / "lib" / "speakers.py")


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ytx tests ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        repo_patch = patch.object(ytx, "REPO", self.root)
        repo_patch.start()
        self.addCleanup(repo_patch.stop)
        env_patch = patch.dict(os.environ, {}, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def file(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_windows_speaker_environment_and_explicit_override(self):
        native = self.file(".venv/Scripts/python.exe")
        custom = self.file("custom/python.exe")
        with patch.object(ytx, "IS_WINDOWS", True):
            self.assertEqual(ytx.sherpa_python(), native)
            with patch.dict(os.environ, {"YTX_PYTHON": str(custom)}):
                self.assertEqual(ytx.sherpa_python(), custom)

    def test_unix_speaker_environment_is_preserved(self):
        native = self.file(".venv/bin/python")
        with patch.object(ytx, "IS_WINDOWS", False):
            self.assertEqual(ytx.sherpa_python(), native)

    def test_windows_whisper_cpu_cuda_and_override(self):
        cpu = self.file(".tools/whisper-cpu/Release/whisper-cli.exe")
        with patch.object(ytx, "IS_WINDOWS", True):
            self.assertEqual(ytx.whisper_cli(), cpu)
            cuda = self.file(".tools/whisper-cuda12/Release/whisper-cli.exe")
            self.assertEqual(ytx.whisper_cli(), cuda)
            with patch.dict(os.environ, {"YTX_WHISPER_CLI": str(cpu)}):
                self.assertEqual(ytx.whisper_cli(), cpu)

    def test_unix_whisper_location_is_preserved(self):
        native = self.file("dev/whisperccp/build/bin/whisper-cli")
        with patch.object(ytx, "IS_WINDOWS", False), patch.object(ytx, "HOME", self.root):
            self.assertEqual(ytx.whisper_cli(), native)

    @unittest.skipUnless(sys.platform == "win32", "Windows registry and environment expansion")
    def test_redirected_desktop_and_fallback(self):
        import winreg

        redirected = self.root / "OneDrive" / "Desktop"
        with patch.dict(os.environ, {"YTX_DESKTOP_TEST": str(redirected)}), \
                patch.object(winreg, "OpenKey", return_value=MagicMock()), \
                patch.object(winreg, "QueryValueEx", return_value=("%YTX_DESKTOP_TEST%", 2)):
            self.assertEqual(ytx.desktop_dir(), redirected)
        with patch.object(winreg, "OpenKey", side_effect=OSError), \
                patch.object(ytx, "HOME", self.root):
            self.assertEqual(ytx.desktop_dir(), self.root / "Desktop")

    def test_windows_reserved_titles_and_trailing_dots(self):
        with patch.object(ytx, "IS_WINDOWS", True):
            for title, expected in [("CON", "_CON"), ("aux.txt", "_aux.txt"),
                                    ("LPT1", "_LPT1"), ("Interview...", "Interview")]:
                self.assertEqual(ytx.safe_name(title), expected)
            self.assertTrue(ytx.safe_name("...").startswith("transcript-"))
        with patch.object(ytx, "IS_WINDOWS", False):
            self.assertEqual(ytx.safe_name("CON"), "CON")

    def test_help_reports_effective_output_directory(self):
        output = str(self.root / "custom output")
        with patch.dict(os.environ, {"YTX_OUT": output}):
            self.assertEqual(ytx.parse_args(["input.wav"]).output_dir, output)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit) as result:
                ytx.parse_args(["--help"])
            self.assertEqual(result.exception.code, 0)
            self.assertIn(output, " ".join(stream.getvalue().split()))

    def test_speaker_count_and_paths_with_spaces(self):
        args = ytx.parse_args(["--speakers", "2", "recording v1.mp4", "my outputs"])
        self.assertTrue(args.speakers)
        self.assertEqual(args.speaker_count, 2)
        self.assertEqual(args.source, "recording v1.mp4")
        self.assertEqual(args.output_dir, "my outputs")

    def test_windows_node_runtime_and_user_options(self):
        with patch.object(ytx, "IS_WINDOWS", True), \
                patch.object(ytx.shutil, "which", return_value="node.exe"), \
                patch.dict(os.environ, {"YTX_YTDLP_ARGS": '--cookies "D:/my files/cookies.txt"'}):
            args = ytx.yt_dlp_base("https://example.com/video")
            self.assertIn("node", args)
            self.assertEqual(args[-2:], ["--cookies", "D:/my files/cookies.txt"])

    def test_multilingual_speaker_files_are_utf8(self):
        source = self.root / "transcript.json"
        text = "你好 café العربية"
        source.write_text(json.dumps({"transcription": [
            {"offsets": {"from": 0, "to": 1000}, "text": text},
        ]}, ensure_ascii=False), encoding="utf-8")
        txt, srt = self.root / "out.txt", self.root / "out.srt"
        turn = types.SimpleNamespace(start=0, end=1, speaker=0)
        with patch.object(speakers, "read_wav16k", return_value=None), \
                patch.object(speakers, "diarize", return_value=[turn]):
            result = speakers.main([
                "--wav", "unused.wav", "--json", str(source),
                "--models", "unused", "--out-txt", str(txt), "--out-srt", str(srt),
            ])
        self.assertEqual(result, 0)
        self.assertIn(text, txt.read_text(encoding="utf-8"))
        self.assertIn(text, srt.read_text(encoding="utf-8"))

    def test_publish_preserves_unicode_names_dots_and_bytes(self):
        source = self.root / "transcript"
        destination = self.root / "中文 recording.v1"
        contents = {".txt": "你好\n".encode(), ".srt": b"1\r\n", ".json": b"{}"}
        for suffix, data in contents.items():
            Path(f"{source}{suffix}").write_bytes(data)
        ytx.copy_whisper_outputs(source, destination, speakers=True)
        for suffix, data in contents.items():
            self.assertEqual(Path(f"{destination}{suffix}").read_bytes(), data)

    def test_missing_fresh_output_does_not_overwrite_previous_transcript(self):
        source, destination = self.root / "transcript", self.root / "previous"
        Path(f"{source}.txt").write_text("new", encoding="utf-8")
        Path(f"{destination}.txt").write_text("keep", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ytx.copy_whisper_outputs(source, destination, speakers=False)
        self.assertEqual(Path(f"{destination}.txt").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
