#!/usr/bin/env python3
"""Generate test dialog MP3 files from markdown scenarios using edge-tts + ffmpeg."""

import asyncio
import edge_tts
import subprocess
import tempfile
from pathlib import Path

PAUSE_MS = 700
SAMPLE_RATE = 8000

VOICES = {
    "operator_female": "ru-RU-SvetlanaNeural",
    "client_female": "ru-RU-SvetlanaNeural",
    "client_male": "ru-RU-DmitryNeural",
}

DIALOGS = [
    {
        "name": "dialog-transfers",
        "md": "docs/sample-dialog-transfers.md",
        "operator_voice": "operator_female",
        "client_voice": "client_female",
    },
    {
        "name": "dialog-incompetent",
        "md": "docs/sample-dialog-incompetent.md",
        "operator_voice": "operator_female",
        "client_voice": "client_male",
    },
    {
        "name": "dialog-complaints",
        "md": "docs/sample-dialog-complaints.md",
        "operator_voice": "operator_female",
        "client_voice": "client_male",
    },
    {
        "name": "dialog-cards",
        "md": "docs/sample-dialog-cards.md",
        "operator_voice": "operator_female",
        "client_voice": "client_female",
    },
]


def parse_dialog(md_path: str) -> list[tuple[str, str]]:
    """Extract (role, text) pairs from a markdown dialog file."""
    lines = Path(md_path).read_text(encoding="utf-8").splitlines()
    dialog = []
    for line in lines:
        line = line.strip()
        if line.startswith("**Оператор:**"):
            text = line.replace("**Оператор:**", "").strip()
            dialog.append(("operator", text))
        elif line.startswith("**Клиент:**"):
            text = line.replace("**Клиент:**", "").strip()
            dialog.append(("client", text))
    return dialog


async def tts_segment(text: str, voice: str, out_path: str):
    """Generate a single TTS segment."""
    comm = edge_tts.Communicate(text, voice)
    await comm.save(out_path)


def generate_silence(duration_ms: int, out_path: str):
    """Generate a silent MP3 segment."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"anullsrc=r=24000:cl=mono",
            "-t", f"{duration_ms / 1000:.3f}",
            "-c:a", "libmp3lame", "-q:a", "9",
            out_path,
        ],
        capture_output=True,
        check=True,
    )


def concat_audio(file_list: str, out_path: str):
    """Concatenate audio files using ffmpeg concat demuxer."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", file_list,
            "-c:a", "libmp3lame", "-q:a", "2",
            out_path,
        ],
        capture_output=True,
        check=True,
    )


async def generate_dialog(dialog_cfg: dict, base_dir: str):
    """Generate one dialog: TTS each line, combine, export."""
    name = dialog_cfg["name"]
    md_path = f"{base_dir}/{dialog_cfg['md']}"
    op_voice = VOICES[dialog_cfg["operator_voice"]]
    cl_voice = VOICES[dialog_cfg["client_voice"]]

    dialog = parse_dialog(md_path)
    print(f"\n=== {name} ({len(dialog)} реплик) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        segments = []
        for i, (role, text) in enumerate(dialog):
            voice = op_voice if role == "operator" else cl_voice
            seg_path = f"{tmpdir}/seg_{i:02d}.mp3"
            print(f"  [{role}] {text[:60]}...")
            await tts_segment(text, voice, seg_path)
            segments.append(seg_path)

        # Build concat list with silence between segments
        silence_path = f"{tmpdir}/silence.mp3"
        generate_silence(PAUSE_MS, silence_path)

        concat_lines = []
        for i, seg in enumerate(segments):
            concat_lines.append(f"file '{seg}'")
            if i < len(segments) - 1:
                concat_lines.append(f"file '{silence_path}'")

        concat_file = f"{tmpdir}/concat.txt"
        Path(concat_file).write_text("\n".join(concat_lines))

        # Export full quality
        out_path = f"{base_dir}/test_data/{name}.mp3"
        concat_audio(concat_file, out_path)

        # Get duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", out_path],
            capture_output=True, text=True,
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0
        print(f"  -> {out_path} ({duration:.1f} сек)")

        # Export phone quality (8kHz mono)
        tel_path = f"{base_dir}/test_data/{name}-tel.mp3"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", out_path,
                "-ar", str(SAMPLE_RATE), "-ac", "1",
                "-c:a", "libmp3lame", "-q:a", "2",
                tel_path,
            ],
            capture_output=True, check=True,
        )
        print(f"  -> {tel_path} (8kHz mono)")


async def main():
    base_dir = str(Path(__file__).resolve().parent.parent)
    for cfg in DIALOGS:
        await generate_dialog(cfg, base_dir)
    print("\nВсе диалоги сгенерированы.")


if __name__ == "__main__":
    asyncio.run(main())
