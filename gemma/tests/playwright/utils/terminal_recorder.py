"""Utilities for recording terminal output and generating snapshots.

This module provides tools for capturing terminal sessions, generating
visual snapshots, and creating comparison images for UI testing.
"""

import subprocess
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import shutil
from datetime import datetime, timedelta
from io import StringIO

from rich.console import Console
from rich.terminal_theme import MONOKAI, DIMMED_MONOKAI, SVG_EXPORT_THEME
import pyte
from PIL import Image, ImageDraw, ImageFont


class TerminalRecorder:
    """Record terminal sessions for testing and documentation."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recordings_dir = output_dir / "recordings"
        self.recordings_dir.mkdir(exist_ok=True)
        self.snapshots_dir = output_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)

    async def record_command(
        self,
        command: List[str],
        duration: int = 10,
        output_name: str = "recording",
        env: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Record terminal command execution using asciinema.

        Args:
            command: Command and arguments to execute
            duration: Maximum duration in seconds
            output_name: Name for output file (without extension)
            env: Environment variables to set

        Returns:
            Path to recording file (.cast format)
        """
        output_path = self.recordings_dir / f"{output_name}.cast"

        # Build asciinema command
        asciinema_cmd = [
            "asciinema",
            "rec",
            str(output_path),
            "--command",
            " ".join(command),
            "--overwrite",
        ]

        if env:
            asciinema_cmd.extend(["--env", ",".join(env.keys())])

        try:
            process = await asyncio.create_subprocess_exec(
                *asciinema_cmd,
                env={**subprocess.os.environ, **(env or {})},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for completion or timeout
            try:
                await asyncio.wait_for(process.wait(), timeout=duration)
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()

            return output_path

        except FileNotFoundError:
            # Fallback to manual recording if asciinema not available
            return await self._manual_record(command, duration, output_name, env)

    async def _manual_record(
        self,
        command: List[str],
        duration: int,
        output_name: str,
        env: Optional[Dict[str, str]],
    ) -> Path:
        """Fallback manual recording without asciinema."""
        output_path = self.recordings_dir / f"{output_name}.log"

        process = await asyncio.create_subprocess_exec(
            *command,
            env={**subprocess.os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        frames = []
        start_time = datetime.now()

        try:
            while True:
                if (datetime.now() - start_time).total_seconds() > duration:
                    break

                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(), timeout=0.1
                    )
                    if not line:
                        break

                    timestamp = (datetime.now() - start_time).total_seconds()
                    frames.append({
                        "time": timestamp,
                        "type": "o",
                        "data": line.decode("utf-8", errors="replace"),
                    })

                except asyncio.TimeoutError:
                    continue

        finally:
            process.terminate()
            await process.wait()

        # Save in asciicast v2 format
        output_data = {
            "version": 2,
            "width": 120,
            "height": 40,
            "timestamp": int(start_time.timestamp()),
            "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color"},
        }

        with open(output_path, "w") as f:
            f.write(json.dumps(output_data) + "\n")
            for frame in frames:
                f.write(json.dumps([frame["time"], frame["type"], frame["data"]]) + "\n")

        return output_path

    async def take_snapshot(
        self,
        console_output: str,
        name: str,
        format: str = "svg",
        theme: str = "monokai",
    ) -> Path:
        """Capture terminal snapshot as image.

        Args:
            console_output: Rich console output or ANSI text
            name: Snapshot name
            format: Output format (svg, png, html)
            theme: Color theme (monokai, dimmed, light)

        Returns:
            Path to snapshot file
        """
        themes = {
            "monokai": MONOKAI,
            "dimmed": DIMMED_MONOKAI,
            "svg": SVG_EXPORT_THEME,
        }

        output_path = self.snapshots_dir / f"{name}.{format}"

        if format == "svg":
            # Use Rich's SVG export
            console = Console(
                record=True,
                width=120,
                force_terminal=True,
            )
            console.print(console_output)

            svg_content = console.export_svg(
                theme=themes.get(theme, MONOKAI),
                title=name,
            )
            output_path.write_text(svg_content)

        elif format == "html":
            # Export as HTML
            console = Console(record=True, width=120, force_terminal=True)
            console.print(console_output)

            html_content = console.export_html(
                theme=themes.get(theme, MONOKAI),
                inline_styles=True,
            )
            output_path.write_text(html_content)

        elif format == "png":
            # Convert SVG to PNG
            svg_path = await self.take_snapshot(console_output, name, "svg", theme)
            output_path = await self._svg_to_png(svg_path, output_path)

        return output_path

    async def _svg_to_png(self, svg_path: Path, output_path: Path) -> Path:
        """Convert SVG to PNG using cairosvg or inkscape."""
        try:
            import cairosvg
            cairosvg.svg2png(url=str(svg_path), write_to=str(output_path))
            return output_path
        except ImportError:
            pass

        # Fallback to inkscape if available
        try:
            process = await asyncio.create_subprocess_exec(
                "inkscape",
                str(svg_path),
                "--export-type=png",
                f"--export-filename={output_path}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
            return output_path
        except FileNotFoundError:
            # Return SVG path if no PNG conversion available
            return svg_path

    async def generate_comparison(
        self,
        snapshot1: Path,
        snapshot2: Path,
        output_name: str,
        diff_only: bool = False,
    ) -> Path:
        """Generate side-by-side comparison of two snapshots.

        Args:
            snapshot1: First snapshot path
            snapshot2: Second snapshot path
            output_name: Output filename
            diff_only: Only highlight differences

        Returns:
            Path to comparison image
        """
        output_path = self.snapshots_dir / f"{output_name}_comparison.png"

        # Load images
        img1 = Image.open(snapshot1).convert("RGBA")
        img2 = Image.open(snapshot2).convert("RGBA")

        # Ensure same size
        max_width = max(img1.width, img2.width)
        max_height = max(img1.height, img2.height)

        img1_resized = Image.new("RGBA", (max_width, max_height), (255, 255, 255, 0))
        img2_resized = Image.new("RGBA", (max_width, max_height), (255, 255, 255, 0))

        img1_resized.paste(img1, (0, 0))
        img2_resized.paste(img2, (0, 0))

        if diff_only:
            # Generate difference image
            diff = Image.new("RGBA", (max_width, max_height), (0, 0, 0, 255))
            pixels1 = img1_resized.load()
            pixels2 = img2_resized.load()
            diff_pixels = diff.load()

            for y in range(max_height):
                for x in range(max_width):
                    if pixels1[x, y] != pixels2[x, y]:
                        diff_pixels[x, y] = (255, 0, 0, 255)  # Red for differences
                    else:
                        diff_pixels[x, y] = (0, 0, 0, 0)  # Transparent

            diff.save(output_path)
        else:
            # Side-by-side comparison
            comparison = Image.new(
                "RGBA",
                (max_width * 2 + 20, max_height + 60),
                (30, 30, 30, 255),
            )

            # Add labels
            draw = ImageDraw.Draw(comparison)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            draw.text((max_width // 2 - 50, 10), "Before", fill=(255, 255, 255, 255), font=font)
            draw.text((max_width + max_width // 2 - 50, 10), "After", fill=(255, 255, 255, 255), font=font)

            # Paste images
            comparison.paste(img1_resized, (0, 40))
            comparison.paste(img2_resized, (max_width + 20, 40))

            comparison.save(output_path)

        return output_path

    def cleanup_old_recordings(self, days: int = 7):
        """Delete recordings older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)

        for recording_dir in [self.recordings_dir, self.snapshots_dir]:
            for file_path in recording_dir.glob("*"):
                if file_path.is_file():
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime < cutoff:
                        file_path.unlink()

    async def capture_live_process(
        self,
        process: asyncio.subprocess.Process,
        name: str,
        duration: float = 10.0,
    ) -> List[Path]:
        """Capture snapshots from a live process at intervals.

        Args:
            process: Running subprocess
            name: Base name for snapshots
            duration: Total capture duration

        Returns:
            List of snapshot paths
        """
        snapshots = []
        start_time = asyncio.get_event_loop().time()
        frame_interval = 0.5  # Capture every 500ms
        frame_count = 0

        screen = pyte.Screen(120, 40)
        stream = pyte.Stream(screen)

        while (asyncio.get_event_loop().time() - start_time) < duration:
            try:
                # Read output
                line = await asyncio.wait_for(
                    process.stdout.readline(), timeout=frame_interval
                )
                if not line:
                    break

                # Update terminal emulator
                stream.feed(line.decode("utf-8", errors="replace"))

                # Capture snapshot
                terminal_text = "\n".join(screen.display)
                snapshot_path = await self.take_snapshot(
                    terminal_text,
                    f"{name}_frame_{frame_count:04d}",
                    format="png",
                )
                snapshots.append(snapshot_path)
                frame_count += 1

            except asyncio.TimeoutError:
                continue

        return snapshots
