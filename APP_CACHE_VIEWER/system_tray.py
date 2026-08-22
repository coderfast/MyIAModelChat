"""System tray icon with dynamic progress display for Cache Viewer."""

import logging
from threading import Thread
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    import pystray
    from pystray import MenuItem as Item
    HAS_PYSTRAY = True
except ImportError:
    HAS_PYSTRAY = False
    logger.warning("pystray not installed - tray icon disabled")

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("Pillow not installed - tray icon disabled")


class SystemTrayIcon:
    """System tray icon with dynamic progress display.

    Shows a circular progress indicator and percentage in the system tray.
    Hover shows loading progress. Right-click shows Quit.
    """

    def __init__(
        self,
        on_quit: Callable,
    ):
        self._on_quit = on_quit
        self._icon: Optional["pystray.Icon"] = None
        self._progress = 0
        self._text = ""
        self._ready = False

    @property
    def available(self) -> bool:
        return HAS_PYSTRAY and HAS_PIL

    def run(self):
        """Start the tray icon in a background daemon thread."""
        if not self.available:
            return

        menu = pystray.Menu(
            Item("Quit", self._on_quit),
        )

        self._icon = pystray.Icon(
            name="cache_viewer",
            icon=self._create_icon(0, False),
            title="Cache Viewer - Loading...",
            menu=menu,
        )

        thread = Thread(target=self._icon.run, daemon=True)
        thread.start()
        logger.info("Tray icon started")

    def update(self, progress: int, text: str = ""):
        """Update icon visual and show balloon notification. Thread-safe."""
        if not self._icon:
            return

        self._progress = progress
        self._text = text
        self._ready = progress >= 100

        try:
            self._icon.icon = self._create_icon(progress, self._ready)
            # Tooltip (read once on hover)
            if text:
                self._icon.title = f"Cache Viewer - {text} ({progress}%)"
            else:
                self._icon.title = f"Cache Viewer - {progress}%"
            # Balloon notification (shows even while hovering)
            if text:
                self._icon.notify(f"{text} ({progress}%)", "Cache Viewer")
            else:
                self._icon.notify(f"{progress}%", "Cache Viewer")
        except Exception as e:
            logger.debug(f"Tray update failed: {e}")

    def set_ready(self, sample_count: int = 0):
        """Mark as ready with a green check icon."""
        self._ready = True
        self._progress = 100
        if self._icon:
            try:
                self._icon.icon = self._create_icon(100, True)
                ready_text = f"Ready ({sample_count:,} samples)" if sample_count else "Ready"
                self._icon.title = f"Cache Viewer - {ready_text}"
                self._icon.notify(ready_text, "Cache Viewer")
            except Exception as e:
                logger.debug(f"Tray set_ready failed: {e}")

    def stop(self):
        """Stop the tray icon."""
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass
            self._icon = None

    def _create_icon(self, progress: int, is_ready: bool) -> "Image.Image":
        """Draw a 64x64 icon: circle with progress arc + percentage text."""
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        cx, cy = size // 2, size // 2
        r = 28

        if is_ready:
            # Green filled circle with checkmark
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=(34, 139, 34),
                outline=(50, 180, 50),
                width=2,
            )
            # Simple checkmark
            draw.line(
                [(cx - 10, cy), (cx - 3, cy + 8), (cx + 12, cy - 8)],
                fill="white",
                width=3,
            )
        elif progress > 0:
            # Dark background circle
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=(40, 40, 40),
                outline=(80, 80, 80),
                width=1,
            )
            # Progress arc (from top, clockwise)
            extent = int(progress * 360 / 100)
            if extent > 0:
                draw.arc(
                    [cx - r, cy - r, cx + r, cy + r],
                    start=90,
                    end=90 + extent,
                    fill=(0, 180, 80),
                    width=4,
                )
            # Percentage text
            label = f"{progress}%"
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except (OSError, IOError):
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (cx - tw // 2, cy - th // 2 - 2),
                label,
                fill="white",
                font=font,
            )
        else:
            # Idle: gray circle with "CV"
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=(50, 50, 50),
                outline=(100, 100, 100),
                width=1,
            )
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except (OSError, IOError):
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), "CV", font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (cx - tw // 2, cy - th // 2),
                "CV",
                fill=(180, 180, 180),
                font=font,
            )

        return img
