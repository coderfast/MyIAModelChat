# System Tray Icon for Cache Viewer

## Context

The Cache Viewer used a `QSplashScreen` overlay during loading that blocked the user from interacting with the PC. Replaced with a system tray icon that shows loading progress in the tooltip and balloon notifications.

## What it does

1. Tray icon appears **before** the window (immediately on app start)
2. Window shows and loads normally (no hidden/minimized state)
3. Hover over tray icon → tooltip shows current loading phase and percentage
4. Balloon notifications update in real-time (even while hovering)
5. Loading complete → green checkmark icon + "Ready (N samples)" balloon
6. Right-click tray → Quit
7. Closing the window removes the tray icon and exits the app
8. Graceful fallback if pystray/Pillow not installed

## Architecture

```
main()
├── SystemTrayIcon (created FIRST, before CacheViewer)
│   ├── pystray.Icon (background daemon thread)
│   ├── Tooltip: "Cache Viewer - Phase 2/5: Loading samples... (45%)"
│   ├── Balloon notification (real-time updates via notify())
│   └── Menu: Quit
├── CacheViewer (created after tray, loads data via QTimer.singleShot)
│   └── _load_cache_data() → tray_icon.update(pct, text) per phase
├── closeEvent → tray_icon.stop() (removes icon from tray)
└── QApplication event loop (no setQuitOnLastWindowClosed)
```

## Bugs fixed during implementation

### 1. Splash screen exception swallowing
- **Bug:** `_cb` callback in `_load_cache_data` had signature `(p, _base=...)` which caused `progress_callback(0.0, "text")` to overwrite `_base` (int) with a string. The `try/except` in the phase loop silently caught the TypeError.
- **Fix:** Changed `_cb` signature to `(p_or_tuple, _desc="", _base=...)` to properly absorb both positional arguments.

### 2. Scroll loading stuck at 500 items
- **Bug:** `ContinuousLoaderWorker.run()` loaded ALL blocks in a `while` loop without yielding to Qt's event loop. Signals `block_ready` were queued but never processed until the thread finished.
- **Fix:** Rewrote worker to load a **single block** per execution. Removed `while` loop, `_lock`, `start_loading()`, and `loading_finished` signal.

### 3. Scroll detection only triggered at scrollbar bottom
- **Bug:** `VirtualScrollTable` used `scrollbar.maximum() - value < 100` which only fired when the user scrolled to the absolute bottom of the scrollbar.
- **Fix:** Rewrote to detect when the user approaches the **loaded data boundary**: `last_visible >= loaded_row_count - (viewport_rows * PREFETCH_VIEWPORTS)`. Added `set_loaded_row_count()` method to track loaded data.

### 4. No re-check after block load
- **Bug:** After `_on_block_ready` loaded a block, the scrollbar position didn't change (still at maximum), so `_check_scroll` never fired again.
- **Fix:** `_on_block_ready` now calls `self.samples_table._on_scroll(current_pos)` to re-evaluate prefetch need after each block.

### 5. Tray icon not receiving updates during loading
- **Bug:** `_load_cache_data()` was called in `CacheViewer.__init__` (via `_set_cache_dir`), but `set_tray_icon()` was called AFTER the viewer was created. The tray icon was `None` during the entire loading.
- **Fix:** Changed `_set_cache_dir` to use `QTimer.singleShot(0, self._load_cache_data)` so loading starts after `main()` has set the tray icon.

### 6. Tooltip not updating while hovering
- **Bug:** Windows reads the tray tooltip only when the cursor enters the icon area. Changing `icon.title` while hovering has no visible effect.
- **Fix:** Added `self._icon.notify(message, title)` calls in `update()` and `set_ready()` to show balloon notifications that update in real-time.

### 7. Tray icon not removed on window close
- **Bug:** `app.setQuitOnLastWindowClosed(False)` prevented the app from quitting when the window was closed. The tray icon stayed in the system tray.
- **Fix:** Removed `setQuitOnLastWindowClosed(False)`. Added `tray_icon.stop()` to `closeEvent`.

### 8. Import error when running as script
- **Bug:** `from APP_CACHE_VIEWER.system_tray import SystemTrayIcon` failed when running `python cache_viewer.py` from inside the APP_CACHE_VIEWER directory.
- **Fix:** Added try/except fallback: `try: from APP_CACHE_VIEWER... except ModuleNotFoundError: from system_tray...`

## Files

### `APP_CACHE_VIEWER/system_tray.py` (NEW)

- `SystemTrayIcon` class with `run()`, `update(progress, text)`, `set_ready(count)`, `stop()`
- Icon: 64x64 RGBA with progress arc + percentage (idle: gray "CV", loading: green arc, ready: green checkmark)
- Tooltip + balloon notifications via `pystray.Icon.notify()`
- Menu: only Quit (no "Open" option)
- Graceful fallback if pystray/Pillow not installed

### `APP_CACHE_VIEWER/cache_viewer.py` (MODIFIED)

- Removed: `QSplashScreen`, `_update_splash()`, `_create_splash_pixmap()`, `self._splash`
- Removed imports: `QPixmap`, `QPainter`, `QColor`, `QSplashScreen`
- Added import: `QTimer`
- Added: `set_tray_icon(tray)` method
- `_load_cache_data()`: uses `tray_icon.update()` instead of splash, removed `processEvents()` calls
- `_cb` callback: fixed signature to handle `(float, str)` positional args
- `closeEvent`: stops worker AND tray icon
- `_set_cache_dir`: uses `QTimer.singleShot(0, _load_cache_data)` instead of direct call
- `VirtualScrollTable`: rewritten with `prefetch_needed` signal, viewport-based detection, `PREFETCH_VIEWPORTS=20`
- `ContinuousLoaderWorker`: loads single block per execution (no while loop)
- `main()`: creates tray icon first, passes `--no-tray` option, no `setQuitOnLastWindowClosed`

### `APP_CACHE_VIEWER/__main__.py` (NEW)

Entry point for `python -m APP_CACHE_VIEWER`.

### `APP_CACHE_VIEWER/requirements.txt` (MODIFIED)

Added `pystray>=0.19.0` and `Pillow>=9.0.0`.

## Usage

```bash
python -m APP_CACHE_VIEWER                    # With tray icon
python -m APP_CACHE_VIEWER --no-tray          # Without tray icon
python -m APP_CACHE_VIEWER /path/to/cache     # With explicit path
python cache_viewer.py                        # Direct execution (also works)
```

## Cross-Platform

| Platform | Backend | Notes |
|----------|---------|-------|
| Windows | win32 | Works out of the box |
| Linux | appindicator/gtk | Needs `libappindicator3-dev` or `ayatana-appindicator3-dev` |
| macOS | darwin | Works with pystray >= 0.19 |

If pystray is not installed, the app runs normally without tray icon.
