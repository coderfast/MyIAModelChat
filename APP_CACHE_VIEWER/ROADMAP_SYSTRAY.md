# Plan: System Tray Icon for Cache Viewer

## Context

The Cache Viewer used a `QSplashScreen` overlay during loading that blocks the user from interacting with the PC. Replaced with a system tray icon that shows loading progress in the tooltip when hovering.

## What it does

1. Tray icon appears **before** the window (immediately on app start)
2. Window shows and loads normally (no hidden/minimized state)
3. Hover over tray icon → tooltip shows current loading phase and percentage
4. Loading complete → tooltip shows "Ready (N samples)"
5. Right-click tray → Quit
6. Graceful fallback if pystray/Pillow not installed

## Architecture

```
main()
├── SystemTrayIcon (created FIRST, before CacheViewer)
│   ├── pystray.Icon (background thread)
│   ├── Tooltip: "Cache Viewer - Loading samples... 45%"
│   └── Menu: Quit
├── CacheViewer (created after tray, loads data normally)
│   └── _load_cache_data() → tray_icon.update(pct, text) per phase
└── QApplication event loop
```

## Files

### `APP_CACHE_VIEWER/system_tray.py` (NEW)

- `SystemTrayIcon` class with `run()`, `update(progress, text)`, `set_ready(count)`, `stop()`
- Icon: 64x64 RGBA with progress arc + percentage (idle: gray "CV", loading: green arc, ready: green checkmark)
- Tooltip updates with phase description and percentage
- Menu: only Quit

### `APP_CACHE_VIEWER/cache_viewer.py` (MODIFIED)

- Removed: `QSplashScreen`, `_update_splash()`, `_create_splash_pixmap()`
- Added: `set_tray_icon(tray)` method
- `_load_cache_data()` calls `tray_icon.update()` instead of splash
- `main()`: creates tray icon first, then viewer

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
