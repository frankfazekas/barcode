"""Screenshot a window WITHOUT focusing it, moving the cursor, or reading the screen.

Uses PrintWindow(PW_RENDERFULLCONTENT), which asks the window to redraw itself into an
off-screen device context. That means it works on a window that is behind other windows,
or parked entirely off the desktop -- unlike pyautogui.screenshot(), which scrapes the
visible screen and therefore needs the window raised and focused.

    python tools/gui_harness/bgshot.py out_name [out_dir]

Prints (path, printwindow_return, (w, h)). A return of 1 means the window rendered.
"""
import sys, os, win32gui, win32ui
from ctypes import windll
from PIL import Image

PW_RENDERFULLCONTENT = 0x00000002
DEFAULT_TITLE = "BARCODE: Biomaterial"


def find(title_prefix=DEFAULT_TITLE):
    hs = []
    win32gui.EnumWindows(
        lambda h, _: hs.append(h) if win32gui.GetWindowText(h).startswith(title_prefix) else None,
        None)
    if not hs:
        raise RuntimeError("no window titled " + title_prefix)
    return hs[0]


def grab(name, out_dir=".", hwnd=None):
    hwnd = hwnd or find()
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    w, h = r - l, b - t
    wdc = win32gui.GetWindowDC(hwnd)
    dc = win32ui.CreateDCFromHandle(wdc)
    mem = dc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(dc, w, h)
    mem.SelectObject(bmp)
    ok = windll.user32.PrintWindow(hwnd, mem.GetSafeHdc(), PW_RENDERFULLCONTENT)
    info, bits = bmp.GetInfo(), bmp.GetBitmapBits(True)
    img = Image.frombuffer("RGB", (info["bmWidth"], info["bmHeight"]), bits, "raw", "BGRX", 0, 1)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".png")
    img.save(path)
    win32gui.DeleteObject(bmp.GetHandle())
    mem.DeleteDC()
    dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, wdc)
    return path, ok, (w, h)


if __name__ == "__main__":
    print(grab(sys.argv[1] if len(sys.argv) > 1 else "shot",
               sys.argv[2] if len(sys.argv) > 2 else "."))
