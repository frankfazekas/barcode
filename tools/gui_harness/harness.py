"""Run the BARCODE GUI so an agent can drive it WITHOUT stealing focus.

Starts the real app exactly as ``main.py`` does -- same pages, same widgets -- plus a
background thread listening on 127.0.0.1:8765. Send it Python source (see ``ctl.py``);
it is exec'd on the Tk main thread via ``root.after`` and the result comes back as a
repr. Widgets are invoked directly, so there is no mouse, no keyboard, and no focus
steal: the window can sit off-screen for the whole session.

    python tools/gui_harness/harness.py            # then use ctl.py from another shell
    python tools/gui_harness/harness.py --onscreen # leave it where a human can watch

See README.md in this folder for the full recipe.
"""
import sys, os, socket, threading, traceback, queue, json, argparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import matplotlib
matplotlib.use("Agg")
import tkinter as tk

from gui.pages.home_page import create_home_page
from gui.pages.processing_page import create_process_page
from gui.pages.analysis_page import create_combine_page
from gui.window import setup_main_window, setup_scrollable_container

HOST, PORT = "127.0.0.1", 8765

parser = argparse.ArgumentParser()
parser.add_argument("--visible", action="store_true",
                    help="leave the window visible so a human can watch along")
parser.add_argument("--port", type=int, default=PORT)
args = parser.parse_args()
PORT = args.port

root = setup_main_window()
root.geometry("1050x1500+60+30")


def _hide():
    """Make the window invisible to the user WITHOUT moving it off-screen.

    Moving it off-screen looks tempting and does not work: Windows stops painting a
    fully off-desktop window, so PrintWindow keeps returning the last pixels drawn while
    it was visible. Every capture after a tab switch comes back stale, silently -- the
    call succeeds and the image looks plausible. An on-screen window keeps repainting
    even when it is completely covered by other windows, so instead we leave it on-screen
    at alpha 0: invisible, still painted, still capturable.

    WS_EX_TRANSPARENT makes it click-through so it cannot swallow mouse clicks aimed at
    whatever is underneath; WS_EX_NOACTIVATE keeps it from taking focus; WS_EX_TOOLWINDOW
    keeps it out of the taskbar and alt-tab.
    """
    import win32gui, win32con
    root.attributes("-alpha", 0.0)
    hwnd = int(root.frame(), 16) if isinstance(root.frame(), str) else root.winfo_id()
    hwnd = win32gui.GetParent(root.winfo_id()) or root.winfo_id()
    ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    win32gui.SetWindowLong(
        hwnd, win32con.GWL_EXSTYLE,
        ex | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
        | win32con.WS_EX_NOACTIVATE | win32con.WS_EX_TOOLWINDOW,
    )
    win32gui.SetWindowPos(hwnd, win32con.HWND_BOTTOM, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)


if not args.visible:
    # after the first paint, so the widget tree exists before it goes transparent
    root.after(400, _hide)

state = {"page": "home", "scrollable": None, "canvas": None}


def switch_page(page_name):
    """Same page switcher main.py wires up, plus a record of what is showing."""
    for widget in root.winfo_children():
        widget.destroy()
    scrollable_frame, canvas = setup_scrollable_container(root)
    state.update(page=page_name, scrollable=scrollable_frame, canvas=canvas)
    if page_name == "home":
        create_home_page(canvas, switch_page)
    elif page_name == "process":
        create_process_page(scrollable_frame, switch_page)
    elif page_name == "combine":
        create_combine_page(scrollable_frame, switch_page)


create_home_page(root, switch_page)


# --------------------------------------------------------------------------
# Helpers available inside anything sent through ctl.py
# --------------------------------------------------------------------------
def walk(w=None, depth=0):
    """Yield (depth, widget) over the whole widget tree."""
    w = w if w is not None else root
    yield depth, w
    for c in w.winfo_children():
        yield from walk(c, depth + 1)


def describe(w):
    d = {"cls": w.winfo_class(), "path": str(w)}
    for opt in ("text", "value", "variable", "textvariable"):
        try:
            v = w.cget(opt)
            if v not in ("", None):
                d[opt] = str(v)
        except Exception:
            pass
    try:
        d["mapped"] = bool(w.winfo_ismapped())
    except Exception:
        pass
    return d


def dump(filter_cls=None):
    """Flat description of every widget, optionally restricted to some classes."""
    return [{"depth": d, **describe(w)} for d, w in walk()
            if not filter_cls or w.winfo_class() in filter_cls]


def find_text(needle, cls=None):
    """Widgets whose -text contains needle. Note many labels are SEPARATE widgets
    from the control they label, so a checkbox usually has no text of its own."""
    hits = []
    for _, w in walk():
        if cls and w.winfo_class() not in cls:
            continue
        try:
            t = str(w.cget("text"))
        except Exception:
            continue
        if needle.lower() in t.lower():
            hits.append(w)
    return hits


def notebooks():
    return [w for _, w in walk() if w.winfo_class() == "TNotebook"]


def select_tab(index_or_text):
    """Switch tabs on the process page. Returns the tab titles."""
    nb = notebooks()[0]
    if isinstance(index_or_text, int):
        nb.select(index_or_text)
    else:
        for i, tid in enumerate(nb.tabs()):
            if index_or_text.lower() in nb.tab(tid, "text").lower():
                nb.select(i)
                break
    root.update_idletasks()
    return [nb.tab(t, "text") for t in nb.tabs()]


def guis():
    """The live GUI config wrappers, which are otherwise local to create_process_page.

    Returns a dict with keys 'config', 'input', 'preview' (whichever exist). These are
    the objects the Process Data button reads, so setting a var here is exactly
    equivalent to a user typing in the box.
    """
    import gc
    from gui.config import BarcodeConfigGUI, InputConfigGUI, PreviewConfigGUI
    out = {}
    for key, cls in (("config", BarcodeConfigGUI), ("input", InputConfigGUI),
                     ("preview", PreviewConfigGUI)):
        found = [o for o in gc.get_objects() if isinstance(o, cls)]
        if found:
            out[key] = found[-1]
    return out


def scroll_to(fraction):
    """Scroll the process page's canvas: 0.0 = top, 1.0 = bottom."""
    for _, w in walk():
        if w.winfo_class() == "Canvas":
            try:
                w.yview_moveto(fraction)
            except Exception:
                pass
    root.update_idletasks()
    return fraction


# --------------------------------------------------------------------------
# Control socket
# --------------------------------------------------------------------------
_req = queue.Queue()


def _pump():
    """Runs on the Tk thread: execute whatever ctl.py queued. Tk is not thread-safe,
    so nothing may touch a widget from the socket thread."""
    try:
        while True:
            src, resp = _req.get_nowait()
            g = globals()
            try:
                try:
                    val = eval(src, g)
                except SyntaxError:
                    exec(src, g)
                    val = g.get("_result")
                resp.put(("ok", repr(val)))
            except Exception:
                resp.put(("err", traceback.format_exc()))
    except queue.Empty:
        pass
    root.after(50, _pump)


def _serve():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(5)
    while True:
        c, _ = srv.accept()
        data = b""
        while not data.endswith(b"\x00"):
            chunk = c.recv(65536)
            if not chunk:
                break
            data += chunk
        resp = queue.Queue()
        _req.put((data.rstrip(b"\x00").decode("utf-8"), resp))
        try:
            status, payload = resp.get(timeout=3600)
        except queue.Empty:
            status, payload = "err", "timeout waiting for the Tk thread"
        c.sendall(json.dumps({"status": status, "result": payload}).encode("utf-8"))
        c.close()


threading.Thread(target=_serve, daemon=True).start()
root.after(50, _pump)
print(f"harness listening on {HOST}:{PORT} "
      f"({'visible' if args.visible else 'invisible (alpha 0, click-through)'})", flush=True)
root.mainloop()
