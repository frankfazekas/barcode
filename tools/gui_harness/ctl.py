"""Send Python into a running harness.py and print what it returns.

    python tools/gui_harness/ctl.py "select_tab('Volumetric')"
    python tools/gui_harness/ctl.py -f snippet.py

A single expression is eval'd and its value returned. A multi-line snippet is exec'd;
assign to ``_result`` for anything you want back.
"""
import socket, sys, json

HOST, PORT = "127.0.0.1", 8765


def send(src, timeout=3600, host=HOST, port=PORT):
    s = socket.socket()
    s.settimeout(timeout)
    s.connect((host, port))
    s.sendall(src.encode("utf-8") + b"\x00")
    buf = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.decode("utf-8"))


if __name__ == "__main__":
    src = open(sys.argv[2], encoding="utf-8").read() if sys.argv[1] == "-f" else sys.argv[1]
    r = send(src)
    print(r["status"])
    print(r["result"])
