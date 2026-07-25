"""Replace the few non-ASCII characters in the viewer template with JS escapes.

The template is an r-string, so the escapes reach the JS source verbatim. Keeping the
generated page pure ASCII means it renders correctly regardless of the charset the host
serves it with -- the earlier mojibake came from exactly this. chr(92) is used for the
backslash so the intent survives any editor/shell round-trip.
"""
import sys

BS = chr(92)
path = sys.argv[1]
src = open(path, encoding="utf-8").read()
for codepoint in (0xB5, 0xB3, 0xB2):
    src = src.replace(chr(codepoint), BS + "u%04x" % codepoint)
open(path, "w", encoding="utf-8").write(src)
print("non-ASCII remaining:", sorted({hex(ord(c)) for c in src if ord(c) > 127}))
