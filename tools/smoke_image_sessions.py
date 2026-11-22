#!/usr/bin/env python3
"""Check that four concurrent conversations retain their own screenshot and forward prefix."""
import argparse
import base64
import concurrent.futures
import json
import re
import secrets
import struct
import urllib.request
import zlib
from pathlib import Path


COLORS = [(232, 17, 16), (19, 218, 14), (14, 22, 230), (240, 219, 12)]
NAMES = ["red", "green", "blue", "yellow"]
QUESTION = "Which colors occupy the top left and bottom right quadrants of the last screenshot? Reply with just the two color words in that order."


def screenshot(lane, index):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    size = 320
    pixels = bytearray()
    for y in range(size):
        pixels.append(0)
        for half in (0, 1):
            pixels.extend(bytes(COLORS[(lane + half + 2 * (y >= size // 2)) % 4]) * (size // 2))
    # Give earlier screenshots their own pixel identity while preserving the quadrant task.
    pixels[1:4] = (index % (1 << 24)).to_bytes(3, "big")
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--screenshots", type=int, default=1, help="distinct synthetic screenshots per conversation")
    parser.add_argument("--padding-lines", type=int, default=0, help="number of inert text lines after the screenshots")
    parser.add_argument("--rounds", type=int, default=3, help="number of simultaneous turns per conversation")
    args = parser.parse_args()
    if args.screenshots < 1 or args.padding_lines < 0 or args.rounds < 2:
        parser.error("use positive screenshot counts, nonnegative padding, and at least two rounds")
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": "Bearer " + args.key_file.read_text().strip(), "Content-Type": "application/json"}
    label = secrets.token_hex(8)
    padding = "\n".join(f"Inventory record {i}: packed and received." for i in range(args.padding_lines))
    histories = [[{"role": "system", "content": "Visual inventory check " + label},
                  {"role": "user", "content": [screenshot(i, j) for j in range(args.screenshots)]}]
                 for i in range(4)]
    if padding:
        for history in histories:
            history.append({"role": "user", "content": padding})
    previous = [0] * 4

    def ask(lane):
        history = histories[lane]
        history.append({"role": "user", "content": QUESTION})
        body = dict(model="DeepSeek-V4.1-Flash-1017", temperature=1, top_p=1, top_k=-1,
                    reasoning_effort="high", max_tokens=384, stream=False, messages=history)
        request = urllib.request.Request(endpoint, json.dumps(body, separators=(",", ":")).encode(), headers)
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
        choice = result["choices"][0]
        answer = choice["message"]["content"]
        words = re.findall(r"red|green|blue|yellow", answer.lower())
        expected = [NAMES[lane], NAMES[(lane + 3) % 4]]
        if choice["finish_reason"] != "stop" or words != expected:
            raise ValueError(f"lane {lane} returned colors {words}, expected {expected}")
        cached = result["timings"]["cache_n"]
        tokens = result["usage"]["prompt_tokens"]
        if previous[lane] >= 8192 and cached < previous[lane] - 256:
            raise ValueError(f"lane {lane} reused {cached} tokens after a {previous[lane]}-token prompt")
        previous[lane] = tokens
        history.append({"role": "assistant", "content": answer})
        return dict(lane=lane, prompt_tokens=tokens, cached_tokens=cached, correct_colors=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for turn in range(args.rounds):
            print(json.dumps({"turn": turn + 1, "results": list(pool.map(ask, range(4)))}), flush=True)


if __name__ == "__main__":
    main()
