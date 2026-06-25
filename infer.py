#!/usr/bin/env python3
"""
infer.py — minimal client for Herb-VISOR.

Sends a specimen image and its taxon name to a running llama.cpp server
(OpenAI-compatible endpoint) and prints the model's structured JSON caption.

The model needs no system prompt and no schema instructions. The only text
input is the taxon binomial; the schema is baked into the weights. Use
temperature 0 for deterministic output.

Start the server first (see README quickstart):
    llama-server --model herb-visor-4b-q8.gguf \
                 --mmproj herb-visor-4b-mmproj-f16.gguf \
                 --temp 0 -c 8192 --host 127.0.0.1 --port 8080

Then:
    python infer.py path/to/specimen.jpg "Acer pseudoplatanus"
    python infer.py path/to/specimen.jpg "Acer pseudoplatanus" --url http://host:8080
"""
import argparse
import base64
import json
import sys
import urllib.request

MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}


def main():
    ap = argparse.ArgumentParser(description="Caption a herbarium specimen with Herb-VISOR.")
    ap.add_argument("image", help="path to the specimen image")
    ap.add_argument("taxon", help="taxon binomial, standard casing (e.g. 'Acer pseudoplatanus')")
    ap.add_argument("--url", default="http://localhost:8080",
                    help="server base URL (default: http://localhost:8080)")
    ap.add_argument("--raw", action="store_true",
                    help="print the full API response instead of just the caption JSON")
    args = ap.parse_args()

    ext = args.image.rsplit(".", 1)[-1].lower()
    mime = MIME.get(ext, "image/jpeg")
    try:
        img = base64.b64encode(open(args.image, "rb").read()).decode()
    except OSError as e:
        sys.exit(f"cannot read image: {e}")

    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": args.taxon},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{img}"}},
            ],
        }],
        "temperature": 0,
    }

    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.load(urllib.request.urlopen(req))
    except urllib.error.URLError as e:
        sys.exit(f"request failed (is the server running at {args.url}?): {e}")

    if args.raw:
        print(json.dumps(resp, indent=2))
        return

    content = resp["choices"][0]["message"]["content"]
    # the model returns a JSON string; pretty-print it, but fall back to raw
    # text if a response ever comes back non-JSON.
    try:
        print(json.dumps(json.loads(content), indent=2))
    except json.JSONDecodeError:
        print(content)


if __name__ == "__main__":
    main()
