"""Minimal streaming demo:  python scripts/ws_client.py "Hello cluster" """
import asyncio
import json
import sys

import websockets


async def main(prompt: str, url: str = "ws://localhost:8000/ws/generate"):
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"prompt": prompt, "max_new_tokens": 48, "temperature": 0.7, "top_k": 40}))
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "token":
                print(m["text"], end="", flush=True)
            else:
                print("\n", {k: v for k, v in m.items() if k != "type"})
                break


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "Hello, distributed world!"))
