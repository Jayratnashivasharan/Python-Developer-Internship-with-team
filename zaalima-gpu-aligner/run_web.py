#!/usr/bin/env python3
"""Start the gpualign web application:   python run_web.py [--host H] [--port P]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpualign.web import serve  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="use 0.0.0.0 to allow other machines to connect")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    serve(a.host, a.port, a.data_dir)
