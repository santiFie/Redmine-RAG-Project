"""
tests/benchmarks/serve_dashboard.py
===================================
Servidor HTTP local para visualizar el Dashboard de Benchmarking de Decision Models.

Uso:
    python tests/benchmarks/serve_dashboard.py [--port 8766]
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import sys
import webbrowser
from pathlib import Path

_BENCHMARKS_DIR = Path(__file__).resolve().parent


def serve(port: int = 8766, open_browser: bool = True):
    os.chdir(_BENCHMARKS_DIR)

    handler = http.server.SimpleHTTPRequestHandler
    # Permitir reuso de dirección si el puerto quedó en TIME_WAIT
    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
            url = f"http://127.0.0.1:{port}/ui/index.html"
            print("=" * 70)
            print("📊 Servidor del Dashboard de Decision Models iniciado")
            print(f"👉 Abre la siguiente URL en tu navegador: {url}")
            print("Presiona Ctrl+C para detener el servidor.")
            print("=" * 70)

            if open_browser:
                try:
                    webbrowser.open(url)
                except Exception:
                    pass

            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    except Exception as e:
        print(f"Error al iniciar servidor en puerto {port}: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servidor HTTP para el Dashboard de Benchmarking")
    parser.add_argument("--port", type=int, default=8766, help="Puerto HTTP (por defecto: 8766)")
    parser.add_argument(
        "--no-browser", action="store_true", help="No intentar abrir el navegador automáticamente"
    )
    args = parser.parse_args()

    serve(port=args.port, open_browser=not args.no_browser)
