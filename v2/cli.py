"""Command-line entry point for v2.

Provides subcommands to:
- run a one-off prediction, or
- start the Flask web server.

This stage focuses on UX, logging, and robustness; it loads artifacts from v0/v1.
"""

from __future__ import annotations

import argparse
import os
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config
from v2.logging_config import setup_logging
from v2.model_loader import load_v0_model, load_v1_model
from v2.predict import predict


def main():
    """CLI entry point.

    Parses command-line arguments, configures logging, loads the requested model
    artifact, and dispatches to prediction or server mode.

    Notes:
        If launched with no arguments, defaults to `serve` (useful for IDE run buttons).
        CLI arguments override values from `config.json` when both are provided.
    """
    ap = argparse.ArgumentParser(prog="v2")

    # Optional config path (default: <project_root>/config.json)
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file (default: <project_root>/config.json if present).",
    )

    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("predict", help="Predict sentiment for a text using v0 or v1 artifact")
    p1.add_argument("--model", choices=["v0", "v1"], default="v0")
    p1.add_argument("--text", required=True)

    p2 = sub.add_parser("serve", help="Run Flask server")
    # Defaults will be injected from config after initial parse
    p2.add_argument("--host", default=None)
    p2.add_argument("--port", type=int, default=None)

    # If launched without args (e.g. VS Code "Run Python File"), default to `serve`.
    argv = sys.argv[1:] or ["serve"]

    # 1) First parse to discover --config (and cmd)
    args_pre, _ = ap.parse_known_args(argv)

    # 2) Load config (optional)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = load_config(project_root, args_pre.config)

    cfg_log = (cfg.get("logging") or {}) if isinstance(cfg.get("logging"), dict) else {}
    cfg_srv = (cfg.get("server") or {}) if isinstance(cfg.get("server"), dict) else {}

    default_log_level = str(cfg_log.get("level", "INFO"))
    default_log_dir = str(cfg_log.get("log_dir", "logs"))

    default_host = str(cfg_srv.get("host", "127.0.0.1"))
    try:
        default_port = int(cfg_srv.get("port", 8000))
    except Exception:
        default_port = 8000

    # 3) Apply config-driven defaults (but do not override explicit CLI values)
    if args_pre.cmd == "serve":
        # If user didn't pass --host/--port, fill from config defaults
        if "--host" not in argv:
            p2.set_defaults(host=default_host)
        if "--port" not in argv:
            p2.set_defaults(port=default_port)

    # Re-parse with updated defaults
    args = ap.parse_args(argv)

    # 4) Logging from config (CLI does not currently expose log settings; config controls defaults)
    # If you later want CLI override: add --log-level/--log-dir and prioritize them over config here.
    setup_logging(log_dir=default_log_dir, level=default_log_level)
    log = logging.getLogger("v2.cli")

    if args.cmd == "predict":
        v0_ctx = None
        v1_ctx = None
        if args.model == "v0":
            v0_ctx = load_v0_model(project_root)
        else:
            v1_ctx = load_v1_model(project_root)

        out = predict(args.model, v0_ctx, v1_ctx, args.text)
        log.info(f"predict model={args.model} label={out['label']} prob_pos={out['prob_pos']}")
        print(out)
        return

    if args.cmd == "serve":
        from v2.server import create_app

        app = create_app()
        log.info(f"Starting server on http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)
        return


if __name__ == "__main__":
    main()
