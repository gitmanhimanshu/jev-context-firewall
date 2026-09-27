import argparse
import sys
import uvicorn

from jev_proxy.config import load_config
from jev_proxy.server import ProxyServer


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Jev Context Firewall Proxy (Python Edition)")
    parser.add_argument("--config", default="config.json", help="Path to config.json file")
    args = parser.parse_args()

    print("==================================================================")
    print("       [+] JEV CONTEXT FIREWALL PROXY (Python Native)             ")
    print("==================================================================")

    cfg = load_config(args.config)

    print(f"Proxy Port         : {cfg.port}")
    print(f"Jev Base URL       : {cfg.jev_base_url}")
    print(f"Jev Model          : {cfg.jev_model}")
    keys = cfg.get_api_keys()
    if len(keys) > 1:
        print(f"Jev API Key Pool   : {len(keys)} keys active (Concurrency: {cfg.get_concurrency()})")
    elif len(keys) == 1:
        print(f"Jev API Key Pool   : 1 key active (Concurrency: {cfg.get_concurrency()})")
    else:
        print("Jev API Key Pool   : none (fail-open mode)")

    print(f"Relevance Gate     : noul >= {cfg.relevance_threshold:.2f}")
    print(f"Web Dashboard      : http://127.0.0.1:{cfg.port}/dashboard")
    print("------------------------------------------------------------------")
    print("Status: Ready to intercept and prune AI IDE requests transparently.")
    print("Press Ctrl+C to stop.")
    print("==================================================================")

    server = ProxyServer(cfg)
    uvicorn.run(server.app, host="0.0.0.0", port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
