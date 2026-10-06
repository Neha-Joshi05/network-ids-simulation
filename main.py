"""
main.py
───────
Network IDS Simulation — Pipeline Runner

Usage:
    python main.py --phase generate
    python main.py --phase ingest
    python main.py --phase all
"""

import argparse
import subprocess
import sys


def run(script: str):
    print(f"\n{'─'*60}")
    print(f"▶  Running: {script}")
    print(f"{'─'*60}")
    subprocess.run([sys.executable, script], check=True)


def main():
    parser = argparse.ArgumentParser(description="Network IDS Simulation Pipeline")
    parser.add_argument("--phase", choices=["generate", "ingest", "all"],
                        default="generate")
    args = parser.parse_args()

    print("🛡️  Network Intrusion Detection System — Simulation")
    print("=" * 60)
    print("⚠️  Defensive educational project — synthetic data only")
    print("=" * 60)

    if args.phase in ("generate", "all"):
        run("simulator/generate_dataset.py")

    if args.phase in ("ingest", "all"):
        run("simulator/01_ingest.py")

    if args.phase == "all":
        print("\n✅ Phase 1 complete!")
        print("   Next: python ids/feature_extractor.py")


if __name__ == "__main__":
    main()