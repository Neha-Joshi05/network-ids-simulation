"""
verify_setup.py
───────────────
Run this FIRST to confirm your environment is ready.

    python verify_setup.py
"""

import sys
import importlib
from pathlib import Path

REQUIRED_PACKAGES = [
    ("pandas",     "pandas"),
    ("numpy",      "numpy"),
    ("sklearn",    "scikit-learn"),
    ("xgboost",    "xgboost"),
    ("scipy",      "scipy"),
    ("matplotlib", "matplotlib"),
    ("seaborn",    "seaborn"),
    ("fastapi",    "fastapi"),
    ("uvicorn",    "uvicorn"),
    ("sqlalchemy", "sqlalchemy"),
    ("joblib",     "joblib"),
    ("tqdm",       "tqdm"),
    ("colorama",   "colorama"),
]

REQUIRED_FOLDERS = [
    "data", "simulator", "ids", "ml",
    "backend", "frontend", "models", "tests", "outputs", "images"
]

print("=" * 58)
print("  Network IDS Simulation — Setup Verifier")
print("=" * 58)

version = sys.version_info
status  = "✅" if version >= (3, 10) else "❌"
print(f"\n{status} Python {version.major}.{version.minor}.{version.micro}")

print("\n📦 Packages:")
missing = []
for import_name, pip_name in REQUIRED_PACKAGES:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"   ✅ {pip_name:<22} {ver}")
    except ImportError:
        print(f"   ❌ {pip_name:<22} NOT INSTALLED")
        missing.append(pip_name)

print("\n📁 Folders:")
for folder in REQUIRED_FOLDERS:
    exists = Path(folder).is_dir()
    print(f"   {'✅' if exists else '❌'} {folder}/")

print("\n📄 Data files:")
for f in ["data/network_traffic.csv", "data/network_traffic.parquet"]:
    p = Path(f)
    if p.exists():
        print(f"   ✅ {f}  ({p.stat().st_size//1024:,} KB)")
    else:
        print(f"   ⚠️  {f}  — run: python simulator/generate_dataset.py")

print("\n" + "=" * 58)
if missing:
    print(f"❌ Missing: pip install {' '.join(missing)}")
else:
    print("✅ All checks passed!")
    print("   Next: python simulator/generate_dataset.py")
print("=" * 58)