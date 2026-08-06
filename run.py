"""
run.py — Quick launcher for EMA SAR Optimizer
Validates environment, then delegates to ema_sar_optimizer.py
"""
import sys
import os
import subprocess

REQUIRED = ["numpy", "pandas", "scipy", "matplotlib", "openpyxl"]
OPTIONAL  = ["statsmodels"]

def check_deps():
    missing, optional_missing = [], []
    for pkg in REQUIRED:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    for pkg in OPTIONAL:
        try: __import__(pkg)
        except ImportError: optional_missing.append(pkg)
    return missing, optional_missing

def install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + pkgs)

def main():
    print("=" * 60)
    print("  EMA SAR Optimizer — Quick Launcher")
    print("=" * 60)

    missing, opt_missing = check_deps()

    if missing:
        print(f"\n[!] Missing required packages: {', '.join(missing)}")
        ans = input("    Install now? [Y/n]: ").strip().lower()
        if ans in ("", "y"):
            install(missing)
            print("[+] Installed.")
        else:
            print("[-] Aborted. Install packages manually and retry.")
            sys.exit(1)

    if opt_missing:
        print(f"\n[i] Optional packages not found: {', '.join(opt_missing)}")
        print("    Installing for full FDR/DSR functionality...")
        try:
            install(opt_missing)
        except Exception:
            print("    [warn] Could not install optional packages — some features disabled.")

    print("\n[+] Environment OK\n")

    # Parse simple args
    args = sys.argv[1:]
    if not args:
        print("Usage:")
        print("  python run.py --mode coarse                 # Fast & optimal: coarse+fine grid (Default)")
        print("  python run.py --mode bo                     # Bayesian optimization search (Smart QMC+RBF)")
        print("  python run.py --fetch-live                  # Fetch fresh live market data via yfinance")
        print("  python run.py --mode full                   # Full exhaustive grid (Brute force)")
        print("  python run.py --risk-aversion 3.0           # Custom risk-aversion lambda")
        print()
        ans = input("Press Enter to run --mode coarse, or Ctrl-C to abort: ").strip()
        args = ["--mode", "coarse"]

    cmd = [sys.executable, "ema_sar_optimizer.py"] + args
    print(f"[>] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
