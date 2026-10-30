"""Download and unpack the real datasets from Kaggle (works on Windows, Linux and Colab).

Needs Kaggle API credentials: ~/.kaggle/kaggle.json, or KAGGLE_USERNAME / KAGGLE_KEY
environment variables. The two Optiver datasets are competition data: accept each
competition's rules on kaggle.com once with the same account, or the download is refused.

    python scripts/fetch_datasets.py                       # everything
    python scripts/fetch_datasets.py btc fi2010            # just these
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR  # noqa: E402

RAW = DATA_DIR / "raw"
FI2010_FILES = [
    "NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_7.txt",
    "NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt",
    "NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt",
    "NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt",
]

DATASETS = {
    # name: (kaggle args, folder, file that marks it as done, approx download size)
    "btc": (["datasets", "download", "martinsn/high-frequency-crypto-limit-order-book-data", "-f", "BTC_1sec.csv"],
            "crypto", "BTC_1sec.csv", "330 MB"),
    "eth": (["datasets", "download", "martinsn/high-frequency-crypto-limit-order-book-data", "-f", "ETH_1sec.csv"],
            "crypto", "ETH_1sec.csv", "340 MB"),
    "fi2010": (["datasets", "download", "ulfricirons/fi-2010"],
               "fi2010", "BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/" + FI2010_FILES[0], "2.1 GB"),
    "optiver_close": (["competitions", "download", "-c", "optiver-trading-at-the-close", "-f", "train.csv"],
                      "optiver_close", "train.csv", "200 MB"),
    "optiver_rvp": (["competitions", "download", "-c", "optiver-realized-volatility-prediction"],
                    "optiver_rvp", "train.csv", "2.6 GB"),
}
RULES = {
    "optiver_close": "https://www.kaggle.com/competitions/optiver-trading-at-the-close/rules",
    "optiver_rvp": "https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/rules",
}


def kaggle_cli() -> list:
    exe = shutil.which("kaggle")
    return [exe] if exe else [sys.executable, "-c", "from kaggle.cli import main; main()"]


FI2010_PREFIX = "BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/"


def extract_fi2010(z: zipfile.ZipFile, dest: Path, depth: int = 0):
    """Extracts only the 4 Setup 2 files (~0.9 GB of 8 GB) to their expected paths.

    Files are matched by name wherever they sit in the archive, so a different top-level
    folder in the Kaggle mirror still works; zips nested inside the archive are searched too.
    """
    wanted = {Path(m).name: m for m in FI2010_FILES}
    found = {}
    for info in z.infolist():
        base = info.filename.rsplit("/", 1)[-1]
        if base in wanted and base not in found:
            found[base] = info
    for base, info in found.items():
        target = dest / FI2010_PREFIX / wanted[base]
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out, length=16 << 20)
    missing = [b for b in wanted if b not in found]
    if missing and depth == 0:
        for inner in [i for i in z.infolist() if i.filename.lower().endswith(".zip")]:
            tmp = dest / Path(inner.filename).name
            with z.open(inner) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, length=16 << 20)
            with zipfile.ZipFile(tmp) as nested:
                missing = extract_fi2010(nested, dest, depth + 1)
            tmp.unlink()
            if not missing:
                break
    if missing and depth == 0:
        sample = "\n      ".join(i.filename for i in z.infolist()[:15])
        raise FileNotFoundError(f"FI-2010 archive does not contain {missing}. First entries:\n      {sample}")
    return missing


def extract(zip_path: Path, dest: Path, name: str):
    with zipfile.ZipFile(zip_path) as z:
        if name == "fi2010":
            extract_fi2010(z, dest)
        else:
            z.extractall(dest)


def fetch(name: str, keep_zips: bool) -> bool:
    args, folder, marker, size = DATASETS[name]
    dest = RAW / folder
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / marker).exists():
        print(f"[=] {name}: already present")
        return True

    print(f"[+] {name}: downloading (~{size}) into {dest}")
    proc = subprocess.run(kaggle_cli() + args + ["-p", str(dest)], capture_output=True, text=True)
    if proc.returncode != 0 or "403" in proc.stdout + proc.stderr or "accept" in (proc.stdout + proc.stderr).lower():
        print(f"[!] {name}: download failed:\n{(proc.stdout + proc.stderr).strip()[-500:]}")
        if name in RULES:
            print(f"    Accept the competition rules with your Kaggle account first: {RULES[name]}")
        return False

    zips = list(dest.glob("*.zip"))
    if not zips and not (dest / marker).exists():
        print(f"[!] {name}: Kaggle reported success but no archive arrived in {dest}. Output:\n"
              f"{(proc.stdout + proc.stderr).strip()[-500:]}")
        return False
    for zip_path in zips:
        print(f"    extracting {zip_path.name}")
        try:
            extract(zip_path, dest, name)
        except (zipfile.BadZipFile, FileNotFoundError, OSError) as e:
            print(f"[!] {name}: extracting {zip_path.name} failed: {e}")
            return False
        if not keep_zips:
            zip_path.unlink()
    ok = (dest / marker).exists()
    print(f"[{'+' if ok else '!'}] {name}: {'ready' if ok else 'expected file missing: ' + marker}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", help=f"any of {list(DATASETS)} (default: all)")
    parser.add_argument("--keep-zips", action="store_true", help="Keep the downloaded archives")
    args = parser.parse_args()
    names = args.datasets or list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; choose from {list(DATASETS)}")
    results = {n: fetch(n, args.keep_zips) for n in names}
    print("\nSummary: " + ", ".join(f"{n}={'ok' if ok else 'FAILED'}" for n, ok in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
