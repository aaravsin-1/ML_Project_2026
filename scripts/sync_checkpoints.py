"""Back up experiment checkpoints to GitHub, and restore them elsewhere (e.g. on Colab).

Experiment checkpoints and results live in the artifacts folder. This script mirrors that
folder to a separate `run-state` branch (via a git worktree, so the main checkout and
`master` are never touched), and can restore it on another machine, so a run interrupted
on one computer resumes on another.

    python scripts/sync_checkpoints.py --loop 600     # push every 10 min (runs until stopped)
    python scripts/sync_checkpoints.py                # push once
    python scripts/sync_checkpoints.py --restore      # pull the branch into the artifacts folder

Raw data is never synced: it is re-downloaded from Kaggle with scripts/fetch_datasets.py.
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR  # noqa: E402

BRANCH = "run-state"
WORKTREE = ROOT.parent / f"{ROOT.name}_run_state"
MAX_FILE_MB = 50  # GitHub rejects files over 100 MB; checkpoints are far smaller


def git(*args, cwd=ROOT, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def ensure_worktree():
    if (WORKTREE / ".git").exists():
        return
    remote_has_branch = bool(git("ls-remote", "--heads", "origin", BRANCH, check=False).stdout.strip())
    if remote_has_branch:
        git("fetch", "origin", BRANCH)
        git("worktree", "add", "-B", BRANCH, str(WORKTREE), f"origin/{BRANCH}")
    else:
        git("worktree", "add", "--detach", str(WORKTREE))
        git("checkout", "--orphan", BRANCH, cwd=WORKTREE)
        git("rm", "-rf", "--quiet", ".", cwd=WORKTREE, check=False)
        (WORKTREE / "README.md").write_text(
            "Experiment checkpoints and results synced by scripts/sync_checkpoints.py.\n"
            "Restore them with `python scripts/sync_checkpoints.py --restore`.\n")


def push_once() -> str:
    ensure_worktree()
    dest = WORKTREE / "artifacts"
    skipped = []
    for src in ARTIFACTS_DIR.rglob("*"):
        if not src.is_file():
            continue
        if src.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            skipped.append(src.name)
            continue
        target = dest / src.relative_to(ARTIFACTS_DIR)
        if not target.exists() or target.stat().st_mtime < src.stat().st_mtime or target.stat().st_size != src.stat().st_size:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    git("add", "-A", "-f", ".", cwd=WORKTREE)
    if not git("diff", "--cached", "--quiet", cwd=WORKTREE, check=False).returncode:
        return "no changes"
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    git("commit", "-q", "-m", f"sync checkpoints {stamp}", cwd=WORKTREE)
    push = git("push", "-q", "-u", "origin", BRANCH, cwd=WORKTREE, check=False)
    note = f" (skipped >{MAX_FILE_MB} MB: {skipped})" if skipped else ""
    return ("pushed" if push.returncode == 0 else f"commit ok, push failed: {push.stderr.strip()[-300:]}") + note


def restore():
    """Copies files from origin/run-state into the artifacts folder, never overwriting newer local files."""
    if not git("ls-remote", "--heads", "origin", BRANCH, check=False).stdout.strip():
        print(f"[=] no {BRANCH} branch on origin yet: nothing to restore")
        return
    git("fetch", "origin", BRANCH)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "state.tar"
        with open(archive, "wb") as fh:
            subprocess.run(["git", "archive", f"origin/{BRANCH}"], cwd=ROOT, check=True, stdout=fh)
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(tmp, filter="data")
            except TypeError:  # Python without tar extraction filters
                tar.extractall(tmp)
        src_root = Path(tmp) / "artifacts"
        copied = 0
        for src in src_root.rglob("*") if src_root.exists() else []:
            if not src.is_file():
                continue
            target = ARTIFACTS_DIR / src.relative_to(src_root)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied += 1
    print(f"[+] restored {copied} file(s) from origin/{BRANCH} into {ARTIFACTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--restore", action="store_true", help="Pull checkpoints from GitHub instead of pushing")
    parser.add_argument("--loop", type=int, default=0, metavar="SECONDS", help="Push repeatedly at this interval")
    args = parser.parse_args()
    if args.restore:
        restore()
        return
    while True:
        try:
            print(f"[{time.strftime('%H:%M:%S')}] {push_once()}", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"[{time.strftime('%H:%M:%S')}] sync error: {(e.stderr or '').strip()[-300:]}", flush=True)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
