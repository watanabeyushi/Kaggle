"""Build Kaggle submission.tar.gz (main.py + cg + deck.csv)."""

from __future__ import annotations

import argparse
import glob
import shutil
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

AGENTS = {
    "mega-abomasnow-ex": "mega_abomasnow_ex.py",
    "dragapult-ex": "dragapult_ex.py",
    "iono-s": "iono_s.py",
    "mega-lucario-ex": "mega_lucario_ex.py",
}


def find_cg_dir() -> Path:
    local = PROJECT_ROOT / "input" / "raw" / "cg-lib" / "cg"
    if local.is_dir():
        return local
    matches = glob.glob("/kaggle/input/**/cg-lib/cg", recursive=True)
    if matches:
        return Path(matches[0])
    raise FileNotFoundError(
        "cg package not found. Place cg-lib under input/raw/cg-lib/ or add cg-lib as Kaggle Input."
    )


def find_deck_csv(agent_key: str) -> Path:
    matches = glob.glob("/kaggle/input/**/deck.csv", recursive=True)
    if matches:
        return Path(matches[0])
    deck_dir = PROJECT_ROOT / "input" / "raw" / "decks" / agent_key
    candidate = deck_dir / "deck.csv"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"deck.csv not found for {agent_key}. Add Kaggle Input dataset with deck.csv "
        f"or place at input/raw/decks/{agent_key}/deck.csv"
    )


def validate_tar(path: Path) -> list[str]:
    required = {"main.py", "deck.csv"}
    required_prefix = "cg/"
    members = set()
    with tarfile.open(path, "r:gz") as tar:
        for m in tar.getmembers():
            members.add(m.name)
    missing = [name for name in required if name not in members]
    has_cg = any(name.startswith(required_prefix) for name in members)
    if not has_cg:
        missing.append("cg/ (directory)")
    return missing


def package(agent_key: str, output: Path | None = None) -> Path:
    if agent_key not in AGENTS:
        raise ValueError(f"Unknown agent {agent_key!r}. Choose from: {', '.join(AGENTS)}")

    agent_src = PROJECT_ROOT / "agents" / AGENTS[agent_key]
    if not agent_src.is_file():
        raise FileNotFoundError(agent_src)

    cg_dir = find_cg_dir()
    deck_csv = find_deck_csv(agent_key)

    out_dir = PROJECT_ROOT / "outputs" / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = output or (out_dir / f"{agent_key}-submission.tar.gz")

    build_dir = out_dir / f".build-{agent_key}"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    shutil.copy2(agent_src, build_dir / "main.py")
    shutil.copytree(cg_dir, build_dir / "cg")
    shutil.copy2(deck_csv, build_dir / "deck.csv")

    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(build_dir / "main.py", arcname="main.py")
        tar.add(build_dir / "cg", arcname="cg")
        tar.add(build_dir / "deck.csv", arcname="deck.csv")

    shutil.rmtree(build_dir)

    missing = validate_tar(out_path)
    if missing:
        raise RuntimeError(f"submission.tar.gz is incomplete. Missing: {missing}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package agent py into Kaggle submission.tar.gz")
    parser.add_argument("--agent", required=True, choices=sorted(AGENTS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    path = package(args.agent, args.output)
    print(f"Created: {path}")
    print("Submit this .tar.gz file to Kaggle (not the .py alone).")


if __name__ == "__main__":
    main()
