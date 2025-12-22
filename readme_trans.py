import argparse
import os
import re
import shutil
from pathlib import Path

IMG_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy images referenced in a README into an assets directory and rewrite links."
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        help="Path to the README markdown file (default: README.md).",
    )
    parser.add_argument(
        "--assets",
        default="./assets",
        help="Target assets directory (absolute or relative).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without copying files or modifying the README.",
    )
    return parser.parse_args()


def sanitize_stem(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem.strip())
    cleaned = cleaned.lstrip(".")
    return cleaned or "image"


def pick_filename(alt_text: str, src_path: Path, used_names: set[str]) -> str:
    stem = alt_text.strip() or src_path.stem or "image"
    stem = sanitize_stem(stem)
    ext = src_path.suffix or ".png"
    candidate = f"{stem}{ext}"
    counter = 1
    while candidate in used_names:
        candidate = f"{stem}_{counter}{ext}"
        counter += 1
    used_names.add(candidate)
    return candidate


def build_used_names(assets_dir: Path) -> set[str]:
    used = set()
    if assets_dir.exists():
        for item in assets_dir.iterdir():
            if item.is_file():
                used.add(item.name)
    return used


def main() -> None:
    args = parse_args()
    readme_path = Path(args.readme).expanduser().resolve()
    if not readme_path.exists():
        raise FileNotFoundError(f"README not found: {readme_path}")

    readme_dir = readme_path.parent
    assets_dir = Path(args.assets)
    if not assets_dir.is_absolute():
        assets_dir = (readme_dir / assets_dir).resolve()

    text = readme_path.read_text(encoding="utf-8")
    matches = list(IMG_PATTERN.finditer(text))
    if not matches:
        print("No image links found in the README. Nothing to do.")
        return

    if not args.dry_run:
        assets_dir.mkdir(parents=True, exist_ok=True)

    used_names = build_used_names(assets_dir)
    copied: dict[Path, str] = {}
    updates: list[tuple[str, Path]] = []

    def resolve_source(raw_path: str) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else (readme_dir / path)

    def replacement(match: re.Match) -> str:
        alt_text = match.group("alt")
        raw_path = match.group("path").strip()
        src = resolve_source(raw_path).resolve()

        if not src.exists():
            print(f"Warning: source image not found, leaving link unchanged: {raw_path}")
            return match.group(0)

        if src not in copied:
            filename = pick_filename(alt_text, src, used_names)
            dest = assets_dir / filename
            if not args.dry_run:
                shutil.copy2(src, dest)
            copied[src] = filename
            updates.append((raw_path, dest))

        filename = copied[src]
        new_rel_path = Path(os.path.relpath(assets_dir / filename, start=readme_dir)).as_posix()
        return f"![{alt_text}]({new_rel_path})"

    new_text = IMG_PATTERN.sub(replacement, text)

    if args.dry_run:
        print("Dry run complete. The README would be rewritten with the following copies:")
        for original, dest in updates:
            print(f"- {original} -> {dest}")
        return

    readme_path.write_text(new_text, encoding="utf-8")
    print(f"Copied {len(copied)} images to {assets_dir}")
    for original, dest in updates:
        print(f"- {original} -> {dest}")


if __name__ == "__main__":
    main()
