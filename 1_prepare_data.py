"""
Prepare real-estate listings: enrich with Egyptian Arabic voice hooks via Gemini.
"""

from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors

API_KEY = "AIzaSyCvxo9rw1VLjM61qF8fL6vhD51c23ykNOo"
COLUMNS = ("type", "title", "location", "bedroom", "bathroom", "size_sqm", "price")
MODEL = "gemini-2.5-flash"
RATE_LIMIT_SLEEP_SEC = 1
BAD_MARKERS = ("Error from Gemini", "API key not valid")

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "properties.csv"
CSV_FALLBACK = ROOT / "data" / "raw" / "properties.csv"
OUTPUT_DIR = ROOT / "data" / "processed"


def cleanup_bad_markdown(processed_dir: Path) -> int:
    """Remove .md files that contain failed Gemini / API-key error text."""
    if not processed_dir.is_dir():
        return 0

    removed = 0
    for md_file in processed_dir.glob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"[cleanup] Could not read {md_file.name}: {exc}")
            continue

        if any(marker in text for marker in BAD_MARKERS):
            try:
                md_file.unlink()
                removed += 1
                print(f"[cleanup] Deleted {md_file.name}")
            except OSError as exc:
                print(f"[cleanup] Could not delete {md_file.name}: {exc}")

    if removed:
        print(f"[cleanup] Removed {removed} bad file(s).")
    else:
        print("[cleanup] No bad markdown files found.")
    return removed


def resolve_csv_path() -> Path:
    if CSV_PATH.is_file():
        return CSV_PATH
    if CSV_FALLBACK.is_file():
        return CSV_FALLBACK
    raise FileNotFoundError(
        f"properties.csv not found at {CSV_PATH} or {CSV_FALLBACK}"
    )


def yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_details(prop: dict[str, str]) -> str:
    return (
        f"**{prop['title']}** is a {prop['type']} in {prop['location']}. "
        f"It offers {prop['bedroom']} bedrooms and {prop['bathroom']} bathrooms "
        f"across {prop['size_sqm']} sqm, listed at {prop['price']} EGP."
    )


def build_markdown(prop: dict[str, str], hook: str) -> str:
    frontmatter_lines = ["---"]
    for key in COLUMNS:
        frontmatter_lines.append(f"{key}: {yaml_quote(prop[key])}")
    frontmatter_lines.append("---")

    return (
        "\n".join(frontmatter_lines)
        + "\n\n"
        + "# Details\n\n"
        + build_details(prop)
        + "\n\n"
        + "# Arabic Hook\n\n"
        + hook.strip()
        + "\n"
    )


def read_all_properties(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")

        missing = [c for c in COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV missing required columns {missing}. "
                f"Found: {list(reader.fieldnames)}"
            )

        for row_num, row in enumerate(reader, start=2):
            prop = {col: (row.get(col) or "").strip() for col in COLUMNS}
            if not any(prop.values()):
                print(f"[skip] Empty row at line {row_num}")
                continue
            rows.append(prop)

    if not rows:
        raise ValueError(f"No property rows read from {csv_path}")
    return rows


def build_hook_prompt(prop: dict[str, str]) -> str:
    return f"""You are a creative Egyptian real-estate copywriter.

Write ONE short, catchy marketing hook in Egyptian Arabic (Ammiya / عامية مصرية) for a voice agent to speak aloud when presenting this listing.

Rules:
- Egyptian dialect only (not formal MSA).
- 1–2 sentences max; natural when spoken.
- Highlight the strongest selling point (location, size, price, or type).
- No emojis, no hashtags, no bullet points.
- Return ONLY the hook text, nothing else.

Property:
- Type: {prop['type']}
- Title: {prop['title']}
- Location: {prop['location']}
- Bedrooms: {prop['bedroom']}
- Bathrooms: {prop['bathroom']}
- Size (sqm): {prop['size_sqm']}
- Price (EGP): {prop['price']}
"""


def generate_arabic_hook(client: genai.Client, prop: dict[str, str]) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=build_hook_prompt(prop),
    )
    text = (response.text or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty hook")
    return text


def save_property(index: int, prop: dict[str, str], hook: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"property_{index}.md"
    out_path.write_text(build_markdown(prop, hook), encoding="utf-8")
    return out_path


def main() -> int:
    cleanup_bad_markdown(OUTPUT_DIR)

    try:
        csv_path = resolve_csv_path()
        properties = read_all_properties(csv_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    total = len(properties)
    print(f"Source: {csv_path}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Processing {total} properties with {MODEL}...\n")

    client = genai.Client(api_key=API_KEY)
    succeeded = 0
    failed = 0

    for index, prop in enumerate(properties):
        title = (prop["title"] or f"row-{index}")[:70]
        try:
            hook = generate_arabic_hook(client, prop)
            out_path = save_property(index, prop, hook)
            succeeded += 1
            print(f"[{index + 1}/{total}] Saved {out_path.name} — {title}")
        except genai_errors.APIError as exc:
            failed += 1
            print(f"[{index + 1}/{total}] Gemini API error — {title}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"[{index + 1}/{total}] Failed — {title}: {exc}")
        finally:
            if index + 1 < total:
                time.sleep(RATE_LIMIT_SLEEP_SEC)

    print(f"\nDone. {succeeded} saved, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
