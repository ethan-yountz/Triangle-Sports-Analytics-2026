"""Apply EvanMiya line data to game CSVs, prioritizing newer source files.

This script reads all CSV files in data/EvanMiya, resolves duplicate
date/home/away rows by file modified time (newest wins), and updates
evan_spread for overlapping rows in:
  - data/all_games.csv
  - data/acc_games.csv
  - data/future_acc_games.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import urllib.request
from dataclasses import dataclass


TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/teams?limit=500"
)
GROUPS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/groups"
)
REQUEST_TIMEOUT_SECONDS = 20

DEFAULT_TARGETS = [
    os.path.join("data", "all_games.csv"),
    os.path.join("data", "acc_games.csv"),
    os.path.join("data", "future_acc_games.csv"),
]


@dataclass
class EvanRow:
    spread: float
    source_file: str
    source_mtime: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update evan_spread in game CSVs using data/EvanMiya rows, "
            "with newest source files overriding older overlaps."
        )
    )
    parser.add_argument(
        "--evan-dir",
        default=os.path.join("data", "EvanMiya"),
        help="Directory containing EvanMiya CSV exports.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="Comma-separated target CSV paths to update.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print update counts without writing files.",
    )
    return parser.parse_args()


def normalize_team_name(name: str) -> str:
    value = (name or "").lower().strip()
    value = value.replace("&", " and ")
    value = value.replace("'", "")
    value = value.replace(".", " ")
    value = value.replace("-", " ")
    value = value.replace("/", " ")
    value = value.replace("(", " ").replace(")", " ")
    value = re.sub(r"[^a-z0-9 ]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    if value.startswith("the "):
        value = value[4:]
    return value


def generate_name_variants(name: str) -> set[str]:
    base = normalize_team_name(name)
    if not base:
        return set()

    variants = {base}
    variants.add(re.sub(r"\bst\b", "state", base))
    variants.add(re.sub(r"\bstate\b", "st", base))
    variants.add(re.sub(r"\bst\b", "saint", base))
    variants.add(re.sub(r"\bsaint\b", "st", base))

    # Allow matching disambiguated labels like "Miami (Fla.)" against ESPN "Miami".
    if base.endswith(" fla"):
        variants.add(base[:-4].strip())
    if base.endswith(" fl"):
        variants.add(base[:-3].strip())
    if base.endswith(" ohio"):
        variants.add(base[:-5].strip())
    if base.endswith(" oh"):
        variants.add(base[:-3].strip())

    out = set()
    for item in variants:
        item = re.sub(r"\s+", " ", item).strip()
        if item:
            out.add(item)
    return out


def parse_spread(value: str | None) -> float | None:
    if value is None:
        return None
    spread = str(value).strip()
    if not spread:
        return None

    spread = spread.replace("−", "-").replace("–", "-")
    upper = spread.upper()
    if upper in {"PK", "PICK", "PICKEM", "PICK'EM", "EVEN"}:
        return 0.0

    spread = spread.replace("+", "")
    try:
        return float(spread)
    except ValueError:
        return None


def format_spread(value: float) -> str:
    return f"{value:.1f}"


def fetch_all_espn_teams() -> dict[str, dict]:
    teams_by_id: dict[str, dict] = {}

    try:
        with urllib.request.urlopen(TEAMS_URL, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.load(response)
        for sport in data.get("sports", []):
            for league in sport.get("leagues", []):
                for item in league.get("teams", []):
                    team = item.get("team", item)
                    team_id = str(team.get("id", "")).strip()
                    if not team_id:
                        continue
                    teams_by_id[team_id] = {
                        "id": team_id,
                        "displayName": team.get("displayName", ""),
                        "shortDisplayName": team.get("shortDisplayName", ""),
                        "name": team.get("name", ""),
                        "abbreviation": team.get("abbreviation", ""),
                    }
    except Exception:
        pass

    if teams_by_id:
        return teams_by_id

    with urllib.request.urlopen(GROUPS_URL, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        data = json.load(response)

    def walk_groups(groups: list[dict]) -> None:
        for group in groups:
            for team in group.get("teams", []):
                team_id = str(team.get("id", "")).strip()
                if not team_id:
                    continue
                teams_by_id[team_id] = {
                    "id": team_id,
                    "displayName": team.get("displayName", ""),
                    "shortDisplayName": team.get("shortDisplayName", ""),
                    "name": team.get("name", ""),
                    "abbreviation": team.get("abbreviation", ""),
                }
            walk_groups(group.get("children", []))

    walk_groups(data.get("groups", []))
    return teams_by_id


def build_team_variants(teams_by_id: dict[str, dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}

    for team_id, team in teams_by_id.items():
        short_name = team.get("shortDisplayName", "")
        display_name = team.get("displayName", "")
        mascot = team.get("name", "")
        abbr = team.get("abbreviation", "")

        if display_name and mascot and display_name.lower().endswith(f" {mascot.lower()}"):
            school_name = display_name[: -(len(mascot) + 1)].strip()
        else:
            school_name = short_name

        variants = set()
        for name in (short_name, display_name, school_name, abbr):
            variants.update(generate_name_variants(name))

        short_norm = normalize_team_name(short_name)
        if short_norm == "miami":
            variants.update({"miami fla", "miami fl", "miami florida"})
        if short_norm in {"miami oh", "miami ohio"}:
            variants.update({"miami oh", "miami ohio"})

        out[team_id] = {v for v in variants if v}
    return out


def load_evan_rows(evan_dir: str) -> tuple[dict[tuple[str, str, str], EvanRow], list[str]]:
    files = glob.glob(os.path.join(evan_dir, "*.csv"))
    files = sorted(files, key=lambda p: os.path.getmtime(p))

    rows: dict[tuple[str, str, str], EvanRow] = {}
    used_files: set[str] = set()

    for path in files:
        mtime = os.path.getmtime(path)
        base = os.path.basename(path)
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue

            date_col = "Date" if "Date" in reader.fieldnames else ("date" if "date" in reader.fieldnames else None)
            home_col = "home" if "home" in reader.fieldnames else None
            away_col = "away" if "away" in reader.fieldnames else None
            line_col = "line" if "line" in reader.fieldnames else None
            if not date_col or not home_col or not away_col or not line_col:
                continue

            for row in reader:
                date_value = (row.get(date_col) or "").strip()
                home_value = normalize_team_name(row.get(home_col) or "")
                away_value = normalize_team_name(row.get(away_col) or "")
                spread_value = parse_spread(row.get(line_col))
                if not date_value or not home_value or not away_value or spread_value is None:
                    continue

                key = (date_value, home_value, away_value)
                rows[key] = EvanRow(
                    spread=spread_value,
                    source_file=base,
                    source_mtime=mtime,
                )
                used_files.add(base)

    return rows, sorted(used_files)


def build_evan_by_date(
    evan_rows: dict[tuple[str, str, str], EvanRow]
) -> dict[str, dict[tuple[str, str], EvanRow]]:
    by_date: dict[str, dict[tuple[str, str], EvanRow]] = {}
    for (date_value, home_norm, away_norm), row in evan_rows.items():
        by_date.setdefault(date_value, {})[(home_norm, away_norm)] = row
    return by_date


def choose_match(
    date_value: str,
    home_variants: set[str],
    away_variants: set[str],
    evan_by_date: dict[str, dict[tuple[str, str], EvanRow]],
) -> tuple[float, str] | None:
    day_rows = evan_by_date.get(date_value)
    if not day_rows:
        return None

    best_direct: EvanRow | None = None
    best_reverse: EvanRow | None = None

    for home_name in home_variants:
        for away_name in away_variants:
            direct = day_rows.get((home_name, away_name))
            if direct and (
                best_direct is None or direct.source_mtime >= best_direct.source_mtime
            ):
                best_direct = direct

            reverse = day_rows.get((away_name, home_name))
            if reverse and (
                best_reverse is None or reverse.source_mtime >= best_reverse.source_mtime
            ):
                best_reverse = reverse

    if best_direct and best_reverse:
        if best_direct.source_mtime >= best_reverse.source_mtime:
            return best_direct.spread, best_direct.source_file
        return -best_reverse.spread, best_reverse.source_file

    if best_direct:
        return best_direct.spread, best_direct.source_file
    if best_reverse:
        return -best_reverse.spread, best_reverse.source_file
    return None


def update_target_csv(
    target_path: str,
    team_variants: dict[str, set[str]],
    evan_by_date: dict[str, dict[tuple[str, str], EvanRow]],
    dry_run: bool,
) -> dict:
    with open(target_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "evan_spread" not in fieldnames:
        fieldnames.append("evan_spread")

    stats = {
        "rows": len(rows),
        "rows_with_evan_date": 0,
        "matched_rows": 0,
        "updated_rows": 0,
        "unchanged_rows": 0,
        "unmatched_rows": 0,
        "samples": [],
    }

    for row in rows:
        date_value = (row.get("date") or "").strip()
        if date_value not in evan_by_date:
            continue

        stats["rows_with_evan_date"] += 1

        home_id = str(row.get("home_team_id") or "").strip()
        away_id = str(row.get("away_team_id") or "").strip()
        home_variants = team_variants.get(home_id, set())
        away_variants = team_variants.get(away_id, set())

        match = choose_match(date_value, home_variants, away_variants, evan_by_date)
        if not match:
            stats["unmatched_rows"] += 1
            continue

        spread, source_file = match
        stats["matched_rows"] += 1
        old_value = (row.get("evan_spread") or "").strip()
        new_value = format_spread(spread)

        if old_value != new_value:
            row["evan_spread"] = new_value
            stats["updated_rows"] += 1
            if len(stats["samples"]) < 8:
                stats["samples"].append(
                    {
                        "event_id": row.get("event_id", ""),
                        "date": date_value,
                        "old": old_value,
                        "new": new_value,
                        "source": source_file,
                    }
                )
        else:
            stats["unchanged_rows"] += 1

    if not dry_run and stats["updated_rows"] > 0:
        with open(target_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return stats


def main() -> None:
    args = parse_args()
    targets = [p.strip() for p in args.targets.split(",") if p.strip()]

    evan_rows, used_files = load_evan_rows(args.evan_dir)
    if not evan_rows:
        raise RuntimeError(f"No usable EvanMiya rows found in: {args.evan_dir}")
    evan_by_date = build_evan_by_date(evan_rows)

    teams_by_id = fetch_all_espn_teams()
    team_variants = build_team_variants(teams_by_id)

    date_values = sorted(evan_by_date.keys())
    print(f"evan_files_used: {len(used_files)}")
    print(f"evan_rows_latest: {len(evan_rows)}")
    print(f"evan_date_range: {date_values[0]} .. {date_values[-1]}")
    print(f"espn_teams_loaded: {len(teams_by_id)}")
    print(f"mode: {'dry-run' if args.dry_run else 'write'}")

    for path in targets:
        stats = update_target_csv(
            target_path=path,
            team_variants=team_variants,
            evan_by_date=evan_by_date,
            dry_run=args.dry_run,
        )
        print(
            f"{path}: rows={stats['rows']} "
            f"rows_with_evan_date={stats['rows_with_evan_date']} "
            f"matched={stats['matched_rows']} unmatched={stats['unmatched_rows']} "
            f"updated={stats['updated_rows']} unchanged={stats['unchanged_rows']}"
        )
        for sample in stats["samples"]:
            print(
                "  sample_update:"
                f" event_id={sample['event_id']}"
                f" date={sample['date']}"
                f" old={sample['old'] if sample['old'] else '<blank>'}"
                f" new={sample['new']}"
                f" source={sample['source']}"
            )


if __name__ == "__main__":
    main()
