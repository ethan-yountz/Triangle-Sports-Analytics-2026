"""
Scrape Bart Torvik ratings for all teams by game date.
Uses Playwright for browser automation to bypass Cloudflare.

Outputs:
- data/torvik_asof_ratings_all_teams.csv: all scraped rows with mapping status
- data/torvik_unmapped_teams.csv: unique unmapped/ambiguous team names for manual mapping
"""

import csv
import json
import os
import re
import time
import urllib.request
import unicodedata
import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta

GROUPS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/groups"
)
TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/teams?limit=500"
)

# Torvik aliases that are usually ambiguous or not direct ESPN short names.
# These overrides are only used before dynamic matching.
TORVIK_TEAM_ID_OVERRIDES = {
    "Albany": "399",
    "Appalachian St.": "2026",
    "Cal Baptist": "2856",
    "Connecticut": "41",
    "Illinois Chicago": "82",
    "Louisiana Monroe": "2433",
    "Miami FL": "2390",
    "Miami OH": "193",
    "Mississippi": "145",
    "Nebraska Omaha": "2437",
    "Nicholls St.": "2447",
    "Ole Miss": "145",
    "Queens": "2511",
    "Seattle": "2547",
    "Southeastern Louisiana": "2545",
    "St. Thomas": "2900",
    "Tennessee Martin": "2630",
    "Grambling St.": "2755",
    "McNeese St.": "2377",
    "Sam Houston St.": "2534",
    "Texas A&M Corpus Chris": "357",
    "UMKC": "140",
    "A&M-Corpus Christi": "357",
    "Ark.-Pine Bluff": "2029",
    "MVSU": "2400",
    "PFW": "2870",
    "LIU": "112358",
    "USC Upstate": "2908",
}


def get_unique_dates(games_path: str) -> list[str]:
    """Get sorted unique dates from a games CSV."""
    dates = set()
    with open(games_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("date"):
                dates.add(row["date"])
    return sorted(dates)


def current_season_year(today: date | None = None) -> int:
    """Return the current CBB season year (e.g., Jan 2026 -> 2026, Nov 2026 -> 2027)."""
    if today is None:
        today = datetime.today().date()
    return today.year + 1 if today.month >= 10 else today.year


def generate_season_dates(season_year: int, end_date: date) -> list[str]:
    """
    Generate daily YYYY-MM-DD dates from season start (Nov 1 of prior year)
    through end_date inclusive.
    """
    start_date = date(season_year - 1, 11, 1)
    if end_date < start_date:
        return []
    days = (end_date - start_date).days + 1
    return [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


def load_team_game_dates(games_path: str) -> dict[str, list[datetime]]:
    """Load game dates per team from a games CSV."""
    team_dates: dict[str, list[datetime]] = {}
    with open(games_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            date_str = row.get("date")
            if not date_str:
                continue
            game_date = datetime.strptime(date_str, "%Y-%m-%d")
            for team_id in (row.get("home_team_id"), row.get("away_team_id")):
                if not team_id:
                    continue
                team_dates.setdefault(str(team_id), []).append(game_date)

    for team_id in team_dates:
        team_dates[team_id] = sorted(team_dates[team_id])
    return team_dates


def calc_days_since_last_game(team_id: str, as_of_date: datetime, team_dates: dict) -> int:
    """Calculate days since last game before as_of_date. Returns 1 if no prior game."""
    dates = team_dates.get(str(team_id), [])
    prior_games = [d for d in dates if d < as_of_date]
    if not prior_games:
        return 1
    return (as_of_date - max(prior_games)).days


def calc_games_last_7_days(team_id: str, as_of_date: datetime, team_dates: dict) -> int:
    """Count games in the 7 days before as_of_date. Returns 1 if no games found."""
    dates = team_dates.get(str(team_id), [])
    seven_days_ago = as_of_date - timedelta(days=7)
    games_in_window = [d for d in dates if seven_days_ago <= d < as_of_date]
    return len(games_in_window) if games_in_window else 1


def normalize_team_name(name: str) -> str:
    """Normalize team name for robust cross-site matching."""
    value = (name or "").lower().strip()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.replace("&", " and ")
    value = value.replace("'", "")
    value = value.replace(".", " ")
    value = value.replace("-", " ")
    value = value.replace("/", " ")
    value = value.replace("(", " ").replace(")", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-z0-9 ]", "", value)
    value = re.sub(r"\s+", " ", value)
    # Collapse split initialisms ("n c state" -> "nc state").
    while True:
        collapsed = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", value)
        if collapsed == value:
            break
        value = collapsed
    return value.strip()


def generate_name_variants(name: str) -> set[str]:
    """Generate normalized name variants for matching."""
    base = normalize_team_name(name)
    if not base:
        return set()

    variants = {base}

    # Common article variation.
    if base.startswith("the "):
        variants.add(base[4:])

    # Common Saint/State abbreviation variation.
    variants.add(re.sub(r"\bst\b", "state", base))
    variants.add(re.sub(r"\bstate\b", "st", base))
    variants.add(re.sub(r"\bst\b", "saint", base))
    variants.add(re.sub(r"\bsaint\b", "st", base))

    cleaned = set()
    for item in variants:
        normalized = re.sub(r"\s+", " ", item).strip()
        if normalized:
            cleaned.add(normalized)
    return cleaned


def fetch_all_espn_teams() -> dict[str, dict]:
    """Fetch all D1 team nodes available via ESPN team list endpoint."""
    teams_by_id: dict[str, dict] = {}

    try:
        with urllib.request.urlopen(TEAMS_URL, timeout=20) as response:
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

    # Fallback to groups endpoint if team list endpoint fails.
    with urllib.request.urlopen(GROUPS_URL, timeout=20) as response:
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


def build_espn_name_maps(teams_by_id: dict[str, dict]) -> dict[str, dict[str, set[str]]]:
    """Build normalized ESPN name indexes by source type."""
    maps = {
        "short": defaultdict(set),
        "school": defaultdict(set),
        "display": defaultdict(set),
        "abbr": defaultdict(set),
    }

    for team_id, team in teams_by_id.items():
        short_name = team.get("shortDisplayName", "")
        display_name = team.get("displayName", "")
        mascot = team.get("name", "")
        abbr = team.get("abbreviation", "")

        for variant in generate_name_variants(short_name):
            maps["short"][variant].add(team_id)

        if display_name and mascot and display_name.lower().endswith(f" {mascot.lower()}"):
            school_name = display_name[: -(len(mascot) + 1)].strip()
        else:
            school_name = short_name

        for variant in generate_name_variants(school_name):
            maps["school"][variant].add(team_id)

        for variant in generate_name_variants(display_name):
            maps["display"][variant].add(team_id)

        for variant in generate_name_variants(abbr):
            maps["abbr"][variant].add(team_id)

    return maps


def map_torvik_team_to_espn_id(
    torvik_team: str,
    espn_maps: dict[str, dict[str, set[str]]],
) -> tuple[str | None, str, list[str]]:
    """Map a Torvik team name to ESPN team_id with mapping status and candidate IDs."""
    if torvik_team in TORVIK_TEAM_ID_OVERRIDES:
        return TORVIK_TEAM_ID_OVERRIDES[torvik_team], "override", []

    variants = generate_name_variants(torvik_team)
    if not variants:
        return None, "unmapped", []

    ambiguous_candidates = set()
    for source in ("short", "school", "display", "abbr"):
        candidates = set()
        index = espn_maps[source]
        for variant in variants:
            candidates.update(index.get(variant, set()))
        if len(candidates) == 1:
            return next(iter(candidates)), source, []
        if len(candidates) > 1:
            ambiguous_candidates.update(candidates)

    combined = set()
    for source in ("short", "school", "display", "abbr"):
        index = espn_maps[source]
        for variant in variants:
            combined.update(index.get(variant, set()))

    if len(combined) == 1:
        return next(iter(combined)), "combined", []
    if len(combined) > 1:
        return None, "ambiguous", sorted(combined)
    if ambiguous_candidates:
        return None, "ambiguous", sorted(ambiguous_candidates)

    return None, "unmapped", []


def scrape_torvik_for_date(page, date_str: str, season_year: int) -> list[dict]:
    """Scrape barttorvik for all teams as of a given date."""
    date_key = date_str.replace("-", "")
    season_start_key = f"{season_year - 1}1101"
    url = (
        f"https://barttorvik.com/?year={season_year}&sort=&hteam=&t2value=&conlimit=All&"
        f"state=All&begin={season_start_key}&end={date_key}&top=0&revquad=0&quad=5&"
        "venue=All&type=All&mingames=0"
    )

    page.goto(url, wait_until="networkidle")
    time.sleep(1)

    for _ in range(10):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(0.3)

    data = page.evaluate(
        """() => {
        const rows = document.querySelectorAll('table tbody tr');
        const data = [];
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length > 21) {
                const teamLink = cells[1]?.querySelector('a');
                const teamName = teamLink?.childNodes[0]?.textContent?.trim() || '';
                const games = cells[3]?.childNodes[0]?.textContent?.trim() || '';
                const record = cells[4]?.childNodes[0]?.textContent?.trim() || '';
                const adjO = cells[5]?.childNodes[0]?.textContent?.trim() || '';
                const adjD = cells[6]?.childNodes[0]?.textContent?.trim() || '';
                const barthag = cells[7]?.childNodes[0]?.textContent?.trim() || '';
                const efg = cells[8]?.childNodes[0]?.textContent?.trim() || '';
                const efgd = cells[9]?.childNodes[0]?.textContent?.trim() || '';
                const tor = cells[10]?.childNodes[0]?.textContent?.trim() || '';
                const tord = cells[11]?.childNodes[0]?.textContent?.trim() || '';
                const orb = cells[12]?.childNodes[0]?.textContent?.trim() || '';
                const drb = cells[13]?.childNodes[0]?.textContent?.trim() || '';
                const ftr = cells[14]?.childNodes[0]?.textContent?.trim() || '';
                const ftrd = cells[15]?.childNodes[0]?.textContent?.trim() || '';
                const twoPtPct = cells[16]?.childNodes[0]?.textContent?.trim() || '';
                const twoPtPctD = cells[17]?.childNodes[0]?.textContent?.trim() || '';
                const threePtPct = cells[18]?.childNodes[0]?.textContent?.trim() || '';
                const threePtPctD = cells[19]?.childNodes[0]?.textContent?.trim() || '';
                const threePtRate = cells[20]?.childNodes[0]?.textContent?.trim() || '';
                const threePtRateD = cells[21]?.childNodes[0]?.textContent?.trim() || '';
                const adjT = cells[22]?.childNodes[0]?.textContent?.trim() || '';
                const wab = cells[23]?.childNodes[0]?.textContent?.trim() || '';

                if (teamName && !teamName.includes('Avg')) {
                    data.push({
                        team: teamName, games: games, record: record,
                        adj_o: adjO, adj_d: adjD, barthag: barthag,
                        efg: efg, efgd: efgd, tor: tor, tord: tord,
                        orb: orb, drb: drb, ftr: ftr, ftrd: ftrd,
                        two_pt_pct: twoPtPct, two_pt_pct_d: twoPtPctD,
                        three_pt_pct: threePtPct, three_pt_pct_d: threePtPctD,
                        three_pt_rate: threePtRate, three_pt_rate_d: threePtRateD,
                        adj_tempo: adjT, wab: wab
                    });
                }
            }
        });
        return data;
    }"""
    )

    return data


def write_csv(rows: list[dict], fieldnames: list[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive end date for scraping in YYYY-MM-DD format (default: today).",
    )
    return parser.parse_args()


def resolve_end_date(end_date_str: str | None) -> date:
    if not end_date_str:
        return datetime.today().date()
    return datetime.strptime(end_date_str, "%Y-%m-%d").date()


def main() -> None:
    from playwright.sync_api import sync_playwright

    args = parse_args()
    rest_source_path = os.path.join("data", "all_games.csv")
    if not os.path.exists(rest_source_path):
        rest_source_path = os.path.join("data", "acc_games.csv")

    all_teams_output_path = os.path.join("data", "torvik_asof_ratings_all_teams.csv")
    unmapped_output_path = os.path.join("data", "torvik_unmapped_teams.csv")

    today = resolve_end_date(args.end_date)
    season_year = current_season_year(today)
    dates = generate_season_dates(season_year, today)
    team_dates = load_team_game_dates(rest_source_path)

    teams_by_id = fetch_all_espn_teams()
    espn_maps = build_espn_name_maps(teams_by_id)

    print(f"Loaded {len(teams_by_id)} ESPN teams for mapping.", flush=True)
    print(f"Season year: {season_year}", flush=True)
    if dates:
        print(f"Scraping {len(dates)} dates ({dates[0]} through {dates[-1]})...", flush=True)
    else:
        print("No dates to scrape.", flush=True)

    all_rows: list[dict] = []
    unmapped_index: dict[str, dict] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for i, date_str in enumerate(dates, start=1):
            print(f"[{i}/{len(dates)}] Scraping {date_str}...", flush=True)
            try:
                as_of_date = datetime.strptime(date_str, "%Y-%m-%d")
                scraped_rows = scrape_torvik_for_date(page, date_str, season_year)

                mapped_count = 0
                for row in scraped_rows:
                    team_id, map_status, candidate_ids = map_torvik_team_to_espn_id(row["team"], espn_maps)
                    days_since = None
                    games_7d = None

                    if team_id:
                        mapped_count += 1
                        days_since = calc_days_since_last_game(team_id, as_of_date, team_dates)
                        games_7d = calc_games_last_7_days(team_id, as_of_date, team_dates)
                    else:
                        info = unmapped_index.setdefault(
                            row["team"],
                            {
                                "team": row["team"],
                                "status": map_status,
                                "row_count": 0,
                                "dates": set(),
                                "candidate_ids": set(),
                            },
                        )
                        info["row_count"] += 1
                        info["dates"].add(date_str)
                        info["candidate_ids"].update(candidate_ids)
                    row_with_map = {
                        "team_id": team_id or "",
                        "date": date_str,
                        "map_status": map_status,
                        **row,
                        "days_since_last_game": days_since,
                        "games_last_7_days": games_7d,
                    }
                    all_rows.append(row_with_map)

                print(
                    f"  Scraped {len(scraped_rows)} teams, mapped {mapped_count}, "
                    f"unmapped {len(scraped_rows) - mapped_count}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  Error: {exc}", flush=True)

            time.sleep(0.5)

        browser.close()

    all_fieldnames = ["team_id", "date", "map_status"] + [
        "team",
        "games",
        "record",
        "adj_o",
        "adj_d",
        "barthag",
        "efg",
        "efgd",
        "tor",
        "tord",
        "orb",
        "drb",
        "ftr",
        "ftrd",
        "two_pt_pct",
        "two_pt_pct_d",
        "three_pt_pct",
        "three_pt_pct_d",
        "three_pt_rate",
        "three_pt_rate_d",
        "adj_tempo",
        "wab",
        "days_since_last_game",
        "games_last_7_days",
    ]

    unmapped_rows = []
    for _, info in sorted(unmapped_index.items(), key=lambda item: item[0].lower()):
        candidate_ids = sorted(info["candidate_ids"])
        candidate_names = [teams_by_id.get(cid, {}).get("shortDisplayName", "") for cid in candidate_ids]
        unmapped_rows.append(
            {
                "team": info["team"],
                "status": info["status"],
                "row_count": info["row_count"],
                "dates_seen": len(info["dates"]),
                "candidate_team_ids": "|".join(candidate_ids),
                "candidate_team_names": "|".join([n for n in candidate_names if n]),
            }
        )

    if all_rows:
        write_csv(all_rows, all_fieldnames, all_teams_output_path)
        print(f"Wrote {len(all_rows)} all-team rows to {all_teams_output_path}", flush=True)

    write_csv(
        unmapped_rows,
        [
            "team",
            "status",
            "row_count",
            "dates_seen",
            "candidate_team_ids",
            "candidate_team_names",
        ],
        unmapped_output_path,
    )
    print(
        f"Wrote {len(unmapped_rows)} unique unmapped/ambiguous teams to {unmapped_output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
