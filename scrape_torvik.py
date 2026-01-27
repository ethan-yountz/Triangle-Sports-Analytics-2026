"""
Scrape Bart Torvik ratings for ACC teams by game date.
Uses Playwright for browser automation to bypass Cloudflare.
"""

import csv
import os
import re
import time
from datetime import datetime, timedelta

# Team name mapping: Torvik name -> ESPN team_id
TORVIK_TO_TEAM_ID = {
    "Duke": "150",
    "Virginia": "258",
    "Louisville": "97",
    "Clemson": "228",
    "North Carolina": "153",
    "N.C. State": "152",
    "SMU": "2567",
    "Miami FL": "2390",
    "Virginia Tech": "259",
    "California": "25",
    "Syracuse": "183",
    "Wake Forest": "154",
    "Stanford": "24",
    "Notre Dame": "87",
    "Florida St.": "52",
    "Pittsburgh": "221",
    "Georgia Tech": "59",
    "Boston College": "103",
}


def get_unique_dates(games_path: str) -> list[str]:
    """Get sorted unique dates from acc_games.csv."""
    dates = set()
    with open(games_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("date"):
                dates.add(row["date"])
    return sorted(dates)


def load_team_game_dates(games_path: str) -> dict[str, list[datetime]]:
    """Load game dates per team from acc_games.csv."""
    team_dates = {}
    with open(games_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            date_str = row.get("date")
            if not date_str:
                continue
            game_date = datetime.strptime(date_str, "%Y-%m-%d")
            for team_id in [row.get("home_team_id"), row.get("away_team_id")]:
                if team_id:
                    if team_id not in team_dates:
                        team_dates[team_id] = []
                    team_dates[team_id].append(game_date)
    # Sort dates for each team
    for team_id in team_dates:
        team_dates[team_id] = sorted(team_dates[team_id])
    return team_dates


def calc_days_since_last_game(team_id: str, as_of_date: datetime, team_dates: dict) -> int:
    """Calculate days since last game before as_of_date. Returns 1 if no prior game."""
    dates = team_dates.get(team_id, [])
    prior_games = [d for d in dates if d < as_of_date]
    if not prior_games:
        return 1
    last_game = max(prior_games)
    return (as_of_date - last_game).days


def calc_games_last_7_days(team_id: str, as_of_date: datetime, team_dates: dict) -> int:
    """Count games in the 7 days before as_of_date. Returns 1 if no games found."""
    dates = team_dates.get(team_id, [])
    seven_days_ago = as_of_date - timedelta(days=7)
    games_in_window = [d for d in dates if seven_days_ago <= d < as_of_date]
    return len(games_in_window) if games_in_window else 1


def scrape_torvik_for_date(page, date_str: str, team_dates: dict) -> list[dict]:
    """Scrape barttorvik for ACC teams as of a given date."""
    as_of_date = datetime.strptime(date_str, "%Y-%m-%d")
    # Convert YYYY-MM-DD to YYYYMMDD
    date_key = date_str.replace("-", "")
    url = f"https://barttorvik.com/?year=2026&sort=&hteam=&t2value=&conlimit=ACC&state=All&begin=20251101&end={date_key}&top=0&revquad=0&quad=5&venue=All&type=All&mingames=0"
    
    page.goto(url, wait_until="networkidle")
    time.sleep(1)  # Extra wait for table to render
    
    # Extract data via JS - all Torvik columns
    data = page.evaluate("""() => {
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
    }""")
    
    # Map to team_ids
    results = []
    for row in data:
        team_id = TORVIK_TO_TEAM_ID.get(row["team"])
        if team_id:
            days_since = calc_days_since_last_game(team_id, as_of_date, team_dates)
            games_7d = calc_games_last_7_days(team_id, as_of_date, team_dates)
            results.append({
                "team_id": team_id,
                "date": date_str,
                "team": row["team"],
                "games": row["games"],
                "record": row["record"],
                "adj_o": row["adj_o"],
                "adj_d": row["adj_d"],
                "barthag": row["barthag"],
                "efg": row["efg"],
                "efgd": row["efgd"],
                "tor": row["tor"],
                "tord": row["tord"],
                "orb": row["orb"],
                "drb": row["drb"],
                "ftr": row["ftr"],
                "ftrd": row["ftrd"],
                "two_pt_pct": row["two_pt_pct"],
                "two_pt_pct_d": row["two_pt_pct_d"],
                "three_pt_pct": row["three_pt_pct"],
                "three_pt_pct_d": row["three_pt_pct_d"],
                "three_pt_rate": row["three_pt_rate"],
                "three_pt_rate_d": row["three_pt_rate_d"],
                "adj_tempo": row["adj_tempo"],
                "wab": row["wab"],
                "days_since_last_game": days_since,
                "games_last_7_days": games_7d,
            })
    return results


def main():
    from playwright.sync_api import sync_playwright
    
    games_path = os.path.join("data", "acc_games.csv")
    output_path = os.path.join("data", "torvik_asof_ratings.csv")
    
    dates = get_unique_dates(games_path)
    team_dates = load_team_game_dates(games_path)
    print(f"Scraping {len(dates)} dates...")
    
    all_rows = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        for i, date_str in enumerate(dates):
            print(f"[{i+1}/{len(dates)}] Scraping {date_str}...")
            try:
                rows = scrape_torvik_for_date(page, date_str, team_dates)
                all_rows.extend(rows)
                print(f"  Got {len(rows)} teams")
            except Exception as e:
                print(f"  Error: {e}")
            time.sleep(0.5)  # Be nice to the server
        
        browser.close()
    
    # Write CSV
    if all_rows:
        fieldnames = [
            "team_id", "date", "team", "games", "record",
            "adj_o", "adj_d", "barthag",
            "efg", "efgd", "tor", "tord",
            "orb", "drb", "ftr", "ftrd",
            "two_pt_pct", "two_pt_pct_d",
            "three_pt_pct", "three_pt_pct_d",
            "three_pt_rate", "three_pt_rate_d",
            "adj_tempo", "wab",
            "days_since_last_game", "games_last_7_days"
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {output_path}")
    else:
        print("No data scraped.")


if __name__ == "__main__":
    main()
