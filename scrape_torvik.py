"""
Scrape Bart Torvik ratings for ACC teams by game date.
Uses Playwright for browser automation to bypass Cloudflare.
"""

import csv
import os
import re
import time
from datetime import datetime

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


def scrape_torvik_for_date(page, date_str: str) -> list[dict]:
    """Scrape barttorvik for ACC teams as of a given date."""
    # Convert YYYY-MM-DD to YYYYMMDD
    date_key = date_str.replace("-", "")
    url = f"https://barttorvik.com/?year=2026&sort=&hteam=&t2value=&conlimit=ACC&state=All&begin=20251101&end={date_key}&top=0&revquad=0&quad=5&venue=All&type=All&mingames=0"
    
    page.goto(url, wait_until="networkidle")
    time.sleep(1)  # Extra wait for table to render
    
    # Extract data via JS
    data = page.evaluate("""() => {
        const rows = document.querySelectorAll('table tbody tr');
        const data = [];
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length > 7) {
                const teamLink = cells[1]?.querySelector('a');
                const teamName = teamLink?.childNodes[0]?.textContent?.trim() || '';
                const adjO = cells[5]?.childNodes[0]?.textContent?.trim() || '';
                const adjD = cells[6]?.childNodes[0]?.textContent?.trim() || '';
                const barthag = cells[7]?.childNodes[0]?.textContent?.trim() || '';
                const adjT = cells[20]?.childNodes[0]?.textContent?.trim() || '';
                const wab = cells[21]?.childNodes[0]?.textContent?.trim() || '';
                if (teamName && !teamName.includes('Avg')) {
                    data.push({team: teamName, adj_o: adjO, adj_d: adjD, barthag: barthag, adj_tempo: adjT, wab: wab});
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
            results.append({
                "team_id": team_id,
                "date": date_str,
                "team": row["team"],
                "barthag": row["barthag"],
                "adj_o": row["adj_o"],
                "adj_d": row["adj_d"],
                "adj_tempo": row["adj_tempo"],
                "wab": row["wab"],
            })
    return results


def main():
    from playwright.sync_api import sync_playwright
    
    games_path = os.path.join("data", "acc_games.csv")
    output_path = os.path.join("data", "torvik_asof_ratings.csv")
    
    dates = get_unique_dates(games_path)
    print(f"Scraping {len(dates)} dates...")
    
    all_rows = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        for i, date_str in enumerate(dates):
            print(f"[{i+1}/{len(dates)}] Scraping {date_str}...")
            try:
                rows = scrape_torvik_for_date(page, date_str)
                all_rows.extend(rows)
                print(f"  Got {len(rows)} teams")
            except Exception as e:
                print(f"  Error: {e}")
            time.sleep(0.5)  # Be nice to the server
        
        browser.close()
    
    # Write CSV
    if all_rows:
        fieldnames = ["team_id", "date", "team", "barthag", "adj_o", "adj_d", "adj_tempo", "wab"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {output_path}")
    else:
        print("No data scraped.")


if __name__ == "__main__":
    main()
