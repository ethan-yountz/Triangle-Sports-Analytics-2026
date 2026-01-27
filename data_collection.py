import concurrent.futures
import csv
import datetime
import json
import os
import re
import urllib.error
import urllib.request


GROUPS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/groups"
)
TEAM_SCHEDULE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/teams/{team_id}/schedule?season={season}"
)
SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/summary?event={event_id}"
)
ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/"
    "basketball/leagues/mens-college-basketball/events/{event_id}/"
    "competitions/{competition_id}/odds"
)
ESPN_BET_PROVIDER_IDS = {"58", "59"}
DRAFTKINGS_PROVIDER_IDS = {"100"}
REQUEST_TIMEOUT_SECONDS = 10
def fetch_acc_teams():
    with urllib.request.urlopen(GROUPS_URL) as response:
        data = json.load(response)

    def walk(groups):
        for group in groups:
            if group.get("abbreviation") == "acc":
                return group.get("teams", [])
            found = walk(group.get("children", []))
            if found:
                return found
        return []

    return walk(data.get("groups", []))


def write_acc_csv(teams, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["team_id", "team_name"])
        for team in teams:
            team_id = team.get("id")
            team_name = team.get("displayName") or team.get("name")
            if team_id and team_name:
                writer.writerow([team_id, team_name])


def current_season_year():
    today = datetime.date.today()
    if today.month >= 10:
        return today.year + 1
    return today.year


def parse_score(score_value):
    if score_value is None:
        return None
    if isinstance(score_value, dict):
        score_value = score_value.get("value") or score_value.get("displayValue")
    try:
        return int(score_value)
    except (TypeError, ValueError):
        return None


def parse_event_date(event_date):
    if not event_date:
        return None
    return event_date.split("T", 1)[0]


def fetch_preferred_spread(event_id, competition_id):
    odds_url = ODDS_URL.format(event_id=event_id, competition_id=competition_id)
    try:
        with urllib.request.urlopen(odds_url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return None
    fallback_spread = None
    for item in data.get("items", []):
        provider_id = item.get("provider", {}).get("id")
        if provider_id is None:
            continue
        provider_id = str(provider_id)
        spread = item.get("spread")
        try:
            spread_value = float(spread)
        except (TypeError, ValueError):
            spread_value = None
        if provider_id in ESPN_BET_PROVIDER_IDS and spread_value is not None:
            return spread_value
        if provider_id in DRAFTKINGS_PROVIDER_IDS and spread_value is not None:
            fallback_spread = spread_value
    return fallback_spread


def add_draftkings_spreads(games, max_workers=8):
    game_by_event = {game.get("event_id"): game for game in games if game.get("event_id")}

    def fetch_for_game(game):
        event_id = game.get("event_id")
        competition_id = game.get("competition_id")
        if not event_id or not competition_id:
            return event_id, None
        return event_id, fetch_preferred_spread(event_id, competition_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_for_game, game) for game in games]
        for future in concurrent.futures.as_completed(futures):
            event_id, spread = future.result()
            if event_id in game_by_event:
                game_by_event[event_id]["spread"] = spread


def fetch_acc_games(teams, season_year):
    games = {}
    for team in teams:
        team_id = team.get("id")
        if not team_id:
            continue
        schedule_url = TEAM_SCHEDULE_URL.format(team_id=team_id, season=season_year)
        with urllib.request.urlopen(schedule_url) as response:
            data = json.load(response)
        for event in data.get("events", []):
            event_id = event.get("id")
            if not event_id or event_id in games:
                continue
            competition = (event.get("competitions") or [{}])[0]
            status = competition.get("status", {}).get("type", {})
            if not status.get("completed"):
                continue
            competition_id = competition.get("id") or event_id
            competitors = competition.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})
            home_score = parse_score(home.get("score"))
            away_score = parse_score(away.get("score"))
            neutral_site = competition.get("neutralSite")
            if neutral_site is None and "neutralSite" not in competition:
                neutral_site = None
            games[event_id] = {
                "event_id": event_id,
                "date": parse_event_date(event.get("date")),
                "home_team_id": home.get("team", {}).get("id"),
                "away_team_id": away.get("team", {}).get("id"),
                "home_score": home_score,
                "away_score": away_score,
                "neutral_site": neutral_site,
                "spread": None,
                "margin": (
                    home_score - away_score
                    if home_score is not None and away_score is not None
                    else None
                ),
                "competition_id": competition_id,
            }
    return list(games.values())


def fetch_event_team_stats(event_id):
    summary_url = SUMMARY_URL.format(event_id=event_id)
    try:
        with urllib.request.urlopen(summary_url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return []
    teams = data.get("boxscore", {}).get("teams", [])
    rows = []
    for team in teams:
        team_id = team.get("team", {}).get("id")
        if not team_id:
            continue
        row = {
            "event_id": event_id,
            "team_id": team_id,
        }
        for stat in team.get("statistics", []):
            name = stat.get("name")
            if not name:
                continue
            row[name] = stat.get("displayValue")
        rows.append(row)
    return rows


def collect_team_stats(event_ids, max_workers=8):
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_event_team_stats, event_id) for event_id in event_ids]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())
    return rows


def add_game_dates_to_team_stats(rows, games):
    event_dates = {game.get("event_id"): game.get("date") for game in games}
    for row in rows:
        row["date"] = event_dates.get(row.get("event_id"))
    return rows



def write_team_stats_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stat_columns = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in {"event_id", "team_id", "date"}
        }
    )
    headers = ["event_id", "team_id", "date"] + stat_columns
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(header) for header in headers])


def write_acc_games_csv(games, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "event_id",
                "date",
                "home_team_id",
                "away_team_id",
                "home_score",
                "away_score",
                "neutral_site",
                "spread",
                "margin",
            ]
        )
        for game in games:
            writer.writerow(
                [
                    game.get("event_id"),
                    game.get("date"),
                    game.get("home_team_id"),
                    game.get("away_team_id"),
                    game.get("home_score"),
                    game.get("away_score"),
                    game.get("neutral_site"),
                    game.get("spread"),
                    game.get("margin"),
                ]
            )


def main():
    teams = fetch_acc_teams()
    output_path = os.path.join("data", "acc_teams.csv")
    write_acc_csv(teams, output_path)
    print(f"Wrote {len(teams)} teams to {output_path}")
    season_year = current_season_year()
    games = fetch_acc_games(teams, season_year)
    add_draftkings_spreads(games)
    games_output_path = os.path.join("data", "acc_games.csv")
    write_acc_games_csv(games, games_output_path)
    print(
        f"Wrote {len(games)} ACC games for season {season_year} to {games_output_path}"
    )
    team_stats = collect_team_stats([game["event_id"] for game in games])
    team_stats = add_game_dates_to_team_stats(team_stats, games)
    team_stats_output = os.path.join("data", "acc_game_stats.csv")
    write_team_stats_csv(team_stats, team_stats_output)
    print(f"Wrote {len(team_stats)} team stat rows to {team_stats_output}")


if __name__ == "__main__":
    main()