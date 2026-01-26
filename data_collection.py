import csv
import datetime
import json
import os
import urllib.request


GROUPS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/groups"
)
TEAM_SCHEDULE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/teams/{team_id}/schedule?season={season}"
)


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
                "margin": (
                    home_score - away_score
                    if home_score is not None and away_score is not None
                    else None
                ),
            }
    return list(games.values())


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
    games_output_path = os.path.join("data", "acc_games.csv")
    write_acc_games_csv(games, games_output_path)
    print(
        f"Wrote {len(games)} ACC games for season {season_year} to {games_output_path}"
    )


if __name__ == "__main__":
    main()