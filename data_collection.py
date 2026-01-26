import csv
import json
import os
import urllib.request


GROUPS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/groups"
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


def main():
    teams = fetch_acc_teams()
    output_path = os.path.join("data", "acc_teams.csv")
    write_acc_csv(teams, output_path)
    print(f"Wrote {len(teams)} teams to {output_path}")


if __name__ == "__main__":
    main()