# Final Project - Using ML and NBA games to predict winners
# 0 = home team loses; 1 = home team wins
# Brenton Wilder

# Import libraries
import sys
from datetime import datetime

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():

    # Load dataframes and clean
    df = pd.read_csv("./data/games.csv", na_filter=False)
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df = df.dropna()
    df.drop_duplicates()
    df.sort_values("GAME_DATE_EST")
    df.set_index("GAME_DATE_EST")
    df.drop(["GAME_STATUS_TEXT", "TEAM_ID_home"], axis=1, inplace=True)

    df_tm = pd.read_csv("./data/teams.csv", na_filter=False)
    df_tm.replace("", nan_value, inplace=True)
    df_tm = df_tm.dropna()

    df_rnk = pd.read_csv("./data/ranking.csv", na_filter=False)
    df_rnk.replace("", nan_value, inplace=True)
    df_rnk = df_rnk.dropna()
    df_rnk.sort_values("STANDINGSDATE")
    df_rnk.set_index("STANDINGSDATE")

    # Test model using only existing features in games
    # Don't use points home or points (win team scores more)
    y = df["HOME_TEAM_WINS"]
    X = df.drop(
        columns=[
            "HOME_TEAM_WINS",
            "PTS_home",
            "PTS_away",
        ],
        axis=1,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("_________Test Model_________")
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print("FEATURE IMPORTANCE:")
    print(imp)

    # Begin Feature Engineering
    # Feature 1: Home Team arena capacity
    # Feature 2: Year Home Team was founded
    # Feature 3: Conference of Home Team, West=0 and East=1
    homID = "HOME_TEAM_ID"
    df_tm = df_tm.rename(columns={"TEAM_ID": "HOME_TEAM_ID"})
    df = pd.merge(
        df,
        df_tm[[homID, "ARENACAPACITY", "YEARFOUNDED", "CONFERENCE"]],
        on="HOME_TEAM_ID",
        how="left",
    )
    df = df.rename(
        columns={
            "YEARFOUNDED": "YEARFOUNDED_homeTeam",
            "CONFERENCE": "CONFERENCE_homeTeam",
            "ARENACAPACITY": "ARENACAPACITY_homeTeam",
        }
    )

    # Feature 4: Away Team arena capacity
    # Feature 5: Year Away Team was founded
    # Feature 6: Conference of Away Team, West=0 and East=1
    visID = "VISITOR_TEAM_ID"
    df_tm = df_tm.rename(columns={"HOME_TEAM_ID": "VISITOR_TEAM_ID"})
    df = pd.merge(
        df,
        df_tm[[visID, "ARENACAPACITY", "YEARFOUNDED", "CONFERENCE"]],
        on="VISITOR_TEAM_ID",
        how="left",
    )
    df = df.rename(
        columns={
            "YEARFOUNDED": "YEARFOUNDED_awayTeam",
            "CONFERENCE": "CONFERENCE_awayTeam",
            "ARENACAPACITY": "ARENACAPACITY_awayTeam",
        }
    )

    # Feature 7: Current win percentage for Home Team
    # Feature 8: Games played so far for Home Team (0-82)
    df_rnk.drop(
        [
            "LEAGUE_ID",
            "SEASON_ID",
            "CONFERENCE",
            "TEAM",
            "W",
            "L",
            "HOME_RECORD",
            "ROAD_RECORD",
        ],
        axis=1,
        inplace=True,
    )
    df = pd.merge_asof(
        df,
        df_rnk.add_suffix("_homeTeam"),
        left_index=True,
        right_index=True,
        left_by="HOME_TEAM_ID",
        right_by="TEAM_ID" + "_homeTeam",
        allow_exact_matches=False,
    )
    df = df.dropna()
    homStd = "STANDINGSDATE_homeTeam"
    df.drop(["TEAM_ID_homeTeam", homStd], axis=1, inplace=True)

    # Feature 9: Current win percentage for Away Team
    # Feature 10: Games played so far for Away Team (0-82)
    df = pd.merge_asof(
        df,
        df_rnk.add_suffix("_awayTeam"),
        left_index=True,
        right_index=True,
        left_by="VISITOR_TEAM_ID",
        right_by="TEAM_ID" + "_awayTeam",
        allow_exact_matches=False,
    )
    df = df.dropna()
    visStd = "STANDINGSDATE_awayTeam"
    df.drop(["TEAM_ID_awayTeam", visStd], axis=1, inplace=True)

    # Feature 11: Day of the week game was on (0-6)
    df["WEEKDAY"] = df["GAME_DATE_EST"].apply(
        lambda x: (
            datetime.fromordinal(datetime(1900, 1, 1).toordinal() + x - 2)
        ).weekday()
    )

    # Feature 12: Month number game was on (1-12)
    df["MONTH_NUM"] = df["GAME_DATE_EST"].apply(
        lambda x: (
            datetime.fromordinal(datetime(1900, 1, 1).toordinal() + x - 2)
        ).strftime("%m")
    )

    # Feature 13: Difference in FG % (Home Team FG - Away Team FG)
    df = df.astype(float)
    df["DIFF_FG"] = df["FG_PCT_home"] - df["FG_PCT_away"]

    # Feature 14: Difference in Reb (Home Team Reb - Away Team Reb)
    df["DIFF_REB"] = df["REB_home"] - df["REB_away"]

    # Feature 15: Difference in Ast (Home Team Ast - Away Team Ast)
    df["DIFF_AST"] = df["AST_home"] - df["AST_away"]

    # Feature 16: Difference in FT (Home Team FT - Away Team FT)
    df["DIFF_FT"] = df["FT_PCT_home"] - df["FT_PCT_away"]

    print(df)


if __name__ == "__main__":
    sys.exit(main())
