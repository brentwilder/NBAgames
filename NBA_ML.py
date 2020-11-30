# Final Project - Using ML and NBA games to predict winners
# 0 = home team loses; 1 = home team wins
# Brenton Wilder

# Import libraries
import sys

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
    df = df.sort_values("GAME_DATE_EST").set_index(["GAME_DATE_EST"])

    df_tms = pd.read_csv("./data/teams.csv", na_filter=False)
    df_tms.replace("", nan_value, inplace=True)
    df_tms = df_tms.dropna()

    df_rnk = pd.read_csv("./data/ranking.csv", na_filter=False)
    df_rnk.replace("", nan_value, inplace=True)
    df_rnk = df_rnk.dropna()

    # Test model using only existing features in games
    # Don't use points home or points (win team scores more)
    y = df["HOME_TEAM_WINS"]
    X = df.drop(
        columns=[
            "HOME_TEAM_WINS",
            "GAME_STATUS_TEXT",
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
    # Feature 3: Conference, West=0 and East=1
    df_tms = df_tms.rename(columns={"TEAM_ID": "HOME_TEAM_ID"})
    df = pd.merge(
        df,
        df_tms[["HOME_TEAM_ID", "ARENACAPACITY", "YEARFOUNDED", "CONFERENCE"]],
        on="HOME_TEAM_ID",
        how="left",
    )

    # Feature 4: Current win percentage for Home Team
    df_rnk.sort_values("STANDINGSDATE", inplace=True)
    df_rnk.set_index("STANDINGSDATE", inplace=True)
    df = pd.merge_asof(
        df,
        df_rnk.add_suffix("_homeTeam"),
        left_index=True,
        right_index=True,
        left_by="HOME_TEAM_ID",
        right_by="TEAM_ID" + "_homeTeam",
        allow_exact_matches=False,
    )

    print(df)


if __name__ == "__main__":
    sys.exit(main())
