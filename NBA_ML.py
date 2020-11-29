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
    df = pd.read_csv("./data/games.csv")
    df_teams = pd.read_csv("./data/teams.csv")
    df = df.dropna()
    df_teams = df_teams.dropna()

    # Test model using only existing features in games
    # Don't use points home or points (win team scores more)
    y = df["HOME_TEAM_WINS"]
    X = df.drop(
        columns=[
            "HOME_TEAM_WINS",
            "GAME_DATE_EST",
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
    df_teams = df_teams.rename(columns={"TEAM_ID": "HOME_TEAM_ID"})
    temp = pd.merge(df, df_teams, on="HOME_TEAM_ID")
    df = pd.merge(
        df,
        temp[["HOME_TEAM_ID", "ARENACAPACITY", "YEARFOUNDED"]],
        on="HOME_TEAM_ID",
        how="left",
    )
    print(df)


if __name__ == "__main__":
    sys.exit(main())
