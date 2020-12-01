# Final Project - Using ML and NBA games to predict winners
# 0 = home team loses; 1 = home team wins
# Brenton Wilder

# Import libraries
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier


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

    # Begin Feature Engineering (1-16 from data)
    # Feature 17: Home Team arena capacity
    # Feature 18: Year Home Team was founded
    # Feature 19: Conference of Home Team, West=0 and East=1
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

    # Feature 20: Away Team arena capacity
    # Feature 21: Year Away Team was founded
    # Feature 22: Conference of Away Team, West=0 and East=1
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

    # Feature 23: Current win percentage for Home Team
    # Feature 24: Games played so far for Home Team (0-82)
    # Feature 25: Current W for Home Team
    # Feature 26: Current L for Home Team
    df_rnk.drop(
        [
            "LEAGUE_ID",
            "SEASON_ID",
            "CONFERENCE",
            "TEAM",
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

    # Feature 27: Current win percentage for Away Team
    # Feature 28: Games played so far for Away Team (0-82)
    # Feature 29: Current W for Away Team
    # Feature 30: Current L for Away Team
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

    # Feature 31: Day of the week game was on (0-6)
    df["WEEKDAY"] = df["GAME_DATE_EST"].apply(
        lambda x: (
            datetime.fromordinal(datetime(1900, 1, 1).toordinal() + x - 2)
        ).weekday()
    )

    # Feature 32: Weekend game?  (1=True,0=False)
    df["WEEKEND_GAME"] = df["WEEKDAY"].apply(lambda x: 0 if x < 5 else 1)

    # Feature 33: Month number game was on (1-12)
    df["MONTH_NUM"] = df["GAME_DATE_EST"].apply(
        lambda x: (
            datetime.fromordinal(datetime(1900, 1, 1).toordinal() + x - 2)
        ).strftime("%m")
    )

    # Feature 34: Difference in FG % (Home Team FG - Away Team FG)
    df = df.astype(float)
    df["DIFF_FG"] = df["FG_PCT_home"] - df["FG_PCT_away"]

    # Feature 35: Difference in Reb (Home Team Reb - Away Team Reb)
    df["DIFF_REB"] = df["REB_home"] - df["REB_away"]

    # Feature 36: Difference in Ast (Home Team Ast - Away Team Ast)
    df["DIFF_AST"] = df["AST_home"] - df["AST_away"]

    # Feature 37: Difference in FT (Home Team FT - Away Team FT)
    df["DIFF_FT"] = df["FT_PCT_home"] - df["FT_PCT_away"]

    # Feature 38: Difference in 3PT percent (Home - Away)
    df["DIFF_3PT"] = df["FG3_PCT_home"] - df["FG3_PCT_away"]

    # print(df)

    # Final model
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=2424
    )
    classifiers = [
        ExtraTreeClassifier(random_state=2408),
        DecisionTreeClassifier(random_state=2408),
        MLPClassifier(),
        KNeighborsClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        BaggingClassifier(),
        RandomForestClassifier(random_state=2408),
        BernoulliNB(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        LogisticRegression(),
        LogisticRegressionCV(),
        NuSVC(probability=True),
        SVC(probability=True),
        XGBClassifier(),
    ]

    result_table = pd.DataFrame(columns=["classifiers", "fpr", "tpr", "auc"])

    for cls in classifiers:
        model = cls.fit(X_train, y_train)
        yproba = model.predict_proba(X_test)[::, 1]
        fpr, tpr, _ = roc_curve(y_test, yproba)
        auc = roc_auc_score(y_test, yproba)
        result_table = result_table.append(
            {
                "classifiers": cls.__class__.__name__,
                "fpr": fpr,
                "tpr": tpr,
                "auc": auc,
            },
            ignore_index=True,
        )

    # Plot bar graph comparing ROC/AUC
    fig = px.bar(result_table, x="classifiers", y="auc")
    fig.update_traces(
        marker_color="rgb(158,202,225)",
        marker_line_color="rgb(8,48,107)",
        marker_line_width=1.5,
        opacity=0.9,
    )
    fig.update_layout(
        paper_bgcolor="rgb(0,0,0,0)",
        title="auc values from ROC curve",
        font=dict(family="Times New Roman", size=20, color="black"),
    )
    fig.show()

    # Show feature importance for best model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    final = GradientBoostingClassifier()
    final.fit(X_train, y_train)
    y_pred = final.predict(X_test)
    y_score = final.predict_proba(X_test)[::, 1]
    print("_________Final Model_________")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    imp = pd.Series(final.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print("FEATURE IMPORTANCE:")
    print(imp)

    # Plot the ROC curve for final model
    fpr2, tpr2, thresholds = roc_curve(y_test, y_score)
    auc2 = roc_auc_score(y_test, y_score)
    fig2 = px.area(
        x=fpr2,
        y=tpr2,
        title=f"ROC Curve for Final Model (AUC={auc2})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
        width=700,
        height=500,
    )
    fig2.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    fig2.update_layout(
        paper_bgcolor="rgb(0,0,0,0)",
        font=dict(family="Times New Roman", size=20, color="black"),
    )
    fig2.update_yaxes(scaleanchor="x", scaleratio=1)
    fig2.update_xaxes(constrain="domain")
    fig2.show()

    # Plot the importance chart for final model
    importances = final.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig3 = go.Figure()
    fig3.add_trace(
        go.Bar(
            x=indices,
            y=importances,
        )
    )
    fig3.update_layout(
        paper_bgcolor="rgb(0,0,0,0)",
        font=dict(family="Times New Roman", size=20, color="black"),
    )
    fig3.show()

    # Plot the heat map of the final model features
    def df_to_plotly(df):
        return {
            "z": df.values.tolist(),
            "x": df.columns.tolist(),
            "y": df.index.tolist(),
        }

    X = df.drop(
        columns=[
            "HOME_TEAM_WINS",
            "PTS_home",
            "PTS_away",
        ],
        axis=1,
    )
    fig4 = go.Figure(data=go.Heatmap(df_to_plotly(X)))
    fig4.show()

    # Export model to pickle file
    with open("./model.pkl", "wb") as model_pkl:
        pickle.dump(final, model_pkl)


if __name__ == "__main__":
    sys.exit(main())
