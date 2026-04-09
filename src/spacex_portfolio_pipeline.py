from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "Spacex.csv"
OUTPUT_PATH = BASE_DIR / "data" / "spacex_cleaned.csv"

SUCCESS_KEYWORDS = ("Success", "Controlled", "True")
FAILURE_KEYWORDS = ("Failure", "No attempt", "Precluded", "Uncontrolled", "False")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)

    # Clean raw column names from the CSV
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace("__", "_", regex=False)
    )

    # Rename to simple portfolio-friendly names
    df = df.rename(
        columns={
            "TimeUTC": "TimeUTC",
            "Booster_Version": "BoosterVersion",
            "Launch_Site": "LaunchSite",
            "PAYLOAD_MASS_KG_": "PayloadMass",
            "Mission_Outcome": "MissionOutcome",
            "Landing_Outcome": "LandingOutcome",
        }
    )

    return df


def build_target(landing_outcome: str) -> int:
    text = str(landing_outcome).strip()

    if any(keyword in text for keyword in SUCCESS_KEYWORDS):
        return 1
    if any(keyword in text for keyword in FAILURE_KEYWORDS):
        return 0
    return 0


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()

    # Extra safeguard in case any column still has odd formatting
    clean_df.columns = [
        col.strip()
        .replace(" ", "_")
        .replace("__", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        for col in clean_df.columns
    ]

    clean_df = clean_df.rename(
        columns={
            "PAYLOAD_MASS_KG": "PayloadMass",
            "Landing_Outcome": "LandingOutcome",
            "Mission_Outcome": "MissionOutcome",
            "Launch_Site": "LaunchSite",
            "Booster_Version": "BoosterVersion",
            "Time_UTC": "TimeUTC",
        }
    )

    clean_df["Class"] = clean_df["LandingOutcome"].apply(build_target)

    clean_df["Date"] = pd.to_datetime(
        clean_df["Date"], format="%d-%m-%Y", errors="coerce"
    )
    clean_df["Year"] = clean_df["Date"].dt.year
    clean_df["Month"] = clean_df["Date"].dt.month

    clean_df["BoosterFamily"] = (
        clean_df["BoosterVersion"]
        .astype(str)
        .str.extract(r"^(F9\s[^B]+|F9\sB\d|F9\sFT|F9\sv1\.\d)", expand=False)
    )
    clean_df["BoosterFamily"] = clean_df["BoosterFamily"].fillna(
        clean_df["BoosterVersion"].astype(str).str.split().str[:2].str.join(" ")
    )

    selected_columns = [
        "Date",
        "Year",
        "Month",
        "BoosterVersion",
        "BoosterFamily",
        "LaunchSite",
        "Payload",
        "PayloadMass",
        "Orbit",
        "Customer",
        "MissionOutcome",
        "LandingOutcome",
        "Class",
    ]

    clean_df = clean_df[selected_columns]
    return clean_df


def print_project_summary(df: pd.DataFrame) -> None:
    print("\n=== PROJECT SUMMARY ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\n=== LANDING SUCCESS RATE ===")
    print(f"Success rate: {df['Class'].mean():.2%}")

    print("\n=== SUCCESS RATE BY LAUNCH SITE ===")
    print(df.groupby("LaunchSite")["Class"].mean().sort_values(ascending=False).round(3))

    print("\n=== SUCCESS RATE BY ORBIT ===")
    print(df.groupby("Orbit")["Class"].mean().sort_values(ascending=False).round(3).head(10))

    print("\n=== PAYLOAD MASS SUMMARY ===")
    print(df["PayloadMass"].describe().round(2))

    print("\nGenerating visualization...")

    df.groupby("LaunchSite")["Class"].mean().plot(kind="bar")

    plt.title("Landing Success Rate by Launch Site")
    plt.ylabel("Success Rate")
    plt.xlabel("Launch Site")
    plt.tight_layout()

    # save figure
    fig_path = BASE_DIR / "data" / "launch_site_success_rate.png"
    plt.savefig(fig_path)

    plt.show()


def train_models(df: pd.DataFrame) -> Dict[str, float]:
    model_df = df[
        ["PayloadMass", "Orbit", "LaunchSite", "BoosterFamily", "Year", "Class"]
    ].copy()

    X = model_df.drop(columns="Class")
    y = model_df["Class"]

    numeric_features = ["PayloadMass", "Year"]
    categorical_features = ["Orbit", "LaunchSite", "BoosterFamily"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=42
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scores: Dict[str, float] = {}
    best_name = ""
    best_score = -1.0
    best_report = ""

    print("\n=== MODEL RESULTS ===")
    for name, model in models.items():
        clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        score = accuracy_score(y_test, predictions)
        scores[name] = score
        print(f"{name}: {score:.3f}")

        if score > best_score:
            best_name = name
            best_score = score
            best_report = classification_report(y_test, predictions, digits=3)

    print(f"\nBest model: {best_name} ({best_score:.3f})")
    print("\nClassification report for best model:")
    print(best_report)

    return scores


def main() -> None:
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    clean_df.to_csv(OUTPUT_PATH, index=False)

    print_project_summary(clean_df)

    scores = train_models(clean_df)

    # Save model scores
    results_path = BASE_DIR / "data" / "model_scores.csv"
    scores_df = pd.DataFrame(scores.items(), columns=["Model", "Accuracy"])
    scores_df.to_csv(results_path, index=False)

    print(f'\nSaved cleaned dataset to: {OUTPUT_PATH.resolve()}')
    print(f'Saved model results to: {results_path.resolve()}')


if __name__ == "__main__":
    main()

