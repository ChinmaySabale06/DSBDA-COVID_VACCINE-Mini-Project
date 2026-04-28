import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import pickle
from datetime import date

import sklearn

# load dataset
df = pd.read_csv("india_statewise_vaccine_template.csv")

# create total vaccinations column
df["Total Vaccinations"] = df["Male(Individuals Vaccinated)"] + df["Female(Individuals Vaccinated)"]

# remove non-numeric columns (important)
df = df.select_dtypes(include=['number']).dropna()

# target column (CHANGE if needed)
target = "Total Vaccinations"

X = df.drop(target, axis=1)
y = df[target]

# train model
model = RandomForestRegressor()
model.fit(X, y)

# save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")


def train_second_dose_model() -> None:
    required_columns = ["First Dose Administered", "Second Dose Administered"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for second-dose model: {missing}")

    data = df[required_columns].dropna().copy()

    X = data[["First Dose Administered"]]
    y = data["Second Dose Administered"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)

    r2 = model.score(X_test, y_test) if len(X_test) else float("nan")
    artifact = {
        "model": model,
        "feature_names": ["First Dose Administered"],
        "target": "Second Dose Administered",
        "clip_to_first_dose": True,
        "trained_at": date.today().isoformat(),
        "metrics": {"r2_holdout": float(r2)},
        "sklearn_version": sklearn.__version__,
    }

    with open("second_dose_model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print("Second-dose model saved successfully as second_dose_model.pkl")
    print(f"Holdout R2 (sanity check): {r2:.4f}")


if __name__ == "__main__":
    train_second_dose_model()