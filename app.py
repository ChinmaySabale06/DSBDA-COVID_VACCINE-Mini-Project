import re
from difflib import get_close_matches
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


DATA_PATH = "india_statewise_vaccine_template.csv"
SECOND_DOSE_MODEL_PATH = Path("second_dose_model.pkl")
STATE_COLUMN = "State"

METRIC_ALIASES = {
    "total": ["total", "how much", "how many", "people", "vaccination", "vaccinations", "done"],
    "first": ["first dose", "first", "dose 1"],
    "second": ["second dose", "second", "dose 2"],
    "male": ["male", "men"],
    "female": ["female", "women"],
}


@st.cache_data
def load_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    frame["Total Vaccinations"] = frame["Male(Individuals Vaccinated)"] + frame["Female(Individuals Vaccinated)"]
    return frame


@st.cache_resource
def load_second_dose_model() -> dict | None:
    if not SECOND_DOSE_MODEL_PATH.exists():
        return None

    with SECOND_DOSE_MODEL_PATH.open("rb") as handle:
        artifact = pickle.load(handle)

    if isinstance(artifact, dict) and "model" in artifact:
        return artifact

    return {"model": artifact, "feature_names": ["First Dose Administered"], "clip_to_first_dose": True}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def find_state(question: str, states: list[str]) -> str | None:
    normalized_question = normalize_text(question)

    for state in sorted(states, key=len, reverse=True):
        if normalize_text(state) in normalized_question:
            return state

    matches = get_close_matches(normalized_question, [normalize_text(state) for state in states], n=1, cutoff=0.8)
    if not matches:
        return None

    matched_state = matches[0]
    for state in states:
        if normalize_text(state) == matched_state:
            return state
    return None


def detect_metrics(question: str) -> list[str]:
    normalized_question = normalize_text(question)
    metrics: list[str] = []

    for metric, aliases in METRIC_ALIASES.items():
        if any(alias in normalized_question for alias in aliases):
            metrics.append(metric)

    if not metrics:
        return ["total", "first", "second", "male", "female"]

    unique_metrics: list[str] = []
    for metric in metrics:
        if metric not in unique_metrics:
            unique_metrics.append(metric)
    return unique_metrics


def build_response(row: pd.Series, metrics: list[str]) -> dict[str, int | str]:
    response = {"State": row[STATE_COLUMN]}

    metric_to_value = {
        "total": int(row["Total Vaccinations"]),
        "first": int(row["First Dose Administered"]),
        "second": int(row["Second Dose Administered"]),
        "male": int(row["Male(Individuals Vaccinated)"]),
        "female": int(row["Female(Individuals Vaccinated)"]),
    }

    metric_to_label = {
        "total": "Total Vaccinations",
        "first": "First Dose Administered",
        "second": "Second Dose Administered",
        "male": "Male Vaccinated",
        "female": "Female Vaccinated",
    }

    for metric in metrics:
        response[metric_to_label[metric]] = metric_to_value[metric]

    return response


def predict_second_dose(first_dose_value: int, artifact: dict) -> int:
    model = artifact["model"]
    feature_names = artifact.get("feature_names", ["First Dose Administered"])
    clip_to_first_dose = bool(artifact.get("clip_to_first_dose", True))

    features = pd.DataFrame([{feature_names[0]: first_dose_value}])
    predicted_second = float(model.predict(features)[0])
    predicted_second_int = int(round(predicted_second))

    if clip_to_first_dose:
        predicted_second_int = max(0, min(predicted_second_int, first_dose_value))

    return predicted_second_int


def build_second_dose_predictions(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    df = frame[[STATE_COLUMN, "First Dose Administered", "Second Dose Administered"]].copy()
    df = df.rename(
        columns={
            STATE_COLUMN: "State",
            "First Dose Administered": "FirstDose",
            "Second Dose Administered": "SecondDoseActual",
        }
    )

    df["SecondDosePredicted"] = df["FirstDose"].apply(lambda value: predict_second_dose(int(value), artifact))
    df["Residual"] = df["SecondDoseActual"] - df["SecondDosePredicted"]
    df["ActualRatioPct"] = df.apply(
        lambda row: (row["SecondDoseActual"] / row["FirstDose"] * 100.0) if row["FirstDose"] else 0.0,
        axis=1,
    )
    df["PredictedRatioPct"] = df.apply(
        lambda row: (row["SecondDosePredicted"] / row["FirstDose"] * 100.0) if row["FirstDose"] else 0.0,
        axis=1,
    )
    return df


def describe_cluster_levels(values: list[float], value: float) -> str:
    if not values:
        return "Medium"
    low = pd.Series(values).quantile(0.33)
    high = pd.Series(values).quantile(0.66)
    if value <= low:
        return "Low"
    if value >= high:
        return "High"
    return "Medium"


def compute_gender_shares(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame[[STATE_COLUMN, "Male(Individuals Vaccinated)", "Female(Individuals Vaccinated)"]].copy()
    df = df.rename(
        columns={
            STATE_COLUMN: "State",
            "Male(Individuals Vaccinated)": "Male",
            "Female(Individuals Vaccinated)": "Female",
        }
    )
    df["Total"] = df["Male"] + df["Female"]
    df["MaleSharePct"] = df.apply(lambda row: (row["Male"] / row["Total"] * 100.0) if row["Total"] else 0.0, axis=1)
    df["FemaleSharePct"] = df.apply(
        lambda row: (row["Female"] / row["Total"] * 100.0) if row["Total"] else 0.0,
        axis=1,
    )
    df["GapPctPoints"] = df["MaleSharePct"] - df["FemaleSharePct"]
    df["AbsGapPctPoints"] = df["GapPctPoints"].abs()
    return df


st.set_page_config(page_title="India Vaccine Q&A", page_icon="💉", layout="wide")

data = load_data()
state_names = data[STATE_COLUMN].dropna().astype(str).tolist()
model_artifact = load_second_dose_model()

st.title("India Vaccine Q&A")
st.caption("Q&A lookup • Prediction • Evaluation • Clustering • Gender gap")

tab_qa, tab_prediction, tab_evaluation, tab_clustering, tab_gender = st.tabs(
    [
        "Q&A",
        "Prediction",
        "Evaluation",
        "Clustering",
        "Gender gap",
    ]
)


with tab_qa:
    st.header("Q&A lookup")
    st.write("Ask a question about any state, for example: how much vaccination is done in Maharashtra?")

    with st.expander("Example questions"):
        st.write("how much vaccination is done in Maharashtra")
        st.write("how many male vaccinated in Bihar")
        st.write("show first dose in Tamil Nadu")

    with st.form("qa_form"):
        question = st.text_input("Your question", placeholder="e.g., show first dose in Tamil Nadu")
        submitted = st.form_submit_button("Ask")

    if submitted:
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            state = find_state(question, state_names)
            if state is None:
                st.error(
                    "I could not find a state name in your question. Try mentioning a state like Maharashtra or Bihar."
                )
            else:
                row = data.loc[data[STATE_COLUMN] == state].iloc[0]
                metrics = detect_metrics(question)
                response = build_response(row, metrics)

                st.success(f"Answer for {state}")

                if len(response) == 2:
                    key = next(iter(response.keys() - {"State"}))
                    st.metric(key, response[key])
                else:
                    st.dataframe(pd.DataFrame([response]), width="stretch")


with tab_prediction:
    st.header("Second-dose prediction")
    st.write("Predicts second dose using the state's first dose value from the dataset.")

    if model_artifact is None:
        st.info("Prediction model not found. Run: python train_model.py to generate second_dose_model.pkl")

    selected_state = st.selectbox("State", state_names, key="predict_state")
    selected_row = data.loc[data[STATE_COLUMN] == selected_state].iloc[0]

    first_dose_value = int(selected_row["First Dose Administered"])
    actual_second_dose = int(selected_row["Second Dose Administered"])

    predicted_second_dose: int | None = None
    if model_artifact is not None:
        predicted_second_dose = predict_second_dose(first_dose_value, model_artifact)

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.metric("First dose", f"{first_dose_value:,}")
    with top2:
        st.metric("Actual second dose", f"{actual_second_dose:,}")
    with top3:
        st.metric("Predicted second dose", f"{predicted_second_dose:,}" if predicted_second_dose is not None else "-")
    with top4:
        if predicted_second_dose is None:
            st.metric("Error (Actual−Pred)", "-")
        else:
            st.metric("Error (Actual−Pred)", f"{(actual_second_dose - predicted_second_dose):,}")

    st.divider()

    left, right = st.columns([1, 2])
    with left:
        st.subheader("State comparison")
        if predicted_second_dose is None:
            st.caption("Model not available. Train it using train_model.py")
        else:
            predicted_ratio = (predicted_second_dose / first_dose_value * 100.0) if first_dose_value else 0.0
            actual_ratio = (actual_second_dose / first_dose_value * 100.0) if first_dose_value else 0.0
            error = actual_second_dose - predicted_second_dose

            comparison = pd.DataFrame(
                [
                    {
                        "State": selected_state,
                        "First Dose": first_dose_value,
                        "Actual Second Dose": actual_second_dose,
                        "Predicted Second Dose": predicted_second_dose,
                        "Error (Actual−Pred)": error,
                        "Actual Ratio (%)": round(actual_ratio, 2),
                        "Predicted Ratio (%)": round(predicted_ratio, 2),
                    }
                ]
            )
            st.dataframe(comparison, width="stretch", hide_index=True)

    with right:
        st.subheader("First dose vs second dose")
        plot_df = data[[STATE_COLUMN, "First Dose Administered", "Second Dose Administered"]].copy()
        plot_df = plot_df.rename(
            columns={
                STATE_COLUMN: "State",
                "First Dose Administered": "FirstDose",
                "Second Dose Administered": "SecondDose",
            }
        )

        selected_actual = pd.DataFrame(
            [
                {
                    "State": selected_state,
                    "FirstDose": first_dose_value,
                    "SecondDose": actual_second_dose,
                    "Point": "Selected (Actual)",
                }
            ]
        )
        selected_pred = (
            pd.DataFrame(
                [
                    {
                        "State": selected_state,
                        "FirstDose": first_dose_value,
                        "SecondDose": predicted_second_dose,
                        "Point": "Selected (Predicted)",
                    }
                ]
            )
            if predicted_second_dose is not None
            else pd.DataFrame(columns=["State", "FirstDose", "SecondDose", "Point"])
        )

        base_points = plot_df.copy()
        base_points["Point"] = "All States (Actual)"
        chart_df = pd.concat([base_points, selected_actual, selected_pred], ignore_index=True)

        vega_spec = {
            "height": 420,
            "mark": {"type": "point", "filled": True},
            "encoding": {
                "x": {
                    "field": "FirstDose",
                    "type": "quantitative",
                    "title": "First Dose Administered",
                },
                "y": {
                    "field": "SecondDose",
                    "type": "quantitative",
                    "title": "Second Dose Administered",
                },
                "color": {
                    "field": "Point",
                    "type": "nominal",
                    "legend": {"title": ""},
                },
                "shape": {"field": "Point", "type": "nominal"},
                "size": {
                    "condition": {
                        "test": "datum.Point != 'All States (Actual)'",
                        "value": 220,
                    },
                    "value": 55,
                },
                "tooltip": [
                    {"field": "State", "type": "nominal"},
                    {"field": "Point", "type": "nominal"},
                    {"field": "FirstDose", "type": "quantitative"},
                    {"field": "SecondDose", "type": "quantitative"},
                ],
            },
        }

        st.vega_lite_chart(chart_df, vega_spec, width="stretch")


with tab_evaluation:
    st.header("Model evaluation")
    st.write("Evaluates the second-dose prediction model across all states in the dataset.")

    if model_artifact is None:
        st.info("Prediction model not found. Run: python train_model.py to generate second_dose_model.pkl")
    else:
        preds = build_second_dose_predictions(data, model_artifact)

        mae = mean_absolute_error(preds["SecondDoseActual"], preds["SecondDosePredicted"])
        mse = mean_squared_error(preds["SecondDoseActual"], preds["SecondDosePredicted"])
        rmse = mse**0.5
        mean_abs_residual = float(preds["Residual"].abs().mean())

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("MAE", f"{mae:,.0f}")
        with c2:
            st.metric("RMSE", f"{rmse:,.0f}")
        with c3:
            st.metric("Mean |Residual|", f"{mean_abs_residual:,.0f}")

        st.divider()
        st.subheader("Residual plot")
        residual_chart_spec: dict[str, Any] = {
            "height": 420,
            "mark": {"type": "point", "filled": True, "opacity": 0.8},
            "encoding": {
                "x": {
                    "field": "SecondDosePredicted",
                    "type": "quantitative",
                    "title": "Predicted Second Dose",
                },
                "y": {
                    "field": "Residual",
                    "type": "quantitative",
                    "title": "Residual (Actual − Predicted)",
                },
                "tooltip": [
                    {"field": "State", "type": "nominal"},
                    {"field": "SecondDoseActual", "type": "quantitative", "title": "Actual"},
                    {"field": "SecondDosePredicted", "type": "quantitative", "title": "Predicted"},
                    {"field": "Residual", "type": "quantitative"},
                ],
            },
        }
        st.vega_lite_chart(preds, residual_chart_spec, width="stretch")

        st.divider()
        st.subheader("Biggest under / over predictions")
        under = preds.sort_values("Residual", ascending=False).head(10)[
            [
                "State",
                "SecondDoseActual",
                "SecondDosePredicted",
                "Residual",
                "ActualRatioPct",
                "PredictedRatioPct",
            ]
        ]
        over = preds.sort_values("Residual", ascending=True).head(10)[
            [
                "State",
                "SecondDoseActual",
                "SecondDosePredicted",
                "Residual",
                "ActualRatioPct",
                "PredictedRatioPct",
            ]
        ]

        left, right = st.columns(2)
        with left:
            st.caption("Under-predicted (Actual > Predicted)")
            st.dataframe(under, width="stretch", hide_index=True)
        with right:
            st.caption("Over-predicted (Actual < Predicted)")
            st.dataframe(over, width="stretch", hide_index=True)


with tab_clustering:
    st.header("State clustering")
    st.write("Groups states/UTs with similar vaccination patterns using K-Means clustering.")

    clusters = st.slider("Number of clusters (k)", min_value=2, max_value=6, value=3, step=1)

    feature_df = data[
        [
            STATE_COLUMN,
            "First Dose Administered",
            "Second Dose Administered",
            "Male(Individuals Vaccinated)",
            "Female(Individuals Vaccinated)",
        ]
    ].copy()
    feature_df["CompletionRatio"] = feature_df.apply(
        lambda row: (row["Second Dose Administered"] / row["First Dose Administered"])
        if row["First Dose Administered"]
        else 0.0,
        axis=1,
    )
    feature_df["TotalGender"] = feature_df["Male(Individuals Vaccinated)"] + feature_df[
        "Female(Individuals Vaccinated)"
    ]
    feature_df["MaleShare"] = feature_df.apply(
        lambda row: (row["Male(Individuals Vaccinated)"] / row["TotalGender"]) if row["TotalGender"] else 0.0,
        axis=1,
    )
    feature_df["FemaleShare"] = feature_df.apply(
        lambda row: (row["Female(Individuals Vaccinated)"] / row["TotalGender"]) if row["TotalGender"] else 0.0,
        axis=1,
    )

    numeric_features = [
        "First Dose Administered",
        "Second Dose Administered",
        "Male(Individuals Vaccinated)",
        "Female(Individuals Vaccinated)",
        "CompletionRatio",
        "MaleShare",
        "FemaleShare",
    ]
    X = feature_df[numeric_features].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    feature_df["Cluster"] = kmeans.fit_predict(X_scaled) + 1

    summary = feature_df.groupby("Cluster").agg(
        Count=(STATE_COLUMN, "count"),
        AvgFirstDose=("First Dose Administered", "mean"),
        AvgSecondDose=("Second Dose Administered", "mean"),
        AvgCompletionRatio=("CompletionRatio", "mean"),
    )
    avg_first_values = summary["AvgFirstDose"].tolist()
    avg_completion_values = summary["AvgCompletionRatio"].tolist()
    summary["UptakeLevel"] = summary["AvgFirstDose"].apply(
        lambda value: describe_cluster_levels(avg_first_values, float(value))
    )
    summary["CompletionLevel"] = summary["AvgCompletionRatio"].apply(
        lambda value: describe_cluster_levels(avg_completion_values, float(value))
    )
    summary["Interpretation"] = summary.apply(
        lambda row: f"{row['UptakeLevel']} uptake / {row['CompletionLevel']} completion",
        axis=1,
    )

    summary_display = summary.reset_index().copy()
    summary_display["AvgFirstDose"] = summary_display["AvgFirstDose"].round().astype(int)
    summary_display["AvgSecondDose"] = summary_display["AvgSecondDose"].round().astype(int)
    summary_display["AvgCompletionRatio"] = (summary_display["AvgCompletionRatio"] * 100).round(2)
    summary_display = summary_display.rename(columns={"AvgCompletionRatio": "AvgCompletionRatio (%)"})

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Cluster summary")
        st.dataframe(
            summary_display[
                [
                    "Cluster",
                    "Count",
                    "AvgFirstDose",
                    "AvgSecondDose",
                    "AvgCompletionRatio (%)",
                    "Interpretation",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    with right:
        st.subheader("Scatter: first dose vs completion")
        plot_cluster = feature_df[[STATE_COLUMN, "First Dose Administered", "CompletionRatio", "Cluster"]].copy()
        plot_cluster = plot_cluster.rename(
            columns={
                STATE_COLUMN: "State",
                "First Dose Administered": "FirstDose",
            }
        )
        plot_cluster["CompletionRatioPct"] = (plot_cluster["CompletionRatio"] * 100).round(2)
        cluster_scatter_spec: dict[str, Any] = {
            "height": 420,
            "mark": {"type": "point", "filled": True},
            "encoding": {
                "x": {"field": "FirstDose", "type": "quantitative", "title": "First Dose Administered"},
                "y": {
                    "field": "CompletionRatioPct",
                    "type": "quantitative",
                    "title": "Completion Ratio (%)",
                },
                "color": {"field": "Cluster", "type": "nominal"},
                "tooltip": [
                    {"field": "State", "type": "nominal"},
                    {"field": "Cluster", "type": "nominal"},
                    {"field": "FirstDose", "type": "quantitative"},
                    {"field": "CompletionRatioPct", "type": "quantitative", "title": "Completion %"},
                ],
            },
        }
        st.vega_lite_chart(plot_cluster, cluster_scatter_spec, width="stretch")

    st.divider()
    st.subheader("State clusters")
    state_clusters = feature_df[[STATE_COLUMN, "Cluster", "CompletionRatio"]].rename(columns={STATE_COLUMN: "State"})
    state_clusters["CompletionRatio (%)"] = (state_clusters["CompletionRatio"] * 100).round(2)
    st.dataframe(
        state_clusters[["State", "Cluster", "CompletionRatio (%)"]].sort_values(["Cluster", "State"]),
        width="stretch",
        hide_index=True,
    )


with tab_gender:
    st.header("Gender gap insights")
    st.write("Shows male/female share among vaccinated individuals and highlights the largest gaps.")

    gender_df = compute_gender_shares(data)
    selected_state = st.selectbox("State", gender_df["State"].tolist(), key="gender_state")
    selected = gender_df.loc[gender_df["State"] == selected_state].iloc[0]

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.metric("Male share (%)", f"{selected['MaleSharePct']:.2f}%")
    with g2:
        st.metric("Female share (%)", f"{selected['FemaleSharePct']:.2f}%")
    with g3:
        st.metric("Gap (pp)", f"{selected['GapPctPoints']:.2f}")
    with g4:
        st.metric("Total (M+F)", f"{int(selected['Total']):,}")

    st.divider()
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Top gaps by state")
        top_gap = gender_df.sort_values("AbsGapPctPoints", ascending=False).head(10)[
            ["State", "MaleSharePct", "FemaleSharePct", "GapPctPoints", "AbsGapPctPoints"]
        ].copy()
        top_gap["MaleSharePct"] = top_gap["MaleSharePct"].round(2)
        top_gap["FemaleSharePct"] = top_gap["FemaleSharePct"].round(2)
        top_gap["GapPctPoints"] = top_gap["GapPctPoints"].round(2)
        top_gap["AbsGapPctPoints"] = top_gap["AbsGapPctPoints"].round(2)
        st.dataframe(top_gap, width="stretch", hide_index=True)

    with right:
        st.subheader("Gap chart")
        bar_spec: dict[str, Any] = {
            "height": 360,
            "mark": {"type": "bar"},
            "encoding": {
                "y": {"field": "State", "type": "nominal", "sort": "-x"},
                "x": {"field": "GapPctPoints", "type": "quantitative", "title": "Male − Female (pp)"},
                "color": {
                    "condition": {"test": "datum.GapPctPoints >= 0", "value": "#1f77b4"},
                    "value": "#d62728",
                },
                "tooltip": [
                    {"field": "State", "type": "nominal"},
                    {"field": "MaleSharePct", "type": "quantitative", "title": "Male %"},
                    {"field": "FemaleSharePct", "type": "quantitative", "title": "Female %"},
                    {"field": "GapPctPoints", "type": "quantitative", "title": "Gap (pp)"},
                ],
            },
        }
        st.vega_lite_chart(top_gap, bar_spec, width="stretch")