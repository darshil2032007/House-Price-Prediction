"""
House Price Prediction — Streamlit Application
-------------------------------------------------
Loads the trained Multiple Linear Regression model and automatically
builds the entire input interface (and the prediction dataframe) from
the processed training dataset. No feature names, categories, or
ranges are hardcoded — everything is derived from
`data/processed/housing_processed.csv` at runtime.

After every prediction, all inputs are reset back to their sensible
default values (median / most common values from the training data)
so the form is ready for a fresh entry.

Run with:
    streamlit run app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
MODEL_PATH = "models/multiple_linear_regression.pkl"
DATA_PATH = "data/processed/housing_processed.csv"
TARGET_COLUMN = "price"
MAX_CATEGORICAL_UNIQUE = 10  # small integer columns are shown as selectboxes

# Human-friendly labels and helper text for common housing features.
# Any column not listed here automatically falls back to a title-cased
# version of its name, so the app still works with unseen datasets.
FEATURE_LABELS = {
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "sqft_living": "Living Area (sqft)",
    "sqft_lot": "Lot Size (sqft)",
    "sqft_above": "Above-Ground Area (sqft)",
    "sqft_basement": "Basement Area (sqft)",
    "floors": "Number of Floors",
    "waterfront": "Waterfront View",
    "view": "View Quality (0 = none, 4 = excellent)",
    "condition": "Condition Rating (1 = poor, 5 = excellent)",
    "house_age": "House Age (years)",
    "is_renovated": "Has Been Renovated",
    "living_lot_ratio": "Living-to-Lot Size Ratio",
    "total_rooms": "Total Rooms",
    "yr_built": "Year Built",
    "yr_renovated": "Year Renovated",
    "city": "City",
    "statezip": "State / Zip",
}

FEATURE_HELP = {
    "bedrooms": "Total number of bedrooms in the house.",
    "bathrooms": "Total number of bathrooms in the house.",
    "sqft_living": "Interior living space, in square feet.",
    "sqft_lot": "Total lot size, in square feet.",
    "sqft_above": "Square footage above ground level.",
    "sqft_basement": "Square footage of the basement, if any.",
    "floors": "Number of floors/levels in the house.",
    "waterfront": "Does the property have a waterfront view?",
    "view": "How good the view from the house is, on a 0–4 scale.",
    "condition": "Overall condition of the house, on a 1–5 scale.",
    "house_age": "Age of the house in years, at the time of sale.",
    "is_renovated": "Has the house been renovated at any point?",
    "living_lot_ratio": "Ratio of living area to total lot size.",
    "total_rooms": "Total number of rooms in the house.",
    "yr_built": "The year the house was originally built.",
    "yr_renovated": "The year the house was last renovated.",
    "city": "City where the property is located.",
    "statezip": "State and ZIP code of the property.",
}


def pretty_label(name: str) -> str:
    """Return a human-friendly label for a feature/prefix name."""
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


def help_text(name: str) -> str | None:
    """Return helper text for a feature/prefix name, if available."""
    return FEATURE_HELP.get(name)


# ---------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
)


# ---------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the trained Multiple Linear Regression model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Please make sure the trained model has been saved there."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading processed dataset...")
def load_processed_data():
    """Load the processed dataset used to build the input interface."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Processed dataset not found at '{DATA_PATH}'. "
            "This file is required to determine feature names, "
            "categories, and numeric ranges."
        )
    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------------------
# Feature-schema detection
# ---------------------------------------------------------------------
def build_feature_schema(df: pd.DataFrame):
    """
    Inspect the processed dataframe and build a schema describing how
    to render each model input feature and how to reconstruct the
    final prediction row.

    Returns a dict with:
        "feature_order": list[str]  -> exact column order the model expects
        "onehot_groups": dict[str, dict] -> {prefix: {"columns": [...], "options": [...]}}
        "grouped_columns": set[str] -> columns consumed by a one-hot group
        "binary_columns": set[str] -> standalone 0/1 flag columns
        "categorical_columns": dict[str, list] -> small-int columns -> sorted options
        "numeric_columns": list[str] -> continuous numeric columns
    """
    feature_order = [c for c in df.columns if c != TARGET_COLUMN]

    # --- Step 1: find columns that are strictly binary (0/1 only) ---
    binary_candidates = []
    for col in feature_order:
        unique_vals = set(pd.unique(df[col].dropna()))
        if unique_vals.issubset({0, 1}) and len(unique_vals) > 0:
            binary_candidates.append(col)

    # --- Step 2: group binary columns by shared prefix (before first "_") ---
    prefix_groups = {}
    for col in binary_candidates:
        if "_" not in col:
            continue
        prefix, suffix = col.split("_", 1)
        prefix_groups.setdefault(prefix, []).append(col)

    onehot_groups = {}
    grouped_columns = set()

    for prefix, cols in prefix_groups.items():
        if len(cols) < 2:
            continue  # not actually a multi-category group
        # Confirm true one-hot behaviour: each row sums to exactly 1
        # across the candidate columns (allowing rows where none apply).
        row_sums = df[cols].sum(axis=1)
        if row_sums.max() == 1:
            options = [c.split("_", 1)[1] for c in cols]
            onehot_groups[prefix] = {"columns": cols, "options": sorted(options)}
            grouped_columns.update(cols)

    # --- Step 3: standalone binary flags (not part of a one-hot group) ---
    binary_columns = set(binary_candidates) - grouped_columns

    # --- Step 4: remaining columns are numeric ---
    remaining = [
        c for c in feature_order if c not in grouped_columns and c not in binary_columns
    ]

    categorical_columns = {}
    numeric_columns = []

    for col in remaining:
        series = df[col].dropna()
        is_integer_like = np.all(np.equal(np.mod(series, 1), 0))
        n_unique = series.nunique()

        if is_integer_like and n_unique <= MAX_CATEGORICAL_UNIQUE:
            categorical_columns[col] = sorted(series.unique().tolist())
        else:
            numeric_columns.append(col)

    return {
        "feature_order": feature_order,
        "onehot_groups": onehot_groups,
        "grouped_columns": grouped_columns,
        "binary_columns": binary_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
    }


def nearest_value(options, target):
    """Return the option value closest to `target` (used for defaults)."""
    return min(options, key=lambda x: abs(x - target))


# ---------------------------------------------------------------------
# Load model & data (with friendly error handling)
# ---------------------------------------------------------------------
try:
    model = load_model()
except Exception as exc:  # noqa: BLE001
    st.error(f"❌ Could not load the trained model.\n\n**Details:** {exc}")
    st.stop()

try:
    processed_df = load_processed_data()
except Exception as exc:  # noqa: BLE001
    st.error(f"❌ Could not load the processed dataset.\n\n**Details:** {exc}")
    st.stop()

if TARGET_COLUMN not in processed_df.columns:
    st.error(
        f"❌ The processed dataset must contain a '{TARGET_COLUMN}' column "
        "so it can be excluded from the model features."
    )
    st.stop()

schema = build_feature_schema(processed_df)

# ---------------------------------------------------------------------
# Session state — used to reset the form after every prediction
# ---------------------------------------------------------------------
# `form_generation` is embedded in every widget's key. Bumping it forces
# Streamlit to treat every widget as brand new on the next run, which
# makes it fall back to its `value=`/`index=` default — i.e. a full
# form reset. `last_result` keeps the most recent prediction around so
# it can still be displayed right after the reset happens.
if "form_generation" not in st.session_state:
    st.session_state.form_generation = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = None

gen = st.session_state.form_generation

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🏠 House Price Prediction")
st.markdown(
    """
    This app uses a **Multiple Linear Regression** model, trained on
    historical housing data, to estimate a house's price from its
    characteristics. Fill in the details below and click
    **Predict Price** to get an instant estimate. The form resets to
    its defaults after every prediction so it's ready for the next one.
    """
)
st.divider()

# ---------------------------------------------------------------------
# Dynamic Input Form
# ---------------------------------------------------------------------
st.subheader("🔧 House Features")

user_values = {}  # col_name -> raw value entered/selected by the user

numeric_cols = schema["numeric_columns"]
categorical_cols = schema["categorical_columns"]
binary_cols = sorted(schema["binary_columns"])
onehot_groups = schema["onehot_groups"]

# --- Continuous numeric inputs ---
if numeric_cols:
    st.markdown("**Numeric Details**")
    cols_per_row = 3
    rows = [numeric_cols[i:i + cols_per_row] for i in range(0, len(numeric_cols), cols_per_row)]
    for row in rows:
        st_cols = st.columns(len(row))
        for st_col, feature in zip(st_cols, row):
            series = processed_df[feature].dropna()
            min_val = float(series.min())
            max_val = float(series.max())
            default_val = round(float(series.median()), 2)
            is_float = not np.all(np.equal(np.mod(series, 1), 0))
            step = round((max_val - min_val) / 100, 2) if is_float else 1.0
            step = step if step > 0 else 1.0

            with st_col:
                user_values[feature] = st.number_input(
                    pretty_label(feature),
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=step,
                    help=help_text(feature),
                    key=f"num__{feature}__{gen}",
                )

# --- Small integer categorical inputs ---
if categorical_cols:
    st.markdown("**Categorical Details**")
    cat_items = list(categorical_cols.items())
    cols_per_row = 3
    rows = [cat_items[i:i + cols_per_row] for i in range(0, len(cat_items), cols_per_row)]
    for row in rows:
        st_cols = st.columns(len(row))
        for st_col, (feature, options) in zip(st_cols, row):
            default_val = nearest_value(options, processed_df[feature].median())
            with st_col:
                user_values[feature] = st.selectbox(
                    pretty_label(feature),
                    options=options,
                    index=options.index(default_val),
                    help=help_text(feature),
                    key=f"cat__{feature}__{gen}",
                )

# --- Standalone binary (Yes/No) inputs ---
if binary_cols:
    st.markdown("**Yes / No Details**")
    cols_per_row = 3
    rows = [binary_cols[i:i + cols_per_row] for i in range(0, len(binary_cols), cols_per_row)]
    for row in rows:
        st_cols = st.columns(len(row))
        for st_col, feature in zip(st_cols, row):
            with st_col:
                choice = st.selectbox(
                    pretty_label(feature),
                    options=["No", "Yes"],
                    help=help_text(feature),
                    key=f"bin__{feature}__{gen}",
                )
                user_values[feature] = 1 if choice == "Yes" else 0

# --- One-hot encoded group inputs (e.g. City, StateZip) ---
if onehot_groups:
    st.markdown("**Location Details**")
    group_items = list(onehot_groups.items())
    cols_per_row = 3
    rows = [group_items[i:i + cols_per_row] for i in range(0, len(group_items), cols_per_row)]
    for row in rows:
        st_cols = st.columns(len(row))
        for st_col, (prefix, info) in zip(st_cols, row):
            with st_col:
                selected_option = st.selectbox(
                    pretty_label(prefix),
                    options=info["options"],
                    help=help_text(prefix),
                    key=f"onehot__{prefix}__{gen}",
                )
                for col in info["columns"]:
                    suffix = col.split("_", 1)[1]
                    user_values[col] = 1 if suffix == selected_option else 0

st.divider()
predict_button = st.button("🔍 Predict Price", use_container_width=True, type="primary")

# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
if predict_button:
    try:
        # Build the prediction row with EXACTLY the same columns, in the
        # same order, as the processed training dataset.
        input_row = {col: user_values[col] for col in schema["feature_order"]}
        input_df = pd.DataFrame([input_row], columns=schema["feature_order"])

        predicted_price = model.predict(input_df)[0]
        predicted_price = max(predicted_price, 0)  # prices should never be negative

        # Save the result so it can be shown after the form resets below.
        st.session_state.last_result = {
            "price": predicted_price,
            "input_df": input_df,
        }

        # Bump the widget generation so every input is recreated fresh
        # (back to its default value) on the next run, then rerun now.
        st.session_state.form_generation += 1
        st.rerun()

    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Something went wrong while generating the prediction.\n\n**Details:** {exc}")

elif st.session_state.last_result is not None:
    result = st.session_state.last_result
    st.subheader("📊 Prediction Result")
    result_col, _ = st.columns([1, 2])
    with result_col:
        st.metric("Estimated House Price", f"${result['price']:,.2f}")

    with st.expander("🔎 View model input row"):
        st.dataframe(result["input_df"], use_container_width=True, hide_index=True)
    
    st.balloons()

else:
    st.info("👆 Set the house details above, then click **Predict Price**.")

st.divider()
st.caption(
    "Built with Streamlit • Model: Multiple Linear Regression (Scikit-Learn) • "
    "For educational purposes only — predictions are estimates, not real appraisals."
)