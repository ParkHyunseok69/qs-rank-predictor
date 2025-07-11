import streamlit as st
import pandas as pd
import joblib
import model

# Load model and get metadata
pipe = joblib.load("uni_rank_predictor.pkl")
expected_cols = list(pipe.feature_names_in_)
qs_data = pd.read_csv("2026 QS World University Rankings.csv")

# UI Setup
st.set_page_config(page_title="QS Rank Predictor", layout="wide")
st.title("🎓 QS University Rank Predictor")
st.markdown("Predict your university's **QS Rank** using key metrics below.")


with st.expander("ℹ️ How to Use"):
    st.markdown(
        "- Fill in the metrics your university has.\n"
        "- Leave unknown ones at default (0 is safe).\n"
        "- Click **Predict Rank** to estimate your 2026 QS ranking.\n"
        "- The model is trained on official QS ranking data."
    )

with st.expander("ℹ️ Why These Features?"):
    st.markdown("These metrics were **automatically selected** by the model using feature importance.")
    st.markdown("They have the most predictive power based on historical QS data.")
    st.markdown("- Scores: academic, employer, research, sustainability\n- Ranks: previous year & category rankings")

# Input fields
metric_descriptions = {
    "Previous Rank": "QS Rank of the university in the previous year.",
    "AR SCORE": "Academic Reputation Score – global academic perception.",
    "ER SCORE": "Employer Reputation Score – graduate employability.",
    "FSR SCORE": "Faculty/Student Ratio Score – teaching capacity.",
    "FSR RANK": "Rank based on FSR score.",
    "CPF SCORE": "Citations per Faculty Score – research impact.",
    "CPF RANK": "Rank based on CPF score.",
    "IFR SCORE": "International Faculty Ratio Score – % of foreign faculty.",
    "IFR RANK": "Rank based on IFR score.",
    "ISR SCORE": "International Student Ratio Score – % of foreign students.",
    "IRN SCORE": "International Research Network Score – cross-border collaborations.",
    "EO SCORE": "Employment Outcomes Score – alumni career success.",
    "SUS SCORE": "Sustainability Score – environmental and social commitment.",
    "Overall SCORE": "Final composite score used for QS ranking."
}

st.subheader("🌍 University Context")
country_input = st.selectbox("Country/Territory", sorted(qs_data["Country/Territory"].dropna().unique()), index=qs_data["Country/Territory"].dropna().unique().tolist().index("Philippines"))
region_input = st.selectbox("Region", sorted(qs_data["Region"].dropna().unique()), index=qs_data["Region"].dropna().unique().tolist().index("Asia"))

user_inputs = {}
st.subheader("📈 Score Metrics")
score_fields = [
    "AR SCORE", "ER SCORE", "FSR SCORE", "CPF SCORE", "IFR SCORE",
    "ISR SCORE", "IRN SCORE", "EO SCORE", "SUS SCORE", "Overall SCORE"
]
for field in score_fields:
    val = st.number_input(
        f"{field}", min_value=0.0, max_value=1500.0, value=50.0,
        help=metric_descriptions.get(field)
    )
    hint = " 🟢 Good" if val > 80 else (" 🟡 Average" if val > 50 else " 🔴 Needs Improvement")
    st.caption(f"{field}{hint}")
    user_inputs[field] = val

st.subheader("🏅 Rank Metrics")
rank_fields = ["Previous Rank", "FSR RANK", "CPF RANK", "IFR RANK"]
for field in rank_fields:
    val = st.number_input(
        f"{field}", min_value=0, max_value=1500, value=400,
        help=metric_descriptions.get(field)
    )
    hint = " 🟢 Strong" if val < 200 else (" 🟡 Moderate" if val < 700 else " 🔴 Weak")
    st.caption(f"{field}{hint}")
    user_inputs[field] = val


input_df = pd.DataFrame([user_inputs])
for col in expected_cols:
    if col not in input_df.columns:
        if col.startswith("num__"):
            input_df[col] = 0
        else:
            input_df[col] = "Unknown"

with st.expander("📊 Benchmark Against Selected Country & Region"):
    for field in score_fields:
        if field in qs_data.columns:
            try:
                selected_country_data = qs_data[qs_data["Country/Territory"] == country_input][field].dropna()
                selected_region_data = qs_data[qs_data["Region"] == region_input][field].dropna()
                user_val = user_inputs[field]
                country_percentile = (selected_country_data < user_val).mean() * 100
                region_percentile = (selected_region_data < user_val).mean() * 100
                st.markdown(f"**{field}**: Top **{100 - int(country_percentile)}%** in s{country_input}, Top **{100 - int(region_percentile)}%** in 🌎 {region_input}")
            except:
                pass

st.subheader("💾 Save/Load Profile")
profile_name = st.text_input("Profile Name")
if st.button("💾 Save Current Inputs"):
    if profile_name:
        st.session_state[profile_name] = user_inputs.copy()
        st.info("Profile saved!")
    else:
        st.warning("Please enter a profile name before saving.")

saved_profiles = list(st.session_state.keys())
if saved_profiles:
    selected_profile = st.selectbox("Choose a saved profile to load", saved_profiles)
    if st.button("📂 Load Saved Inputs"):
        try:
            loaded_profile = st.session_state[selected_profile]
            for key in loaded_profile:
                user_inputs[key] = loaded_profile[key]
            st.success(f"Profile '{selected_profile}' ")
        except Exception as e:
            st.error(f"Could not load saved inputs: {e}")


# Prediction
if st.button("📊 Predict Rank"):
    try:
        prediction = pipe.predict(input_df)[0]
        st.success(f"🎯 Predicted QS Rank: **{int(prediction)}** ± {int(model.rmse)}")
    except Exception as e:
        st.error(f"❌ Error: {e}")


