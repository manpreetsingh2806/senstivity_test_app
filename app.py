"""
Sanctions Sensitivity Test Pack Creation Agent
------------------------------------------------
Streamlit app that:
1. Ingests an OFAC SDN.csv file
2. Samples names from it
3. For each name, asks Claude (Anthropic API) which name-manipulation
   scenarios structurally apply, and to generate the manipulated test name
   for each applicable scenario
4. Assembles a downloadable sensitivity test pack (CSV / Excel)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    - Push app.py + requirements.txt to a GitHub repo
    - Create a new app on https://share.streamlit.io pointing at app.py
    - No secrets need to be pre-configured — users paste their own
      Anthropic API key into the sidebar each session.
"""

import io
import json
import re
import time

import pandas as pd
import streamlit as st
from anthropic import Anthropic

# --------------------------------------------------------------------------
# Scenario definitions
# --------------------------------------------------------------------------
# Scenario 1 (Exact match) is handled deterministically in code — it always
# applies and needs no LLM judgement. Scenarios 2-6 are judged + generated
# by the LLM, one name at a time, so new scenarios can be appended here
# without touching the rest of the app.

SCENARIOS = {
    1: {
        "name": "Exact match",
        "description": "No manipulation — the test name is identical to the original name.",
    },
    2: {
        "name": "Middle name removal",
        "description": (
            "Remove the middle name/initial from an individual's name. "
            "Only applies if a middle name/initial is present (3+ name tokens for a person). "
            "E.g. 'Peter A Smith' -> 'Peter Smith'."
        ),
    },
    3: {
        "name": "First and last name interchanged",
        "description": (
            "Swap the first and last name. Only applies to individual names with a clear "
            "first and last name — not single-word names or organization names. "
            "E.g. 'Peter Smith' -> 'Smith Peter'."
        ),
    },
    4: {
        "name": "First name part addition",
        "description": (
            "Prepend a plausible, unrelated common first name to the existing name. "
            "E.g. 'Flipos Wander' -> 'John Flipos Wander'."
        ),
    },
    5: {
        "name": "Prefix/title addition",
        "description": (
            "Add an appropriate title/prefix (Mr, Dr, Capt, etc.) before an individual's name. "
            "Not applicable to organizations, vessels, or aircraft. "
            "E.g. 'Dmitri Kovtlin' -> 'Dr Dmitri Kovtlin'."
        ),
    },
    6: {
        "name": "Common word addition at end",
        "description": (
            "Append a plausible common organizational/descriptive word (e.g. Organization, "
            "Group, Trading, Foundation, Company) to the end of an entity name. Primarily "
            "applies to organization/entity names, not individual person names. "
            "E.g. 'Armed Forces Geographical' -> 'Armed Forces Geographical Organization'."
        ),
    },
}

LLM_SCENARIO_IDS = [sid for sid in SCENARIOS if sid != 1]

MODEL = "claude-sonnet-5"

# --------------------------------------------------------------------------
# OFAC SDN.csv loading
# --------------------------------------------------------------------------

OFAC_STANDARD_COLUMNS = [
    "ent_num", "SDN_Name", "SDN_Type", "Program", "Title", "Call_Sign",
    "Vess_type", "Tonnage", "GRT", "Vess_flag", "Vess_owner", "Remarks",
]


def load_ofac_csv(uploaded_file) -> pd.DataFrame:
    """Load an OFAC SDN.csv, whether it has a header row or the raw
    headerless government feed format."""
    raw = uploaded_file.getvalue()

    # First try: does it look like it already has a header?
    probe = pd.read_csv(io.BytesIO(raw), nrows=5, header=0)
    probe_cols_lower = [str(c).strip().lower() for c in probe.columns]
    if any("sdn_name" in c or c == "name" for c in probe_cols_lower):
        df = pd.read_csv(io.BytesIO(raw), header=0, dtype=str)
        # normalize the name column to SDN_Name
        for c in list(df.columns):
            if str(c).strip().lower() in ("sdn_name", "name"):
                df = df.rename(columns={c: "SDN_Name"})
        for c in list(df.columns):
            if str(c).strip().lower() == "sdn_type":
                df = df.rename(columns={c: "SDN_Type"})
        for c in list(df.columns):
            if str(c).strip().lower() == "program":
                df = df.rename(columns={c: "Program"})
        return df

    # Otherwise: assume the raw headerless OFAC feed layout
    df = pd.read_csv(io.BytesIO(raw), header=None, dtype=str)
    ncols = df.shape[1]
    if ncols <= len(OFAC_STANDARD_COLUMNS):
        cols = OFAC_STANDARD_COLUMNS[:ncols]
    else:
        cols = OFAC_STANDARD_COLUMNS + [f"col_{i}" for i in range(len(OFAC_STANDARD_COLUMNS), ncols)]
    df.columns = cols
    return df


# --------------------------------------------------------------------------
# LLM calls
# --------------------------------------------------------------------------

def build_prompt(name: str, entity_type: str, program: str) -> str:
    scenario_block = "\n".join(
        f"{sid}. {SCENARIOS[sid]['name']} — {SCENARIOS[sid]['description']}"
        for sid in LLM_SCENARIO_IDS
    )
    return f"""You are assisting a bank's sanctions screening QA team in building a
name-sensitivity test pack. For the sanctioned name below, evaluate each listed
manipulation scenario and decide whether it structurally applies to this specific
name (based on the name's structure — number of tokens, whether it looks like an
individual vs an organization, etc). If a scenario applies, generate the manipulated
test name exactly as the scenario describes, realistically. If it does not apply,
set "applicable" to false and "test_name" to null.

Name: {name}
Entity type (from sanctions list): {entity_type or "unknown"}
Program: {program or "unknown"}

Scenarios:
{scenario_block}

Return ONLY valid JSON, no markdown code fences, no commentary, in exactly this shape:
{{
  "scenarios": [
    {{"scenario_id": 2, "applicable": true, "test_name": "string or null", "rationale": "short reason"}},
    {{"scenario_id": 3, "applicable": true, "test_name": "string or null", "rationale": "short reason"}},
    {{"scenario_id": 4, "applicable": true, "test_name": "string or null", "rationale": "short reason"}},
    {{"scenario_id": 5, "applicable": true, "test_name": "string or null", "rationale": "short reason"}},
    {{"scenario_id": 6, "applicable": true, "test_name": "string or null", "rationale": "short reason"}}
  ]
}}"""


def parse_llm_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    return json.loads(cleaned)


def get_scenarios_for_name(client: Anthropic, name: str, entity_type: str, program: str) -> list:
    prompt = build_prompt(name, entity_type, program)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    data = parse_llm_json(text)
    return data.get("scenarios", [])


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Sanctions Sensitivity Test Pack", layout="wide")
st.title("Sanctions Sensitivity Test Pack Creation Agent")
st.caption(
    "Upload an OFAC SDN.csv, sample names, and let Claude decide which manipulation "
    "scenarios apply to each name and generate the resulting test names."
)

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Anthropic API key", type="password", help="Not stored — used only for this session.")
    st.markdown("---")
    st.subheader("Scenarios")
    for sid, s in SCENARIOS.items():
        st.markdown(f"**{sid}. {s['name']}**")
        st.caption(s["description"])

uploaded = st.file_uploader("Upload OFAC SDN.csv", type=["csv"])

if uploaded is not None:
    try:
        df = load_ofac_csv(uploaded)
    except Exception as e:
        st.error(f"Could not parse this file as an OFAC SDN.csv: {e}")
        st.stop()

    if "SDN_Name" not in df.columns:
        st.error("Couldn't find a name column (expected 'SDN_Name' or 'Name') in this file.")
        st.stop()

    st.success(f"Loaded {len(df):,} records.")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        type_options = sorted(df["SDN_Type"].dropna().unique().tolist()) if "SDN_Type" in df.columns else []
        selected_types = st.multiselect(
            "Filter by entity type (optional)", options=type_options, default=type_options
        )
    with col2:
        sample_size = st.number_input(
            "Number of names to sample", min_value=1, max_value=200, value=10, step=1
        )

    filtered = df.copy()
    if selected_types and "SDN_Type" in df.columns:
        filtered = filtered[filtered["SDN_Type"].isin(selected_types)]

    filtered = filtered.dropna(subset=["SDN_Name"])
    sample_n = min(int(sample_size), len(filtered))
    sample_df = filtered.sample(n=sample_n, random_state=None) if sample_n > 0 else filtered

    run = st.button("Generate test pack", type="primary", disabled=(not api_key or sample_n == 0))
    if not api_key:
        st.info("Enter your Anthropic API key in the sidebar to run.")

    if run:
        client = Anthropic(api_key=api_key)
        rows = []
        progress = st.progress(0.0)
        status = st.empty()
        errors = []

        names = sample_df.to_dict("records")
        for i, rec in enumerate(names):
            name = str(rec.get("SDN_Name", "")).strip()
            entity_type = str(rec.get("SDN_Type", "")).strip()
            program = str(rec.get("Program", "")).strip()
            status.text(f"Processing {i + 1}/{len(names)}: {name}")

            # Scenario 1 — Exact match — always applicable, no LLM call needed
            rows.append({
                "Original Name": name,
                "Entity Type": entity_type,
                "Program": program,
                "Scenario ID": 1,
                "Scenario": SCENARIOS[1]["name"],
                "Applicable": True,
                "Test Name": name,
                "Rationale": "Baseline exact match — always applicable.",
            })

            try:
                scenario_results = get_scenarios_for_name(client, name, entity_type, program)
                for sr in scenario_results:
                    sid = sr.get("scenario_id")
                    if sid not in SCENARIOS:
                        continue
                    rows.append({
                        "Original Name": name,
                        "Entity Type": entity_type,
                        "Program": program,
                        "Scenario ID": sid,
                        "Scenario": SCENARIOS[sid]["name"],
                        "Applicable": bool(sr.get("applicable")),
                        "Test Name": sr.get("test_name") if sr.get("applicable") else None,
                        "Rationale": sr.get("rationale", ""),
                    })
            except Exception as e:
                errors.append(f"{name}: {e}")

            progress.progress((i + 1) / len(names))
            time.sleep(0.05)

        status.empty()
        progress.empty()

        if errors:
            with st.expander(f"{len(errors)} name(s) had errors"):
                for e in errors:
                    st.text(e)

        result_df = pd.DataFrame(rows)
        st.session_state["result_df"] = result_df

if "result_df" in st.session_state:
    result_df = st.session_state["result_df"]
    st.markdown("---")
    st.subheader("Test pack")

    applicable_only = st.checkbox("Show only applicable scenarios", value=True)
    display_df = result_df[result_df["Applicable"]] if applicable_only else result_df
    st.dataframe(display_df, use_container_width=True)

    st.markdown(f"**Total test cases generated:** {len(display_df)}")

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV", data=csv_bytes, file_name="sensitivity_test_pack.csv", mime="text/csv"
    )

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Test Pack")
    st.download_button(
        "Download as Excel",
        data=excel_buffer.getvalue(),
        file_name="sensitivity_test_pack.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
