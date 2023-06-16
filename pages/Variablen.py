import streamlit as st
import json
from streamlit_extras.app_logo import add_logo
from streamlit_extras.stoggle import stoggle

st.set_page_config(
    page_title="Variablen Einstellungen",
    page_icon="⚙️",
)
add_logo("resized_image.png")

st.sidebar.header("Einstellung der Variablen")

st.sidebar.info("Passen Sie die Simulationsparameter an")

st.markdown("# Parameter")

with open("data.json", "r") as f:
    input_data = json.load(f)

# Using Streamlit's number_input function to let user input the constants
input_data["λD"] = st.number_input("Dicke der Dämmung (m)", value=0.034)
input_data["λB"] = st.number_input("Wärmeleitfähigkeit Boden (W/m·K)", value=1.2)
input_data["rM"] = st.number_input("Außendurchmesser Rohr (m)", value=0.26)
input_data["rR"] = st.number_input("Innendurchmesser Rohr (m)", value=0.22)
input_data["a"] = st.number_input("Verlegungstiefe (m)", value=1.0)
input_data["ζ"] = st.number_input(
    "Summe Einzelwiderstände im Netz", value=150
)  # Units needed
input_data["l_Netz"] = st.number_input("Netzlänge in Metern (m)", value=5000)
input_data["ηPump"] = st.number_input("Wirkungsgrad Druckerhaltungpumpe (%)", value=0.7)


expander = st.expander("weitere Parameter")

with expander:
    input_data["hÜ"] = st.number_input(
        "minimale Überdeckungshöhe (m)", value=(1 - input_data["rM"])
    )
    input_data["ρ_water"] = st.number_input("Wasserdichte (g/cm³)", value=0.98)
    input_data["cp_water"] = st.number_input(
        "Spezifische Wärmekapazität von Wasser (Wh/kg·K)", value=1.162
    )
    input_data["ηWüHüs"] = st.number_input("Wirkungsgrad WÜ HÜS (%)", value=0.95)
    input_data["ηWüE"] = st.number_input(
        "Wirkungsgrad WÜ Erzeugungsanlage-Netz (%)", value=0.95
    )
    input_data["ηVerdichter"] = st.number_input(
        "Wirkungsgrad Verdichter Wärmepumpen (%)", value=0.85
    )
    input_data["p_WP_loss"] = st.number_input(
        "Druckverluste Wärmepumpe (%)", value=0.95
    )
    input_data["ηSpitzenkessel"] = st.number_input(
        "Wirkungsgrad Spitzenlastkessel (%)", value=0.92
    )
    input_data["ηBHKW_el"] = st.number_input(
        "elektrischer Wirkungsgrad BHKW (%)", value=0.35
    )
    input_data["ηBHKW_therm"] = st.number_input(
        "thermischer Wirkungsgrad BHKW (%)", value=0.55
    )

# Saving the input data to a json file
with open("data.json", "w") as f:
    json.dump(input_data, f)

if st.button("Speichern"):
    with open("data.json", "w") as f:
        json.dump(input_data, f)
    st.sidebar.success("Data saved successfully.")
