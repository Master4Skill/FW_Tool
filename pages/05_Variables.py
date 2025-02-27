import streamlit as st
import json
from streamlit_extras.app_logo import add_logo
from streamlit_extras.stoggle import stoggle

st.set_page_config(
    page_title="Variable Settings",
    page_icon="⚙️",
)
# Inject custom CSS to move the Streamlit logo to the top
st.markdown("""
    <style>
        [data-testid="stSidebarNav"]::before {
            content: "";
            display: block;
            margin: 20px auto;
            height: 50px;
            width: 20px;
            background-image: url('resized_image.png');
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
        }
    </style>
    """, unsafe_allow_html=True)

add_logo("resized_image.png")

st.sidebar.header("Variable Settings")

st.sidebar.info("Adjust the simulation parameters")

st.markdown("# Parameters")


with open("results/variables.json", "r") as f:
    input_data = json.load(f)
# input_data = {}

# Using Streamlit's number_input function to let user input the constants
input_data["λD"] = st.number_input(
    "Pipe isolation thermal conductivity (W/m·K)", value=0.034
)
input_data["λB"] = st.number_input("Soil thermal conductivity (W/m·K)", value=1.5)
input_data["rM"] = st.number_input("Pipe outer diameter (m)", value=0.26)
input_data["rR"] = st.number_input("Pipe inner diameter (m)", value=0.22)
input_data["a"] = st.number_input(
    "Spacing between flow and return pipes (m)", value=0.3
)
input_data["ζ"] = st.number_input(
    "Network resistance coefficient", value=150
)  # Units needed
input_data["l_Netz"] = st.number_input("Network length in meters (m)", value=5000)
input_data["ηPump"] = st.number_input(
    "Efficiency of pressure maintenance pump (%)", value=0.7
)
input_data["p_network"] = st.number_input(
    "Pressure in the network, only for pressurized networks with temperatures above 100°C (Pa)",
    value=101325,
)


expander = st.expander("Additional Parameters")

with expander:
    input_data["hÜ"] = st.number_input(
        "Minimum cover height (m)", value=(1 - input_data["rM"])
    )
    input_data["ρ_water"] = st.number_input("Water density (kg/m³)", value=980)
    input_data["cp_water"] = (
        st.number_input("Specific heat capacity of water (J/kg·K)", value=4184)
        / 3600000
    )
    # only relevant for solarthermal electricity consumption
    # input_data["ρ_glycol_water"] = st.number_input(
    #    "Densitiy Glykol-Water Mixture in solarthermal units (kg/m³)", value=1025
    # )

    input_data["ηWüHüs"] = st.number_input(
        "Efficiency of heat exchanger at home transfer station (%)", value=0.95
    )
    input_data["ηWüE"] = st.number_input(
        "Efficiency of heat exchanger at producer-network (%)", value=0.95
    )
    input_data["ηVerdichter"] = st.number_input(
        "Efficiency of compressor(%)", value=0.85
    )
    input_data["T_q_diffmax"] = st.number_input(
        "Heat pump: maximum temperature difference of the heat source(°K)", value=5
    )
    input_data["T_Wü_delta_r"] = st.number_input(
        "minimal residual temperature delta at return side of heat exchanger (°K)",
        value=2,
    )
    input_data["T_Wü_delta_f"] = st.number_input(
        "minimal temperature delta at flow side of heat exchanger (°K)", value=5
    )
    input_data["p_WP_loss"] = 1 - st.number_input(
        "Pressure losses heat pump (%)", value=0.05
    )
    input_data["ηSpitzenkessel"] = st.number_input(
        "Peak Load Boiler efficiency (%)", value=0.8
    )
    input_data["ηBHKW_el"] = st.number_input(
        "Electrical efficiency CHP (%)", value=0.35
    )
    input_data["ηBHKW_therm"] = st.number_input(
        "Thermal efficiency CHP (%)", value=0.55
    )

if st.button("Save"):
    with open("results/variables.json", "w") as f:
        json.dump(input_data, f)
    st.sidebar.success("Data saved successfully.")

st.image(
    "variables.png",
    use_column_width=True,
    caption="Variables for the Pipe Dimensioning",
)
