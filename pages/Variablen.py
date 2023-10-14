import streamlit as st
import json
from streamlit_extras.app_logo import add_logo
from streamlit_extras.stoggle import stoggle

st.set_page_config(
    page_title="Variable Settings",
    page_icon="⚙️",
)
add_logo("resized_image.png")

st.sidebar.header("Variable Settings")

st.sidebar.info("Adjust the simulation parameters")

st.markdown("# Parameters")

with open("results/data.json", "r") as f:
    input_data = json.load(f)

# Using Streamlit's number_input function to let user input the constants
input_data["λD"] = st.number_input("Insulation Thickness (m)", value=0.034)
input_data["λB"] = st.number_input("Soil Thermal Conductivity (W/m·K)", value=1.2)
input_data["rM"] = st.number_input("Pipe Outer Diameter (m)", value=0.26)
input_data["rR"] = st.number_input("Pipe Inner Diameter (m)", value=0.22)
input_data["a"] = st.number_input("Laying Depth (m)", value=1.0)
input_data["ζ"] = st.number_input(
    "Sum of Individual Resistances in the Network", value=150
)  # Units needed
input_data["l_Netz"] = st.number_input("Network Length in Meters (m)", value=5000)
input_data["ηPump"] = st.number_input(
    "Efficiency of Pressure Maintenance Pump (%)", value=0.7
)
input_data["p_network"] = st.number_input(
    "Pressure in the Network, only for pressurized networks with temperatures above 100°C (Pa)",
    value=101325,
)


expander = st.expander("Additional Parameters")

with expander:
    input_data["hÜ"] = st.number_input(
        "Minimum Cover Height (m)", value=(1 - input_data["rM"])
    )
    input_data["ρ_water"] = st.number_input("Water Density (kg/m³)", value=980)
    input_data["cp_water"] = (
        st.number_input("Specific Heat Capacity of Water (J/kg·K)", value=4184)
        / 3600000
    )
    input_data["ρ_glycol_water"] = st.number_input(
        "Densitiy Glykol-Water Mixture in solarthermal units (kg/m³)", value=1025
    )

    input_data["ηWüHüs"] = st.number_input(
        "Efficiency of Heat Exchanger at Home Transfer Station (%)", value=0.95
    )
    input_data["ηWüE"] = st.number_input(
        "Efficiency of Heat Exchanger at Power Plant-Network (%)", value=0.95
    )
    input_data["ηVerdichter"] = st.number_input(
        "Efficiency of Compressor Heat Pumps (%)", value=0.85
    )
    input_data["T_q_diffmax"] = st.number_input(
        "Heat pump: maximum Temperature Difference of the Heat Source(°K)", value=5
    )
    input_data["T_Wü_delta_r"] = st.number_input(
        "minimal residual Temperature delta at return side of heat Exchanger (°K)",
        value=2,
    )
    input_data["T_Wü_delta_f"] = st.number_input(
        "minimal Temperature delta at flow side of heat Exchanger (°K)", value=5
    )
    input_data["p_WP_loss"] = st.number_input(
        "Pressure Losses Heat Pump (%)", value=0.95
    )
    input_data["ηSpitzenkessel"] = st.number_input(
        "Peak Load Boiler Efficiency (%)", value=0.92
    )
    input_data["ηBHKW_el"] = st.number_input(
        "Electrical Efficiency CHP (%)", value=0.35
    )
    input_data["ηBHKW_therm"] = st.number_input(
        "Thermal Efficiency CHP (%)", value=0.55
    )

# Saving the input data to a json file
# with open("results/data.json", "w") as f:
#    json.dump(input_data, f)

if st.button("Save"):
    with open("results/data.json", "w") as f:
        json.dump(input_data, f)
    st.sidebar.success("Data saved successfully.")
