import streamlit as st
import numpy as np
import pandas as pd


def replace_comma(value):
    return float(value.replace(".", "").replace(",", "."))


# Read the CSV file
demand_df = pd.read_csv(
    "Input_Netz.csv",
    converters={
        "Lastgang": replace_comma,
        "Bodentemp": replace_comma,
        "Lufttemp": replace_comma,
    },
)

# Print the DataFrame to verify it was read in correctly
print(demand_df)


"""from CoolProp.CoolProp import PropsSI


def water_viscosity_CoolProp(T):
    
    #Calculate water viscosity given temperature in Celsius using CoolProp.
    #T : temperature (degrees Celsius)
    
    T_K = T + 273.15  # Convert temperature to Kelvin
    P = 101325  # Assume atmospheric pressure in Pa

    # Get viscosity (in Pa.s)
    viscosity = PropsSI("V", "P", P, "T", T_K, "Water")

    return viscosity


def test_water_viscosity():
    
    #Test the water_viscosity_CoolProp function.
    
    for T in range(20, 101, 5):  # Temperatures from 20°C to 100°C
        viscosity = water_viscosity_CoolProp(T)
        print(f"Temperature: {T}°C, Dynamic viscosity: {viscosity} Pa.s")


test_water_viscosity()


import pandas as pd
import streamlit as st

# Initialize an empty DataFrame
df = pd.DataFrame(
    columns=["Erzeuger", "Volumenstrom_quelle", "Quelltemperatur", "Leistung_max"]
)

# Define the Erzeuger options
erzeuger_options = [
    "Wärmepumpe",
    "Geothermie",
    "Solarthermie",
    "Spitzenlastkessel",
    "BHKW",
]

# Ask the user for the number of Erzeuger
num_erzeuger = st.number_input(
    "Enter the number of Erzeuger", min_value=1, max_value=5, value=1
)

# For each Erzeuger, get the necessary information
for i in range(num_erzeuger):
    st.subheader(f"Erzeuger {i+1}")
    erzeuger = st.selectbox(f"Erzeuger {i+1} Type", erzeuger_options)
    Volumenstrom_quelle = st.number_input(f"Erzeuger {i+1} Volumenstrom_quelle")
    Quelltemperatur = st.number_input(f"Erzeuger {i+1} Quelltemperatur")
    Leistung_max = st.number_input(f"Erzeuger {i+1} Leistung_max")

    # Append the new row to the DataFrame
    df = df.append(
        {
            "Erzeuger": erzeuger,
            "Volumenstrom_quelle": Volumenstrom_quelle,
            "Quelltemperatur": Quelltemperatur,
            "Leistung_max": Leistung_max,
        },
        ignore_index=True,
    )

# Display the DataFrame
st.dataframe(df)



# Assuming these constants already exist, can easily become Userinputs
λD = 0.034  # Dicke der Dämmung
λB = 1.2  # Wärmeleitfähigkeit Boden
rM = 0.26  # Außendurchmesser Rohr
rR = 0.22  # Innendurchmesser Rohr
hÜ = 1 - rM  # minimale Überdeckungshöhe
a = 1.0  # Verlegungstiefe
ζ = 150  # Summe Einzelwiderstände im Netz (Einheit!?)
l_Netz = 5000  # Netzlänge in Metern
ηPump = 0.7  # Wirkungsgrad Druckerhaltungpumpe
ρ_water = 0.98
cp_water = 1.162

ηWüHüs = 0.95  # Wirkungsgrad WÜ HÜS
ηWüE = 0.95  # Wirkungsgrad WÜ Erzeugungsanlage-Netz


def calc_verlust(T_vl, T_b):
    term1 = 4 * np.pi * ((T_vl + Trl_vor) / 2 - T_b)
    term2 = (1 / λD) * np.log(rM / rR)
    term3 = (1 / λB) * np.log(4 * (hÜ + rM) / rM)
    term4 = (1 / λB) * np.log(((2 * (hÜ + rM) / a + 2 * rM) ** 2 + 1) ** 0.5)
    print(f"term1: {term1}, term2: {term2}, term3: {term3}, term4: {term4}")
    return l_Netz / 1000 * term1 / (term2 + term3 + term4)


T_vl_example = 88  # Replace with the actual value
T_b_example = 5.5  # Replace with the actual value
Trl_vor = 60

# Call the function with your parameters
result = calc_verlust(T_vl_example, T_b_example)

# Print the result
print(f"The calculated Netzverluste is: {result}")
"""

    with st.container():
        st.header("Erzeugungsgang")
        st.subheader("vor")
        sorted_df_vor = plot_actual_production(
            df_input, actual_production_df_vor, color_FFE, "Erzeugungsgang vor"
        )
        st.subheader("nach")
        sorted_df_nach = plot_actual_production(
            df_input, actual_production_df_nach, color_FFE, "Erzeugungsgang nach"
        )

    # Create the second container
    with st.container():
        st.header("Jahresdauerlinie")
        st.subheader("vor")
        plot_df_vor = plot_sorted_production(
            df_input,
            sorted_df_vor,
            actual_production_df_vor,
            color_FFE,
            "Jahresdauerlinie vor",
        )
        st.subheader("nach")
        plot_df_nach = plot_sorted_production(
            df_input,
            sorted_df_nach,
            actual_production_df_nach,
            color_FFE,
            "Jahresdauerlinie nach",
        )