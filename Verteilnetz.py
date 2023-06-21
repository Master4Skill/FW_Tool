import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from CoolProp.CoolProp import PropsSI
from streamlit_extras.app_logo import add_logo
from PIL import Image
import sys
import json

# Load the data from the json file
with open("results/data.json", "r") as f:
    input_data = json.load(f)

st.set_page_config(page_title="Verteilnetz", page_icon=":house:")

st.markdown("# Fernwärmenetzsimulation")

st.sidebar.header("Temperaturabsenkung im Fernwärmenetz")

st.sidebar.info("Geben Sie die gewünschten Temperaturen ein")

with Image.open("Logo-dunkelblau-1000-px.webp") as img:
    # Resize the image
    width = 180
    wpercent = width / float(img.size[0])
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize(
        (width, hsize), Image.Resampling.LANCZOS
    )  # Replace ANTIALIAS with Resampling.LANCZOS
    # Save it back to the file
    img.save("resized_image.png")

# Now use this resized image as the logo
add_logo("resized_image.png")

# Load the CSV data
df_input = pd.read_csv("Input_Netz.csv", delimiter=",", decimal=",")
df_input.columns = df_input.columns.str.strip()
# st.write(df_input.columns)
# Convert 'Zeit' to numeric (just to be safe)
df_input["Zeit"] = pd.to_numeric(df_input["Zeit"], errors="coerce")

df_input["Lastgang"] = (
    df_input["Lastgang"]
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# Sort DataFrame by 'Zeit'
df_input = df_input.sort_values(by="Zeit")

df_input.to_json("results/Input_Netz.json", orient="columns")

# get user Input
input_data["Tvl_max_vor"] = st.number_input(
    "maximale Vorlauftemperatur vor Temperaturabsenkung", value=95
)
input_data["Tvl_min_vor"] = st.number_input(
    "minimale Vorlauftemperatur vor Temperaturabsenkung", value=85
)
input_data["Trl_vor"] = st.number_input(
    "Rücklauftemperatur vor Temperautrabsenkung", value=60
)
input_data["Tvl_max_nach"] = st.number_input(
    "maximale Vorlauftemperatur nach Temperaturabsenkung", value=75
)
input_data["Tvl_min_nach"] = st.number_input(
    "minimale Vorlauftemperatur nach Temperaturabsenkung", value=65
)
input_data["Trl_nach"] = st.number_input(
    "Rücklauftemperatur nach Temperautrabsenkung", value=40
)

with open("results/data.json", "w") as f:
    json.dump(input_data, f)

if st.button("Speichern"):
    with open("results/data.json", "w") as f:
        json.dump(input_data, f)
    st.sidebar.success("Data saved successfully.")


# Assuming these constants already exist, can easily become Userinputs
λD = input_data["λD"]
λB = input_data["λB"]
rM = input_data["rM"]
rR = input_data["rR"]
hÜ = input_data["hÜ"]
a = input_data["a"]
ζ = input_data["ζ"]
l_Netz = input_data["l_Netz"]
ηPump = input_data["ηPump"]
ρ_water = input_data["ρ_water"]
cp_water = input_data["cp_water"]
ηWüHüs = input_data["ηWüHüs"]
ηWüE = input_data["ηWüE"]

Tvl_max_vor = input_data["Tvl_max_vor"]
Tvl_min_vor = input_data["Tvl_min_vor"]
Trl_vor = input_data["Trl_vor"]
Tvl_max_nach = input_data["Tvl_max_nach"]
Tvl_min_nach = input_data["Tvl_min_nach"]
Trl_nach = input_data["Trl_nach"]

calculate_button = st.button("Calculate")

# Create a placeholder
placeholder = st.empty()
# Check if the button is clicked
# st.dataframe(df_input)
if not calculate_button:
    # Do calculation here
    placeholder.write("Please click the calculate button to proceed")
else:
    placeholder.write("Button is clicked. Calculating...")

    # Create the new DataFrame
    df_results = pd.DataFrame()
    df_results["hours"] = df_input["Zeit"]

    # calculate T_VL
    # Calculate the 24-hour moving average of 'Lufttemp' and add it to df_results
    df_results["Air_average"] = df_input["Lufttemp"].rolling(window=24).mean()

    # Definition of T_vl, right now just a linear fkt similar to Grubers Excel tool
    def calculate_T_vl_vor(air_average):
        return Tvl_max_vor - (1 / 15) * air_average * (
            Tvl_max_vor - Tvl_min_vor
        )  # Placeholder function

    def calculate_T_vl_nach(air_average):
        return Tvl_max_nach - (1 / 15) * air_average * (
            Tvl_max_nach - Tvl_min_nach
        )  # Placeholder function

    # Apply the function to calculate 'T_vl'
    df_results["T_vl_vor"] = df_results["Air_average"].apply(calculate_T_vl_vor)
    df_results["T_vl_nach"] = df_results["Air_average"].apply(calculate_T_vl_nach)

    # Calculate Netzverluste
    def calc_verlust(T_vl, T_b):
        term1 = 4 * np.pi * ((T_vl + Trl_vor) / 2 - T_b)
        term2 = (1 / λD) * np.log(rM / rR)
        term3 = (1 / λB) * np.log(4 * (hÜ + rM) / rM)
        term4 = (1 / λB) * np.log(((2 * (hÜ + rM) / a + 2 * rM) ** 2 + 1) ** 0.5)
        return l_Netz / 1000 * term1 / (term2 + term3 + term4)

    # Calculate Wärmelast
    def calc_totalLast(Netzverluste, Lastgang):
        return (Lastgang + Netzverluste) / (ηWüHüs * ηWüE)

    def calc_VerlustProzentual(Netzverluste, Wärmelast):
        return Netzverluste / Wärmelast

    # Calculate Volumenstrom und Strömungsgeschwindigkeit
    def calc_flowRate(T_vl, Wärmelast):
        return Wärmelast / (ρ_water * cp_water * (T_vl - Trl_nach))

    def calc_flowSpeed(flowRate):
        return (flowRate / 3600) / (((np.pi) * rR**2) / 4)

    # calculate Pumpleistung

    def water_viscosity_CoolProp(T):
        """
        Calculate water viscosity given temperature in Celsius using CoolProp.
        T : temperature (degrees Celsius)
        """
        if pd.isna(T) or T < 0:  # add this check
            return 0  # or some other default value

        T_K = T + 273.15  # Convert temperature to Kelvin
        P = 101325  # Assume atmospheric pressure in Pa

        # Get viscosity (in Pa.s)
        viscosity = PropsSI("V", "P", P, "T", T_K, "Water")

        return viscosity

    def calc_Reynolds(flowSpeed, T):
        return flowSpeed * rR * ρ_water / water_viscosity_CoolProp(T)

    def calc_pressureloss(Reynolds, flowSpeed):
        λ = 64 / Reynolds if Reynolds < 2300 else 0.3164 / Reynolds**0.25
        return λ * ρ_water / 2 * (flowSpeed**2) / rR * (l_Netz + ζ * rR / λ)

    def calc_pumpleistung(flowSpeed, flowRate, Tvl, Trl):
        Rey_vl = calc_Reynolds(flowSpeed, Tvl)
        Rey_rl = calc_Reynolds(flowSpeed, Trl)
        p_loss_vl = calc_pressureloss(Rey_vl, flowSpeed)
        p_loss_rl = calc_pressureloss(Rey_rl, flowSpeed)
        return (p_loss_vl + p_loss_rl) * flowRate / ηPump

    # create new Netzverluste column
    df_results["Netzverluste_vor"] = df_results.apply(
        lambda row: calc_verlust(row["T_vl_vor"], df_input["Bodentemp"][row.name]),
        axis=1,
    )

    df_results["Netzverluste_nach"] = df_results.apply(
        lambda row: calc_verlust(row["T_vl_nach"], df_input["Bodentemp"][row.name]),
        axis=1,
    )
    # create new Wärmelast column
    df_results["Wärmelast_vor"] = df_results.apply(
        lambda row: calc_totalLast(
            row["Netzverluste_vor"], df_input["Lastgang"][row.name]
        ),
        axis=1,
    )

    df_results["Wärmelast_nach"] = df_results.apply(
        lambda row: calc_totalLast(
            row["Netzverluste_nach"], df_input["Lastgang"][row.name]
        ),
        axis=1,
    )

    # create new Verlust% column
    df_results["VerlustProzentual_vor"] = df_results.apply(
        lambda row: calc_VerlustProzentual(
            row["Netzverluste_vor"], row["Wärmelast_vor"]
        ),
        axis=1,
    )
    df_results["VerlustProzentual_nach"] = df_results.apply(
        lambda row: calc_VerlustProzentual(
            row["Netzverluste_nach"], row["Wärmelast_nach"]
        ),
        axis=1,
    )

    # create new Volumenstrom und Strömungsgeschwindigkeit column
    df_results["Volumenstrom_vor"] = df_results.apply(
        lambda row: calc_flowRate(row["T_vl_vor"], row["Wärmelast_vor"]),
        axis=1,
    )
    df_results["Volumenstrom_nach"] = df_results.apply(
        lambda row: calc_flowRate(row["T_vl_nach"], row["Wärmelast_nach"]),
        axis=1,
    )
    df_results["Strömungsgeschwindigkeit_vor"] = df_results.apply(
        lambda row: calc_flowSpeed(row["Volumenstrom_vor"]),
        axis=1,
    )
    df_results["Strömungsgeschwindigkeit_nach"] = df_results.apply(
        lambda row: calc_flowSpeed(row["Volumenstrom_nach"]),
        axis=1,
    )

    # create new Pumpleistung column
    df_results["Pumpleistung_vor"] = df_results.apply(
        lambda row: calc_pumpleistung(
            row["Strömungsgeschwindigkeit_vor"],
            row["Volumenstrom_vor"],
            row["T_vl_vor"],
            Trl_vor,
        ),
        axis=1,
    )
    df_results["Pumpleistung_nach"] = df_results.apply(
        lambda row: calc_pumpleistung(
            row["Strömungsgeschwindigkeit_nach"],
            row["Volumenstrom_nach"],
            row["T_vl_nach"],
            Trl_nach,
        ),
        axis=1,
    )

    # display the result dataframe (for development puposes)
    # st.dataframe(df_results)
    df_results.to_json("results/df_results.json")

    # Create the plot of the different Temperatures
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_results["hours"],
        np.full(df_results["hours"].shape, Trl_nach),
        "--",
        linewidth=2,
        label="RL-Temperatur nach T-Absenkung",
        color="#EC9302",
    )
    plt.plot(
        df_results["hours"],
        np.full(df_results["hours"].shape, Trl_vor),
        "--",
        linewidth=2,
        label="RL-Temperatur vor T-Absenkung",
        color="#3795D5",
    )
    plt.plot(
        df_input["Zeit"],
        df_input["Lufttemp"],
        color="#356CA5",
        linewidth=2,
        label="Lufttemperatur",
    )
    plt.plot(
        df_input["Zeit"],
        df_input["Bodentemp"],
        color="#BFA405",
        linewidth=2,
        label="Bodentemperatur",
    )
    plt.plot(
        df_results["hours"],
        df_results["T_vl_vor"],
        color="#F7D507",
        linewidth=2,
        label="VL-Temperatur vor T-Absenkung",
    )
    plt.plot(
        df_results["hours"],
        df_results["T_vl_nach"],
        color="#7A1C1C",
        linewidth=2,
        label="VL-Temperatur nach T-Absenkung",
    )

    plt.xlabel("Zeit")
    plt.ylabel("Temperature")
    plt.title("Temperature vs Zeit")
    plt.legend()
    plt.grid(True)

    # Show the plot in Streamlit
    st.pyplot(plt)

    # Create the plot of the Netzverluste
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_results["hours"],
        df_results["Netzverluste_vor"],
        color="#1F4E79",
        linewidth=2,
        label="Netzverluste vor T-Absenkung",
    )
    plt.plot(
        df_results["hours"],
        df_results["Netzverluste_nach"],
        color="#3795D5",
        linewidth=2,
        label="Netzverluste nach T-Absenkung",
    )
    plt.xlabel("Zeit")
    plt.ylabel("Verluste")
    plt.title("Netzverluste vs Zeit")
    plt.legend()
    plt.grid(True)

    # Show the plot in Streamlit
    st.pyplot(plt)

    # Create the plot of VerlustProzentual (meint den prozentualen anteil der Verluste an der Gesamtwärmelast)
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_results["hours"],
        df_results["VerlustProzentual_nach"],
        color="#BFA405",
        linewidth=2,
        label="Prozentualer Verlust vor T-Absenkung",
    )
    plt.plot(
        df_results["hours"],
        df_results["VerlustProzentual_vor"],
        color="#F7D507",
        linewidth=2,
        label="Prozentualer Verlust nach T-Absenkung",
        alpha=0.5,
    )

    plt.xlabel("Zeit")
    plt.ylabel("%")
    plt.title("Prozentualer Verlust vs Zeit")
    plt.legend()
    plt.grid(True)

    # Show the plot in Streamlit
    st.pyplot(plt)

    # Create the plot of Volumenstrom
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_results["hours"],
        df_results["Volumenstrom_vor"],
        color="#AB2626",
        linewidth=2,
        label="Volumenstrom vor T-Absenkung",
    )
    plt.plot(
        df_results["hours"],
        df_results["Volumenstrom_nach"],
        color="#DD2525",
        linewidth=2,
        label="Volumenstrom nach T-Absenkung",
        alpha=0.7,
    )

    plt.xlabel("Zeit")
    plt.ylabel("Volumenstrom pro Stunde")
    plt.title("Volumenstrom pro Stunde vs Zeit")
    plt.legend()
    plt.grid(True)

    # Show the plot in Streamlit
    st.pyplot(plt)

    # Create the plot of Strömungsgeschwindigkeit
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_results["hours"],
        df_results["Strömungsgeschwindigkeit_vor"],
        color="#B77201",
        linewidth=2,
        label="Strömungsgeschwindigkeit vor T-Absenkung",
    )
    plt.plot(
        df_results["hours"],
        df_results["Strömungsgeschwindigkeit_nach"],
        color="#EC9302",
        linewidth=2,
        label="Strömungsgeschwindigkeit nach T-Absenkung",
        alpha=0.5,
    )

    plt.xlabel("Zeit")
    plt.ylabel("Strömungsgeschwindigkeit pro sekunde")
    plt.title("Strömungsgeschwindigkeit pro sekunde vs Zeit")
    plt.legend()
    plt.grid(True)

    # Show the plot in Streamlit
    # st.pyplot(plt)

    # Create the plot of Pumpleistung
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_results["hours"],
        df_results["Pumpleistung_vor"],
        color="#639729",
        linewidth=2,
        label="Pumpleistung vor T-Absenkung",
    )
    plt.plot(
        df_results["hours"],
        df_results["Pumpleistung_nach"],
        color="#92D050",
        linewidth=2,
        label="Pumpleistung nach T-Absenkung",
        alpha=0.5,
    )

    plt.xlabel("Zeit")
    plt.ylabel("Pumpleistung [kW]")
    plt.title("Pumpleistung vs Zeit")
    plt.legend()
    plt.grid(True)

    # Show the plot in Streamlit
    st.pyplot(plt)

    ##Bargraph to show the part of the Wärmelast that is due to losses in the Netz
    # Calculate the sums
    sum_warmelast_vor = df_results["Wärmelast_vor"].sum()
    sum_warmelast_nach = df_results["Wärmelast_nach"].sum()

    sum_netzverluste_vor = df_results["Netzverluste_vor"].sum()
    sum_netzverluste_nach = df_results["Netzverluste_nach"].sum()

    # Create a DataFrame from the sums
    df_sum = pd.DataFrame(
        {
            "Warmelast": ["vor", "nach"],
            "Wärmelast": [sum_warmelast_vor, sum_warmelast_nach],
            "Netzverluste": [sum_netzverluste_vor, sum_netzverluste_nach],
        }
    )

    # Create the bar plot
    fig, ax = plt.subplots()

    # Create the 'Wärmelast' bars
    ax.bar(df_sum["Warmelast"], df_sum["Wärmelast"], color="#515151", label="Wärmelast")

    # Add 'Netzverluste' on top
    ax.bar(
        df_sum["Warmelast"],
        df_sum["Netzverluste"],
        bottom=df_sum["Wärmelast"],
        color="#A3A3A3",
        label="Netzverluste",
    )

    ax.set_xlabel("Warmelast")
    ax.set_ylabel("Sum")
    ax.set_title("Sum of Wärmelast and Netzverluste vor vs nach")
    ax.legend()

    # Show the plot in Streamlit
    st.pyplot(fig)

    placeholder.write("Calculation finished")
    st.sidebar.success("Simulation erfolgreich")
