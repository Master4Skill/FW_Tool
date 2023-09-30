import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st
import numpy as np
from CoolProp.CoolProp import PropsSI
from streamlit_extras.app_logo import add_logo
from PIL import Image
import sys
import json
from scipy.interpolate import interp1d
import logging


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

# get user input
input_data["Tvl_max_vor"] = st.number_input(
    "Maximum supply temperature before temperature reduction", value=95
)
input_data["Tvl_min_vor"] = st.number_input(
    "Minimum supply temperature before temperature reduction", value=85
)
input_data["Trl_vor"] = st.number_input(
    "Return temperature before temperature reduction", value=60
)
input_data["Tvl_max_nach"] = st.number_input(
    "Maximum supply temperature after temperature reduction", value=75
)
input_data["Tvl_min_nach"] = st.number_input(
    "Minimum supply temperature after temperature reduction", value=65
)
input_data["Trl_nach"] = st.number_input(
    "Return temperature after temperature reduction", value=40
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
p_network = input_data["p_network"]

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
    first_50_rows = df_input.iloc[:50]
    # df_results["hours"] = first_50_rows["Zeit"]
    df_results["hours"] = df_input["Zeit"]
    # calculate T_VL
    # Calculate the 24-hour moving average of 'Lufttemp' and add it to df_results
    df_results["Air_average"] = df_input["Lufttemp"].rolling(window=24).mean()

    st.dataframe(df_results)

    def calculate_T_vl_vor(air_average):
        if air_average < 0:
            return Tvl_max_vor
        if 0 < air_average < 15:
            return Tvl_max_vor - (1 / 15) * air_average * (Tvl_max_vor - Tvl_min_vor)
        if air_average > 15:
            return Tvl_min_vor

    def calculate_T_vl_nach(air_average):
        if air_average < 0:
            return Tvl_max_nach
        if 0 < air_average < 15:
            return Tvl_max_nach - (1 / 15) * air_average * (Tvl_max_nach - Tvl_min_nach)
        if air_average > 15:
            return Tvl_min_nach

    # Apply the function to calculate 'T_vl'
    # df_results["T_vl_vor"] = df_results["Air_average"].apply(calculate_T_vl_vor)
    # For the first 24 entries, set "T_vl_vor" as Tvl_max_vor
    df_results.loc[:23, "T_vl_vor"] = Tvl_max_vor

    # For entries from the 24th onward, apply the function to calculate "T_vl_vor"
    df_results.loc[24:, "T_vl_vor"] = df_results.loc[24:, "Air_average"].apply(
        calculate_T_vl_vor
    )
    df_results.loc[:23, "T_vl_nach"] = Tvl_max_nach

    # For entries from the 24th onward, apply the function to calculate "T_vl_vor"
    df_results.loc[24:, "T_vl_nach"] = df_results.loc[24:, "Air_average"].apply(
        calculate_T_vl_nach
    )
    # df_results["T_vl_nach"] = df_results["Air_average"].apply(calculate_T_vl_nach)
    st.dataframe(df_results)

    # Calculate Netzverluste
    def calc_verlust(T_vl, T_b, T_rl):
        term1 = 4 * np.pi * ((T_vl + T_rl) / 2 - T_b)
        term2 = (1 / λD) * np.log(rM / rR)
        term3 = (1 / λB) * np.log(4 * (hÜ + rM) / rM)
        term4 = (1 / λB) * np.log(((2 * (hÜ + rM) / a + 2 * rM) ** 2 + 1) ** 0.5)
        return l_Netz / 1000 * term1 / (term2 + term3 + term4)

    # Calculate Wärmelast
    def calc_totalLast(Netzverluste, Lastgang):
        return (Lastgang / ηWüHüs + Netzverluste) / ηWüE

    def calc_VerlustProzentual(Netzverluste, Wärmelast):
        return Netzverluste / Wärmelast * 100

    # Calculate Volumenstrom und Strömungsgeschwindigkeit
    def calc_flowRate(T_vl, T_rl, Wärmelast):
        return Wärmelast / (ρ_water * cp_water * (T_vl - T_rl))

    def calc_flowSpeed(flowRate):
        return (flowRate / 3600) / (((np.pi) * rR**2) / 4)

    # calculate Pumpleistung

    def water_viscosity_CoolProp(T, p_network):
        """
        Calculate water viscosity given temperature in Celsius using CoolProp.
        T : temperature (degrees Celsius)
        """
        if pd.isna(T) or T < 0 or T > 150:  # Additional check for temperature range
            return 0  # Return None or some other default value if out of range or NaN

        T_K = T + 273.15  # Convert temperature to Kelvin
        P = 101325  # Assume atmospheric pressure in Pa

        # Get viscosity (in Pa.s)
        viscosity = PropsSI("VISCOSITY", "P", p_network, "T", T_K, "Water")

        return viscosity

    def calc_Reynolds(flowSpeed, T):
        return flowSpeed * rR * ρ_water / water_viscosity_CoolProp(T, p_network)

    def calc_pressureloss(Reynolds, flowSpeed):
        λ = 64 / Reynolds if Reynolds < 2300 else 0.3164 / Reynolds**0.25
        return λ * ρ_water / 2 * (flowSpeed**2) / rR * (l_Netz + ζ * rR / λ)

    def calc_pumpleistung(flowSpeed, flowRate, Tvl, Trl):
        Rey_vl = calc_Reynolds(flowSpeed, Tvl)
        Rey_rl = calc_Reynolds(flowSpeed, Trl)
        p_loss_vl = calc_pressureloss(Rey_vl, flowSpeed)
        p_loss_rl = calc_pressureloss(Rey_rl, flowSpeed)
        return ((p_loss_vl + p_loss_rl) * (flowRate / 3600)) / (ηPump * 1000)

    # create new Netzverluste column
    df_results["Netzverluste_vor"] = df_results.apply(
        lambda row: calc_verlust(
            row["T_vl_vor"], df_input["Bodentemp"][row.name], Trl_vor
        ),
        axis=1,
    )

    df_results["Netzverluste_nach"] = df_results.apply(
        lambda row: calc_verlust(
            row["T_vl_nach"], df_input["Bodentemp"][row.name], Trl_nach
        ),
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
        lambda row: calc_flowRate(row["T_vl_vor"], Trl_vor, row["Wärmelast_vor"]),
        axis=1,
    )
    df_results["Volumenstrom_nach"] = df_results.apply(
        lambda row: calc_flowRate(row["T_vl_nach"], Trl_nach, row["Wärmelast_nach"]),
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

    mpl.rcParams["font.size"] = 14

    # display the result dataframe (for development puposes)
    st.dataframe(df_results)
    df_results.to_json("results/df_results.json")

    # Create the plot of the different Temperatures
    plt.figure(figsize=(12, 6))

    # Plotting the data
    plt.plot(
        df_results["hours"],
        np.full(df_results["hours"].shape, Trl_nach),
        "--",
        linewidth=2,
        label="Return Temperature After Temp. Reduction",
        color="#EC9302",
    )
    plt.plot(
        df_results["hours"],
        np.full(df_results["hours"].shape, Trl_vor),
        "--",
        linewidth=2,
        label="Return Temperature Before Temp. Reduction",
        color="#3795D5",
    )
    plt.plot(
        df_input["Zeit"],
        df_input["Lufttemp"],
        color="#356CA5",
        linewidth=2,
        label="Air Temperature",
    )
    plt.plot(
        df_input["Zeit"],
        df_input["Bodentemp"],
        color="#BFA405",
        linewidth=2,
        label="Ground Temperature",
    )
    plt.plot(
        df_results["hours"],
        df_results["T_vl_vor"],
        color="#F7D507",
        linewidth=2,
        label="Flow Temperature Before Temp. Reduction",
    )
    plt.plot(
        df_results["hours"],
        df_results["T_vl_nach"],
        color="#7A1C1C",
        linewidth=2,
        label="Flow Temperature After Temp. Reduction",
    )

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    # Adding labels and title with specified font style
    plt.xlabel("Time [h]", fontsize=16, color="#777777", fontfamily="Segoe UI")
    plt.ylabel(
        "Temperature [°C]",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI",
    )
    plt.title(
        "Temperature vs Time",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI",
    )

    # Setting the legend style
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        fontsize=16,
        title_fontsize="16",
        labelcolor="#777777",
    )

    # Setting the tick parameters
    plt.tick_params(
        axis="x", colors="#777777", direction="out", which="both", labelsize=16
    )
    plt.tick_params(
        axis="y", colors="#777777", direction="out", which="both", labelsize=16
    )

    # Setting grid style
    plt.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

    # Setting the spines style
    plt.gca().spines["bottom"].set_edgecolor("#A3A3A3")
    plt.gca().spines["bottom"].set_linewidth(1)
    plt.gca().spines["left"].set_edgecolor("#A3A3A3")
    plt.gca().spines["left"].set_linewidth(1)

    # Setting the background color and hiding the top and right spines
    plt.gca().set_facecolor("white")
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)

    # Display the plot in Streamlit
    st.pyplot(plt)

    def styled_plot(x, y1, y2, color1, color2, label1, label2, xlabel, ylabel, title):
        plt.figure(figsize=(12, 6))
        plt.plot(x, y1, color=color1, linewidth=2, label=label1)
        plt.plot(x, y2, color=color2, linewidth=2, label=label2)

        # Styling
        plt.xlabel(xlabel, fontsize=16, color="#777777", fontfamily="Segoe UI")
        plt.ylabel(ylabel, fontsize=16, color="#777777", fontfamily="Segoe UI")
        plt.title(title, fontsize=16, color="#777777", fontfamily="Segoe UI")

        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            frameon=False,
            fontsize=16,
            title_fontsize="16",
            labelcolor="#777777",
        )

        plt.tick_params(
            axis="x", colors="#777777", direction="out", which="both", labelsize=16
        )
        plt.tick_params(
            axis="y", colors="#777777", direction="out", which="both", labelsize=16
        )

        plt.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

        plt.gca().spines["bottom"].set_edgecolor("#A3A3A3")
        plt.gca().spines["bottom"].set_linewidth(1)
        plt.gca().spines["left"].set_edgecolor("#A3A3A3")
        plt.gca().spines["left"].set_linewidth(1)
        plt.gca().set_facecolor("white")
        for spine in ["top", "right"]:
            plt.gca().spines[spine].set_visible(False)

        # Display
        st.pyplot(plt)

    # Applying the function for each plot
    styled_plot(
        df_results["hours"],
        df_results["Netzverluste_vor"],
        df_results["Netzverluste_nach"],
        "#1F4E79",
        "#3795D5",
        "Network Losses Before Temp. Reduction",
        "Network Losses After Temp. Reduction",
        "Time [h]",
        "Losses [kW]",
        "Network Losses vs Time",
    )

    styled_plot(
        df_results["hours"],
        df_results["VerlustProzentual_nach"],
        df_results["VerlustProzentual_vor"],
        "#BFA405",
        "#F7D507",
        "Percentage Loss Before Temp. Reduction",
        "Percentage Loss After Temp. Reduction",
        "Time [h]",
        "%",
        "Percentage Loss vs Time",
    )
    styled_plot(
        df_results["hours"],
        df_results["Volumenstrom_vor"],
        df_results["Volumenstrom_nach"],
        "#AB2626",
        "#DD2525",
        "Flow Rate Before Temp. Reduction",
        "Flow Rate After Temp. Reduction",
        "Time [h]",
        "Flow Rate per Hour [m³/h]",
        "Flow Rate per Hour vs Time",
    )
    styled_plot(
        df_results["hours"],
        df_results["Strömungsgeschwindigkeit_vor"],
        df_results["Strömungsgeschwindigkeit_nach"],
        "#B77201",
        "#EC9302",
        "Flow Velocity Before Temp. Reduction",
        "Flow Velocity After Temp. Reduction",
        "Time [h]",
        "Flow Velocity per Second [m/s]",
        "Flow Velocity per Second vs Time",
    )
    styled_plot(
        df_results["hours"],
        df_results["Pumpleistung_vor"],
        df_results["Pumpleistung_nach"],
        "#639729",
        "#92D050",
        "Pump Performance Before Temp. Reduction",
        "Pump Performance After Temp. Reduction",
        "Time [h]",
        "Pump Performance [kW]",
        "Pump Performance vs Time",
    )

    st.dataframe(df_results)

    def print_stats(x):
        flowrate = df_results[x].min()
        st.write(f"min:{x, flowrate}")
        flowrate = df_results[x].max()
        st.write(f"max:{x, flowrate}")
        flowrate = df_results[x].mean()
        st.write(f"mean:{x, flowrate}")
        flowrate = df_results[x].median()
        st.write(f"median:{x, flowrate}")
        return

    print_stats("Volumenstrom_vor")
    print_stats("Volumenstrom_nach")

    print_stats("Strömungsgeschwindigkeit_vor")
    print_stats("Strömungsgeschwindigkeit_nach")

    print_stats("Pumpleistung_vor")
    print_stats("Pumpleistung_nach")

    ##Bargraph to show the part of the Wärmelast that is due to losses in the Netz
    # Calculate the sums
    sum_consumerload = df_input["Lastgang"].sum()

    sum_netzverluste_vor = df_results["Netzverluste_vor"].sum()
    sum_netzverluste_nach = df_results["Netzverluste_nach"].sum()

    sum_HE_losses_vor = (
        df_results["Wärmelast_vor"].sum() - sum_consumerload - sum_netzverluste_vor
    )
    sum_HE_losses_nach = (
        df_results["Wärmelast_nach"].sum() - sum_consumerload - sum_netzverluste_nach
    )
    # Create a DataFrame from the sums
    df_sum = pd.DataFrame(
        {
            "Warmelast": ["before", "after"],
            "Wärmelast": [sum_consumerload / 1000000, sum_consumerload / 1000000],
            "Netzverluste": [
                sum_netzverluste_vor / 1000000,
                sum_netzverluste_nach / 1000000,
            ],
            "HE_losses": [sum_HE_losses_vor / 1000000, sum_HE_losses_nach / 1000000],
        }
    )

    st.dataframe(df_sum)

    # Create the bar plot
    fig, ax = plt.subplots()

    # Create the 'Wärmelast' bars
    bars1 = ax.bar(
        df_sum["Warmelast"], df_sum["Wärmelast"], color="#356CA5", label="Heat Load"
    )

    # Add 'Netzverluste' on top
    bars2 = ax.bar(
        df_sum["Warmelast"],
        df_sum["Netzverluste"],
        bottom=df_sum["Wärmelast"],
        color="#3795D5",
        label="Network Losses",
    )

    # Add 'HE_losses' on top
    bars3 = ax.bar(
        df_sum["Warmelast"],
        df_sum["HE_losses"],
        bottom=df_sum["Wärmelast"] + df_sum["Netzverluste"],
        color="#D7E6F5",
        label="Heat Exchanger Losses",
    )

    # Label and title configurations
    ax.set_ylabel(
        "Total Heat Load [GWh]",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI",
    )
    ax.set_title(
        "Sum of Heat Load and Network Losses before vs after Temp. reduction",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI",
    )

    # Legend with style configuration
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=16,
        labelcolor="#777777",
    )

    # X-axis properties
    ax.xaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["bottom"].set_linewidth(1)

    # Y-axis properties
    ax.yaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_linewidth(1)

    # Setting x-ticks with style configuration
    ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

    # Background and other spines color
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Function to add labels/numbers in the middle of the bars
    def add_labels(bars, prev_bars_heights):
        for bar, prev_bar_height in zip(bars, prev_bars_heights):
            height = bar.get_height()
            middle = prev_bar_height + height / 2
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, middle),
                ha="center",
                va="center",
            )

    # Empty array to store the previous bars' heights
    prev_bars_heights = [0] * len(df_sum["Warmelast"])

    # Add labels to each bar
    add_labels(bars1, prev_bars_heights)
    prev_bars_heights = df_sum["Wärmelast"]
    add_labels(bars2, prev_bars_heights)
    prev_bars_heights += df_sum["Netzverluste"]
    add_labels(bars3, prev_bars_heights)

    # Show the plot in Streamlit
    st.pyplot(fig)

    placeholder.write("Calculation finished")
    st.sidebar.success("Simulation erfolgreich")
