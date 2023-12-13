import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from CoolProp.CoolProp import PropsSI
from streamlit_extras.app_logo import add_logo
from PIL import Image
import sys
import json
import seaborn as sns
import ErzeugerparkClasses as ep
from plotting_functions import (
    plot_actual_production,
    plot_sorted_production,
    plot_power_usage,
    plot_total_change,
    plot_total_emissions,
)

with open("color_FFE.json", "r") as file:
    # Load the contents of the file into a Python object
    color_FFE = json.load(file)
add_logo("resized_image.png")

st.header("Final Results")

st.info("  Please run the individual simulations first", icon="⚠️")

if st.button("Show the Results"):
    # Load the data from the json file
    with open("results/variables.json", "r") as f:
        input_data = json.load(f)

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

    # Load the JSON data
    df_results = pd.read_json("results/df_results.json")

    st.header("Verteilnetz Simulation")

    # Create the plot of the different Temperatures
    plt.figure(figsize=(12, 6))

    # Plotting the data
    plt.plot(
        df_results["hours"],
        np.full(df_results["hours"].shape, Trl_nach),
        "--",
        linewidth=2,
        label="Return Temperature after Temp. Reduction",
        color="#EC9302",
    )
    plt.plot(
        df_results["hours"],
        np.full(df_results["hours"].shape, Trl_vor),
        "--",
        linewidth=2,
        label="Return Temperature before Temp. Reduction",
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
        label="Flow Temperature before Temp. Reduction",
    )
    plt.plot(
        df_results["hours"],
        df_results["T_vl_nach"],
        color="#7A1C1C",
        linewidth=2,
        label="Flow Temperature after Temp. Reduction",
    )


    # Adding labels and title with specified font style
    plt.xlabel("Time [h]", fontsize=16, color="#777777", fontfamily="Segoe UI")
    plt.ylabel(
        "Temperature [°C]",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI",
    )
    plt.title(
        "Course of Network Temperatures before and after Temperature Reduction",
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
        "Network Losses before Temp. Reduction",
        "Network Losses after Temp. Reduction",
        "Time [h]",
        "Losses [kW]",
        "Network Losses before and after Temperature Reduction",
    )

    styled_plot(
        df_results["hours"],
        df_results["VerlustProzentual_nach"],
        df_results["VerlustProzentual_vor"],
        "#BFA405",
        "#F7D507",
        "Percentage Loss before Temp. Reduction",
        "Percentage Loss after Temp. Reduction",
        "Time [h]",
        "%",
        "Notwork Losses Percentage of Total Heat Load before and after Temperature Reduction",
    )
    styled_plot(
        df_results["hours"],
        df_results["Volumenstrom_vor"],
        df_results["Volumenstrom_nach"],
        "#AB2626",
        "#DD2525",
        "Flow Rate before Temp. Reduction",
        "Flow Rate after Temp. Reduction",
        "Time [h]",
        "Flow Rate per Hour [m³/h]",
        "Flow Rate per Hour before and after Temperature Reduction",
    )
    styled_plot(
        df_results["hours"],
        df_results["Strömungsgeschwindigkeit_vor"],
        df_results["Strömungsgeschwindigkeit_nach"],
        "#B77201",
        "#EC9302",
        "Flow Velocity before Temp. Reduction",
        "Flow Velocity after Temp. Reduction",
        "Time [h]",
        "Flow Velocity per Second [m/s]",
        "Flow Velocity per Second before and after Temperature Reduction",
    )
    styled_plot(
        df_results["hours"],
        df_results["Pumpleistung_vor"],
        df_results["Pumpleistung_nach"],
        "#639729",
        "#92D050",
        "Pump Performance before Temp. Reduction",
        "Pump Performance after Temp. Reduction",
        "Time [h]",
        "Pump Performance [kW]",
        "Pump Performance before and after Temperature Reduction",
    )

    # st.dataframe(df_results)

    def print_stats(x):
        # Funktion do print the stats of a column, mainly for development and troubleshooting purposes
        flowrate = df_results[x].min()
        st.write(f"min:{x, flowrate}")
        flowrate = df_results[x].max()
        st.write(f"max:{x, flowrate}")
        flowrate = df_results[x].mean()
        st.write(f"mean:{x, flowrate}")
        flowrate = df_results[x].median()
        st.write(f"median:{x, flowrate}")
        return

    # print_stats("Volumenstrom_vor")
    # print_stats("Volumenstrom_nach")

    # print_stats("Strömungsgeschwindigkeit_vor")
    # print_stats("Strömungsgeschwindigkeit_nach")

    # print_stats("Pumpleistung_vor")
    # print_stats("Pumpleistung_nach")

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

    # st.dataframe(df_sum)

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

    ################# Ergebnisse Erzeugerpark######################

    # Load the data from the json file
    actual_production_df_vor = pd.read_json("results/actual_production_df_vor.json")
    actual_production_df_nach = pd.read_json("results/actual_production_df_nach.json")

    Power_df_vor = pd.read_json("results/Power_df_vor.json")
    Power_df_nach = pd.read_json("results/Power_df_nach.json")

    CO2_df_vor = pd.read_json("results/CO2_df_vor.json")
    CO2_df_nach = pd.read_json("results/CO2_df_nach.json")

    st.header("Ergebnisse Erzeugerpark")

    # Define color list
    with open("results/color_FFE.json", "r") as f:
        color_FFE = json.load(f)

    # Create and sort sorted_df before plotting it
    sorted_df = actual_production_df_vor.copy()
    for col in sorted_df.columns:
        sorted_df[col] = sorted_df[col].sort_values(ascending=False).values

    #names = [name_mapping.get(obj.__class__.__name__, obj.__class__.__name__)for obj in erzeugerpark]

    with open("erzeuger_df_vor.json") as f:
        data = json.load(f)

    names = data["names"]

    name_mapping = {
        "waste_heat": "Waste Heat",
        "heatpump_1": "Waste Heat Pump",
        "heatpump_2": "Ambient\nHeat Pump",
        "solarthermal": "Solar Thermal",
        "geothermal": "Geothermal",
        "PLB": "Peak Load Boiler",
        "CHP": "CHP",
    }

    names_mapped = [name_mapping.get(name, name) for name in names]

    
    my_dict = {f"Erzeuger_{i+1}": name for i, name in enumerate(names_mapped)}

    with st.container():
        st.header("Generation load profile")
        st.subheader("Before Temperature Reduction")
        sorted_df_vor = plot_actual_production(
            df_results,
            "Wärmelast_vor",
            actual_production_df_vor,
            color_FFE,
            "Generation load curve berfore",
            my_dict,
            0,
        )
        st.subheader("After Temperature Reduction")
        sorted_df_nach = plot_actual_production(
            df_results,
            "Wärmelast_nach",
            actual_production_df_nach,
            color_FFE,
            "Generation load curve after",
            my_dict,
            0,
        )

    # Create the second container
    with st.container():
        st.header("Annual duration line")
        st.subheader("Before Temperature Reduction")
        plot_df_vor = plot_sorted_production(
            df_results,
            "Wärmelast_vor",
            sorted_df_vor,
            actual_production_df_vor,
            color_FFE,
            "Annual duration line before",
            my_dict,
        )
        st.subheader("After Temperature Reduction")
        plot_df_nach = plot_sorted_production(
            df_results,
            "Wärmelast_nach",
            sorted_df_nach,
            actual_production_df_nach,
            color_FFE,
            "Annual duration line after",
            my_dict,
        )

    plot_power_usage(Power_df_vor, Power_df_nach, my_dict, color_FFE)

    plot_total_change(
        actual_production_df_vor,
        actual_production_df_nach,
        color_FFE,
        "before Temp. Reduction",
        "after Temp. Reduction",
        "Erzeuger",
        "Change in Heat Generation",
        "",
        "Total Production [GWh]",
        my_dict,
        0.7,
        0.8,
    )

    plot_total_change(
        CO2_df_vor,
        CO2_df_nach,
        color_FFE,
        "before Temp. Reduction",
        "after Temp. Reduction",
        "Erzeuger",
        "Change in CO2 Emissions",
        "",
        "Total Emissions [kt CO2]",
        my_dict,
        0.7,
        0.8,
    )

    plot_total_change(
        Power_df_vor,
        Power_df_nach,
        color_FFE,
        "before Temp. Reduction",
        "after Temp. Reduction",
        "Erzeuger",
        "Change in Power Consumption",
        "",
        "Total Usage [GWh]",
        my_dict,
        0.4,
        0.8,
    )

    st.sidebar.success("Simulation erfolgreich")
