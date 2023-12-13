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

    
    my_dict = {f"Erzeuger_{i+1}": name for i, name in enumerate(names)}

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
