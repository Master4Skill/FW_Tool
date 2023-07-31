import gurobipy as gp
from gurobipy import GRB
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, PULP_CBC_CMD, GUROBI_CMD
import streamlit as st
from streamlit_extras.app_logo import add_logo
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import json as json
import numpy as np

# from pages.Erzeugerpark import names, erzeuger_df_vor


st.set_page_config(page_title="Plotting Demo2", page_icon="📈")
add_logo("resized_image.png")
st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")


def plot_data3(df, strompreise_export, strompreise):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Prepare data for producers and s_out
    cumulative = np.zeros_like(df.index, dtype=np.float64)
    colors = ["blue", "green", "red", "orange", "purple", "cyan"]

    for i, column in enumerate(["g1", "g2", "g4", "g6"]):
        if column == "s_in":
            cumulative -= df[column]
        else:
            cumulative += df[column]
        axs[0].plot(df.index, cumulative, label=column, linewidth=1, color=colors[i])

    # Plot P_to_dem as a standalone line
    axs[0].plot(df.index, df["P_to_dem"], label="P_to_dem", color="black", linewidth=1)

    # Plot E_stored as a standalone line
    axs[0].plot(
        df.index,
        -df["E_stored"],
        label="E_stored (negative)",
        color="olive",
        linewidth=1,
    )

    # Plot electricity prices on the second subplot
    axs[1].plot(
        df.index,
        strompreise_export,
        label="strompreise_export",
        color="magenta",
        linewidth=1,
    )
    axs[1].plot(df.index, strompreise, label="strompreise", color="teal", linewidth=1)

    # Set labels and title
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("Energy Values")
    axs[1].set_ylabel("Electricity Prices")
    axs[0].set_title("Energy Flow")
    axs[1].set_title("Electricity Prices")

    # Add legend
    axs[0].legend()
    axs[1].legend()

    # Add grid

    if hours < 400:
        for i in range(0, hours, 24):
            for ax in axs:
                ax.axvline(x=i, color="0.5", linestyle="--", linewidth=0.5)
        axs[0].yaxis.grid(True)
        axs[1].yaxis.grid(True)
    else:
        axs[0].grid(True)
        axs[1].grid(True)
    # Show the plot
    st.pyplot(fig)


def plot_data2(df, strompreise_export, strompreise):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for producers and s_out
    cumulative = np.zeros_like(df.index, dtype=np.float64)
    colors = ["blue", "green", "red", "orange", "purple", "cyan"]

    for i, column in enumerate(["g1", "g2", "g4", "g6"]):
        if column == "s_in":
            cumulative -= df[column]
        else:
            cumulative += df[column]
        ax.plot(df.index, cumulative, label=column, linewidth=1, color=colors[i])

    # Plot P_to_dem as a standalone line
    ax.plot(df.index, df["P_to_dem"], label="P_to_dem", color="black", linewidth=1)

    # Plot E_stored as a standalone line
    ax.plot(
        df.index,
        -df["E_stored"],
        label="E_stored (negative)",
        color="olive",
        linewidth=1,
    )

    # Create a secondary y-axis
    ax2 = ax.twinx()

    # Plot electricity prices on the secondary y-axis
    ax2.plot(
        df.index,
        strompreise_export,
        label="strompreise_export",
        color="magenta",
        linewidth=1,
    )
    ax2.plot(df.index, strompreise, label="strompreise", color="teal", linewidth=1)

    # Set labels and title
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Energy Values")
    ax2.set_ylabel("Electricity Prices")
    ax.set_title("Energy Flow")

    # Add legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    # Add grid
    ax.grid(True)

    # Show the plot
    st.pyplot(fig)


def plot_data(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for producers and s_out
    cumulative = np.zeros_like(df.index, dtype=np.float64)
    colors = ["blue", "green", "red", "orange", "purple", "cyan"]

    for i, column in enumerate(["g1", "g2", "g4", "g6"]):
        if column == "s_in":
            cumulative -= df[column]
        else:
            cumulative += df[column]
        ax.plot(df.index, cumulative, label=column, linewidth=1, color=colors[i])

    # Plot P_to_dem as a standalone line
    ax.plot(df.index, df["P_to_dem"], label="P_to_dem", color="black", linewidth=1)

    # Plot E_stored as a standalone line
    ax.plot(
        df.index,
        -df["E_stored"],
        label="E_stored (negative)",
        color="olive",
        linewidth=1,
    )

    # Set labels and title
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title("Energy Flow")

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True)

    # Show the plot
    st.pyplot(fig)


def main_pulp():
    df_results = pd.DataFrame()
    P_to_dem = [
        8490.9,
        8596.7,
        8649.7,
        8861.4,
        8940.8,
        8993.7,
        9020.2,
        9046.7,
        9073.1,
        8490.9,
        7485.1,
        6585.2,
        5897.1,
        5526.5,
        5447.1,
        5764.7,
        6638.1,
        7114.6,
        7352.8,
        7511.6,
        7696.8,
        7855.6,
        8067.4,
    ]

    def read_demand(file_name, hours):
        # read csv file into a dataframe
        df = pd.read_csv(file_name, nrows=hours, sep=",")

        # Convert strings with comma as decimal separator to float
        df["Lastgang"] = (
            df["Lastgang"]
            .str.replace('"', "")
            .str.replace(".", "", regex=False)
            .str.replace(",", ".")
            .astype(float)
        )

        # convert column to a list
        P_to_dem = df["Lastgang"].tolist()

        return P_to_dem

    def read_data(file_name, hours):
        # read csv file into a dataframe
        df = pd.read_csv(file_name, nrows=hours, sep=";")

        # Convert strings with comma as decimal separator to float
        df["gaspreis_22"] = df["gaspreis_22"].str.replace(",", ".").astype(float)

        # convert each column to a list
        strompreise_export = df["strompreis_22_ex"].tolist()
        strompreise = df["strompreis_22_in"].tolist()
        gaspreise = df["gaspreis_22"].tolist()
        Flusstemperatur = df["Isartemp"].tolist()

        return strompreise_export, strompreise, gaspreise, Flusstemperatur

    file_name = "Input_Netz.csv"
    # hours = st.number_input("Enter hours", min_value=1, value=2000)

    P_to_dem = read_demand(file_name, hours)

    file_name = "Zeitreihen/zeitreihen_22.csv"

    strompreise_export, strompreise, gaspreise, Flusstemperatur = read_data(
        file_name, hours
    )

    with open("data.json") as f:
        data = json.load(f)

    names = data["names"]
    erzeuger_df_vor = pd.read_json(data["erzeuger_df_vor"])
    Tiefentemperatur = 120

    average_strompreise = sum(strompreise) / len(strompreise)
    I = list(range(len(P_to_dem)))

    m = LpProblem("Fernwärme_Erzeugungslastgang", LpMinimize)
    ###Parameters
    T = 24  # timesteps
    ΔT = 1  # hours
    # comodities
    K_elec_imp = 100000
    K_elec_exp = 100000
    # storage
    K_s_pow_in = 3000  # kW
    K_s_pow_out = 3000
    # K_s_en = 5000
    Y_s_self = 0.02
    # Masterarbeit Therese Farber Formel/Prozent für Speicherverluste

    # processes; später hier für K_i Potentiale aus Erzeugerpark importieren

    K_p = {}

    for i, name in enumerate(names):
        K_p[name] = [
            0 if v is None else v
            for v in erzeuger_df_vor[f"Erzeuger_{i+1}_vor"].tolist()
        ]

    # Abwärme
    K_p1 = K_p["Abwärme"] if "Abwärme" in names else [0] * 8761

    # Waermepumpe1
    K_p2 = K_p["Waermepumpe1"] if "Waermepumpe1" in names else [0] * 8761
    PL_p2 = 0.2

    # Waermepumpe2
    K_p3 = K_p["Waermepumpe2"] if "Waermepumpe2" in names else [0] * 8761
    PL_p3 = 0.2

    # Geothermie
    K_p4 = K_p["Geothermie"] if "Geothermie" in names else [0] * 8761
    PL_p4 = 0.2

    # Solar
    K_p5 = K_p["Solar"] if "Solar" in names else [0] * 8761
    PL_p5 = 0

    # Spitzenlastkessel
    K_p6 = K_p["Spitzenlastkessel"] if "Spitzenlastkessel" in names else [0] * 8761
    PL_p6 = 0.01

    # BHKW
    K_p7 = K_p["BHKW"] if "BHKW" in names else [0] * 8761
    PL_p7 = 0.1

    ###decision variables
    e_imp = LpVariable.dicts(
        "ρ_imp_elec_t", I, lowBound=0, upBound=K_elec_imp, cat="Continuous"
    )
    e_exp = LpVariable.dicts(
        "ρ_exp_elec_t", I, lowBound=0, upBound=K_elec_exp, cat="Continuous"
    )
    g_imp = LpVariable.dicts("ρ_imp_gas_t", I, lowBound=0, cat="Continuous")
    s_in = LpVariable.dicts(
        "ρ_in_s_t", I, lowBound=0, upBound=K_s_pow_in, cat="Continuous"
    )
    s_out = LpVariable.dicts(
        "ρ_out_s_t", I, lowBound=0, upBound=K_s_pow_out, cat="Continuous"
    )
    E_stored = LpVariable.dicts("E_stored", I, lowBound=0, cat="Continuous")
    f21 = LpVariable.dicts("ρ_in_WP1_elec_t", I, lowBound=0, cat="Continuous")
    # f22 = LpVariable.dicts("ρ_in_WP1_water_t", I, lowBound=0, cat="Continuous")
    f31 = LpVariable.dicts("ρ_in_WP2_elec_t", I, lowBound=0, cat="Continuous")
    # f32 = LpVariable.dicts("ρ_in_WP2_water_t", I, lowBound=0, cat="Continuous")
    f4 = LpVariable.dicts("ρ_in_Geo_elec_t", I, lowBound=0, cat="Continuous")
    f5 = LpVariable.dicts("ρ_in_Solar_sun_t", I, lowBound=0, cat="Continuous")
    f6 = LpVariable.dicts("ρ_in_Spizenkessel_gas_t", I, lowBound=0, cat="Continuous")
    f7 = LpVariable.dicts("ρ_in_BHKW_gas_t", I, lowBound=0, cat="Continuous")
    g1 = LpVariable.dicts("ρ_out_Abwärme_heat_t", I, lowBound=0, cat="Continuous")
    g2 = LpVariable.dicts("ρ_out_WP1_heat_t", I, lowBound=0, cat="Continuous")
    g3 = LpVariable.dicts("ρ_out_WP2_heat_t", I, lowBound=0, cat="Continuous")
    g4 = LpVariable.dicts("ρ_out_Geo_heat_t", I, lowBound=0, cat="Continuous")
    g5 = LpVariable.dicts("ρ_out_Solar_heat_t", I, lowBound=0, cat="Continuous")
    g6 = LpVariable.dicts("ρ_out_Spizenkessel_heat_t", I, lowBound=0, cat="Continuous")
    g71 = LpVariable.dicts("ρ_out_BHKW_heat_t", I, lowBound=0, cat="Continuous")
    g72 = LpVariable.dicts("ρ_out_BHKW_elec_t", I, lowBound=0, cat="Continuous")
    x1 = LpVariable.dicts("Abwärme_ON_OFF", I, lowBound=0, upBound=1, cat="Integer")
    x2 = LpVariable.dicts("WP1_ON_OFF", I, lowBound=0, upBound=1, cat="Integer")
    x3 = LpVariable.dicts("WP2_ON_OFF", I, lowBound=0, upBound=1, cat="Integer")
    x4 = LpVariable.dicts("Geo_ON_OFF", I, lowBound=0, upBound=1, cat="Integer")
    x5 = LpVariable.dicts("Solar_ON_OFF", I, lowBound=0, upBound=1, cat="Integer")
    x6 = LpVariable.dicts(
        "Spizenkessel_ON_OFF", I, lowBound=0, upBound=1, cat="Integer"
    )
    x7 = LpVariable.dicts("BHKW_ON_OFF", I, lowBound=0, upBound=1, cat="Integer")
    x_s = LpVariable.dicts(
        "storage_Charge_Discharge", I, lowBound=0, upBound=1, cat="Integer"
    )
    E_tot = LpVariable("total_Euros")
    E_imp = LpVariable("Euros_for_Import")
    E_exp = LpVariable("Euros_for_Export")
    # K_s_en = LpVariable("storage_energy_capacity", lowBound=0, cat="Continuous")
    K_s_en = 3000
    ###Constraints
    m += E_tot
    m += E_tot == E_imp + E_exp + 1 * K_s_en
    m += (
        E_imp
        == sum(strompreise[i] * e_imp[i] for i in I) / 100
        + sum(gaspreise[i] * g_imp[i] for i in I) / 100
    )
    m += E_exp == sum(strompreise_export[i] * e_exp[i] for i in I) / 100
    # m += X == sum(average_strompreise * x[i] for i in I) / 100

    for i in range(len(P_to_dem)):
        P = P_to_dem[i]
        A_p2_in = 0.45 * (60 + 273.15) / (60 - Flusstemperatur[i])
        A_p3_in = 0.45 * (60 + 273.15) / (60 - Flusstemperatur[i])
        A_p4_in = -0.45 * (60 + 273.15) / (60 - Tiefentemperatur)
        A_p5_in = 0.5
        A_p6_in = 0.8
        A_p7_in = 1
        A_p7_out_elec = 0.38
        A_p7_out_heat = 0.42

        # Storage
        if i == 0:  # for the first period, E_stored_prev is 0
            m += E_stored[i] == s_in[i] - s_out[i]
        else:  # for other periods, E_stored_prev is E_stored from last period
            m += E_stored[i] == (1 - Y_s_self) * E_stored[i - 1] + s_in[i] - s_out[i]
        m += E_stored[i] <= K_s_en
        # Process Flows
        # p1
        m += g1[i] <= K_p1[i]
        # p2
        m += g2[i] == f21[i] * A_p2_in
        m += g2[i] >= x2[i] * K_p2[i] * PL_p2
        m += g2[i] <= x2[i] * K_p2[i]
        # p3
        m += g3[i] == f31[i] * A_p3_in
        m += g3[i] >= x3[i] * K_p3[i] * PL_p3
        m += g3[i] <= x3[i] * K_p3[i]
        # p4
        m += g4[i] == f4[i] * A_p4_in
        m += g4[i] >= x4[i] * K_p4[i] * PL_p4
        m += g4[i] <= x4[i] * K_p4[i]
        # p5
        m += g5[i] == f5[i] * A_p5_in
        m += g5[i] <= x5[i] * K_p5[i]
        # p6
        m += g6[i] == f6[i] * A_p6_in
        m += g6[i] >= x6[i] * K_p6[i] * PL_p6
        m += g6[i] <= x6[i] * K_p6[i]
        # p7
        m += g71[i] == f7[i] * A_p7_out_heat
        m += g72[i] == f7[i] * A_p7_out_elec
        m += g71[i] + g72[i] >= x7[i] * K_p7[i] * PL_p7
        m += g71[i] + g72[i] <= x7[i] * K_p7[i]

        # Commodities
        # heat
        m += (
            g1[i] + g2[i] + g3[i] + g4[i] + g5[i] + g6[i] + g72[i] + s_out[i]
            == s_in[i] + P
        )
        # electricity
        m += e_imp[i] == f21[i] + f31[i] + f4[i] + e_exp[i]
        # gas
        m += g_imp[i] == f6[i] + f7[i]

    solver = GUROBI_CMD(options=[("MIPGap", 0.01)])
    # solver = GUROBI_CMD()

    m.solve(solver)

    # Extract solution
    e_imp_m = {i: e_imp[i].varValue for i in I}
    e_exp_m = {i: e_exp[i].varValue for i in I}
    s_in_m = {i: s_in[i].varValue for i in I}
    s_out_m = {i: s_out[i].varValue for i in I}
    E_stored_m = {i: E_stored[i].varValue for i in I}
    f21_m = {i: f21[i].varValue for i in I}
    f31_m = {i: f31[i].varValue for i in I}
    f4_m = {i: f4[i].varValue for i in I}
    f5_m = {i: f5[i].varValue for i in I}
    f6_m = {i: f6[i].varValue for i in I}
    f7_m = {i: f7[i].varValue for i in I}
    g1_m = {i: g1[i].varValue for i in I}
    g2_m = {i: g2[i].varValue for i in I}
    g3_m = {i: g3[i].varValue for i in I}
    g4_m = {i: g4[i].varValue for i in I}
    g5_m = {i: g5[i].varValue for i in I}
    g6_m = {i: g6[i].varValue for i in I}
    g71_m = {i: g71[i].varValue for i in I}
    g72_m = {i: g72[i].varValue for i in I}
    P_to_dem_m = {i: P_to_dem[i] for i in I}
    # K_s_en_m = K_s_en.varValue
    E_tot_m = E_tot.varValue

    df_results = pd.DataFrame()

    for key in e_imp_m.keys():
        data = {
            # "K_s_en": K_s_en_m,
            "E_tot": E_tot_m,
            "e_imp": e_imp_m[key],
            "e_exp": e_exp_m[key],
            "s_in": s_in_m[key],
            "s_out": s_out_m[key],
            "E_stored": E_stored_m[key],
            "g1": g1_m[key],
            "g2": g2_m[key],
            "g4": g4_m[key],
            "g6": g6_m[key],
            "P_to_dem": P_to_dem_m[key],
        }
        df_results = pd.concat(
            [df_results, pd.DataFrame(data, index=[key])], ignore_index=True
        )

    print(df_results)

    plot_data3(df_results, strompreise_export, strompreise)


hours = st.number_input("Enter hours", min_value=1, value=2000)
if st.button("Submit"):
    st.write(f"You entered {hours} hours.")

    main_pulp()
