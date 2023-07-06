import streamlit as st
import pandas as pd
from streamlit_extras.app_logo import add_logo
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import ErzeugerparkClasses as ep
from plotting_functions import (
    plot_actual_production,
    plot_sorted_production,
    plot_power_usage,
    plot_total_change,
    plot_total_emissions,
)


with open("results/data.json", "r") as f:
    input_data = json.load(f)


df_results = pd.read_json("results/df_results.json")


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


color_dict = {
    "♨️ Abwärme - begrenzter Volumenstrom der Quelle (m³/h), Abwärmetemperatur konstant über Netztemperatur (°C)": "#639729",
    "🚰 Wärmepumpe1 - begrenzter Volumenstrom der Quelle (m³/h), Quelltemperatur konstant (°C)": "#1F4E79",
    "🌊 Wärmepumpe2 - Begrenzte Leistung (kW), bei schwankender Quelltemperatur (°C)": "#F7D507",
    "⛰️ Geothermie - Maximale Leistung (kW)": "#DD2525",
    "☀️ Solarthermie - Sonneneinstrahlung (kW/m²)": "#92D050",
    "🔥 Spitzenlastkessel - Maximale Leistung (kW)": "#EC9302",
    "🏭 BHKW - Maximale Leistung (kW)": "#639729",
}


st.set_page_config(
    page_title="Erzeugerpark",
    page_icon="🏭",
)
add_logo("resized_image.png")

st.sidebar.header("Temperaturabsenkung in Erzeugerkombinationen")

st.sidebar.info("Wählen Sie die gewünschten Erzeuger")

st.markdown("# Erzeugerparksimulation")

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

# st.dataframe(df_input)

erzeugerpark = []

# Maximum Anzahl an Erzeuger Eingaben
max_erzeuger = 7
anzahl_erzeuger = st.slider(
    "Wie viele Erzeuger möchten Sie hinzufügen?", 1, max_erzeuger
)


for i in range(anzahl_erzeuger):
    st.markdown(f"### Erzeuger {i+1}")
    erzeuger_type = st.selectbox(
        f"Bitte den Typ des Erzeugers {i+1} auswählen",
        [
            "♨️ Abwärme - begrenzter Volumenstrom der Quelle (m³/h), Abwärmetemperatur konstant über Netztemperatur (°C)",
            "🚰 Wärmepumpe1 - begrenzter Volumenstrom der Quelle (m³/h), Quelltemperatur konstant (°C)",
            "🌊 Wärmepumpe2 - Begrenzte Leistung (kW), bei schwankender Quelltemperatur (°C)",
            "⛰️ Geothermie - Maximale Leistung (kW)",
            "☀️ Solarthermie - Sonneneinstrahlung (kW/m²)",
            "🔥 Spitzenlastkessel - Maximale Leistung (kW)",
            "🏭 BHKW - Maximale Leistung (kW)",
        ],
        key=f"Erzeuger{i}",
    )

    erzeuger_color = color_dict[erzeuger_type]

    if "Abwärme" in erzeuger_type:
        Volumenstrom_quelle = st.number_input(
            "Bitte Volumenstrom_quelle eingeben (m³/h)",
            value=10,
            key=f"Volumenstrom_quelle{i}",
        )
        Abwärmetemperatur = st.number_input(
            "Bitte Quelltemperatur eingeben (°C)", value=120, key=f"Quelltemperatur{i}"
        )
        erzeuger = ep.Abwärme(
            Volumenstrom_quelle,
            Abwärmetemperatur,
            color=erzeuger_color,
            co2_emission_factor=0,
        )

    elif "Wärmepumpe1" in erzeuger_type:
        Volumenstrom_quelle = st.number_input(
            "Bitte Volumenstrom_quelle eingeben (m³/h)",
            value=100,
            key=f"Volumenstrom_quelle{i}",
        )
        T_q = st.number_input(
            "Bitte Quelltemperatur eingeben (°C)", value=25, key=f"Quelltemperatur{i}"
        )
        Gütegrad = st.number_input(
            "Bitte Gütegrad Wasser-Wasser angeben", value=0.45, key=f"Gütegrad{i}"
        )
        # Leistung_max = st.number_input("Bitte maximale Leistung eingeben (kW)", key=f"Leistung_max{i}")
        erzeuger = ep.Waermepumpe1(
            Volumenstrom_quelle,
            T_q,
            Gütegrad,
            color=erzeuger_color,
            co2_emission_factor=0.468,
        )

    elif "Wärmepumpe2" in erzeuger_type:
        Leistung_max = st.number_input(
            "Bitte maximale Leistung eingeben (kW)", value=5000, key=f"Leistung_max{i}"
        )
        T_q = st.number_input(
            "Bitte Quelltemperatur eingeben (°C)", value=20, key=f"T_q{i}"
        )
        Gütegrad = st.number_input(
            "Bitte Gütegrad Wasser-Wasser angeben", value=0.45, key=f"Gütegrad{i}"
        )
        erzeuger = ep.Waermepumpe2(
            Leistung_max, T_q, Gütegrad, color=erzeuger_color, co2_emission_factor=0.468
        )

    elif "Geothermie" in erzeuger_type:
        Leistung_max = st.number_input(
            "Bitte maximale Leistung eingeben (kW)", value=3000, key=f"Leistung_max{i}"
        )
        Tgeo = st.number_input(
            "Bitte Temperatur der Geothermie eingeben (°C)", value=100, key=f"Tgeo{i}"
        )
        h_förder = st.number_input(
            "Bitte Förderhöhe eingeben (m)", value=2000, key=f"h_förder{i}"
        )
        η_geo = st.number_input(
            "Bitte Wirkungsgrad der Geothermiepumpe eingeben (%)",
            value=0.8,
            key=f"η_geo{i}",
        )
        erzeuger = ep.Geothermie(
            Leistung_max,
            Tgeo,
            h_förder,
            η_geo,
            color=erzeuger_color,
            co2_emission_factor=0.468,
        )

    elif "Solarthermie" in erzeuger_type:
        Sun_in = st.number_input(
            "Bitte Sonneneinstrahlung eingeben (kW/m²)", key=f"Sun_in{i}"
        )
        erzeuger = ep.Solarthermie(Sun_in, color=erzeuger_color)

    elif "Spitzenlastkessel" in erzeuger_type:
        Leistung_max = st.number_input(
            "Bitte maximale Leistung eingeben (kW)", value=10000, key=f"Leistung_max{i}"
        )
        erzeuger = ep.Spitzenlastkessel(
            Leistung_max, color=erzeuger_color, co2_emission_factor=0.201
        )

    elif "BHKW" in erzeuger_type:
        Leistung_max = st.number_input(
            "Bitte maximale Leistung eingeben (kW)",
            value=5000,
            key=f"Leistung_max{i}",
        )
        erzeuger = ep.BHKW(
            Leistung_max, color=erzeuger_color, co2_emission_factor=0.201
        )
    else:
        st.write("Bitte wählen Sie einen gültigen Erzeugertyp aus")

    erzeugerpark.append(erzeuger)


if st.button("Calculate"):
    # Zeige den Erzeugerpark
    # df_erzeuger = pd.DataFrame([vars(erzeuger) for erzeuger in erzeugerpark])
    # st.dataframe(df_erzeuger)
    st.write(erzeugerpark)
    df_input = df_input.iloc[:-2]

    erzeuger_df_vor = pd.DataFrame(index=df_input.index)
    erzeuger_df_vor.index = df_input.index

    erzeuger_df_nach = pd.DataFrame(index=df_input.index)
    erzeuger_df_nach.index = df_input.index

    for i, erzeuger in enumerate(erzeugerpark):
        # Calculate Wärmeleistung for each hour
        waermeleistung_vor = [
            erzeuger.calc_output(hour, df_results.loc[hour, "T_vl_vor"], Trl_vor)
            for hour in df_input.index
        ]
        waermeleistung_nach = [
            erzeuger.calc_output(hour, df_results.loc[hour, "T_vl_nach"], Trl_nach)
            for hour in df_input.index
        ]
        # Add a column to the dataframe with Wärmeleistung for each hour
        erzeuger_df_vor[f"Erzeuger_{i+1}_vor"] = waermeleistung_vor
        erzeuger_df_nach[f"Erzeuger_{i+1}_nach"] = waermeleistung_nach

    st.dataframe(erzeuger_df_vor)
    st.dataframe(erzeuger_df_nach)

    actual_production_data_vor = []
    actual_production_data_nach = []

    # Iterate over each hour
    for hour in df_input.index:
        # Get the demand for this hour
        demand_vor = demand_nach = df_input.loc[hour, "Lastgang"]

        # Initialize a dictionary to store the actual production of each Erzeuger
        actual_production_vor = {}
        actual_production_nach = {}

        # Iterate over each Erzeuger
        for i, erzeuger in enumerate(erzeugerpark):
            # Get the production potential of this Erzeuger for this hour
            potential_vor = erzeuger_df_vor.loc[hour, f"Erzeuger_{i+1}_vor"]

            # If the potential is greater than the remaining demand,
            # the Erzeuger only needs to produce the remaining demand
            if potential_vor > demand_vor:
                actual_production_vor[f"Erzeuger_{i+1}_vor"] = demand_vor
                demand_vor = 0
            # Otherwise, the Erzeuger produces its maximum potential,
            # and the remaining demand is reduced accordingly
            else:
                actual_production_vor[f"Erzeuger_{i+1}_vor"] = potential_vor
                demand_vor -= potential_vor

        for i, erzeuger in enumerate(erzeugerpark):
            # Get the production potential of this Erzeuger for this hour
            potential_nach = erzeuger_df_nach.loc[hour, f"Erzeuger_{i+1}_nach"]

            # If the potential is greater than the remaining demand,
            # the Erzeuger only needs to produce the remaining demand
            if potential_nach > demand_nach:
                actual_production_nach[f"Erzeuger_{i+1}_nach"] = demand_nach
                demand_nach = 0
            # Otherwise, the Erzeuger produces its maximum potential,
            # and the remaining demand is reduced accordingly
            else:
                actual_production_nach[f"Erzeuger_{i+1}_nach"] = potential_nach
                demand_nach -= potential_nach

        # Add the actual production for this hour to the list
        actual_production_data_vor.append(actual_production_vor)
        actual_production_data_nach.append(actual_production_nach)

    # Convert list of dictionaries to DataFrame
    actual_production_df_vor = pd.DataFrame(actual_production_data_vor)

    actual_production_df_nach = pd.DataFrame(actual_production_data_nach)

    # create df of the Powerusage
    Power_df_vor = pd.DataFrame(index=df_input.index)
    Power_df_nach = pd.DataFrame(index=df_input.index)

    Power_df_vor.index = df_input.index
    for i, erzeuger in enumerate(erzeugerpark):
        Powerusage_vor = [
            erzeuger.calc_Poweruse(
                hour,
                df_results.loc[hour, "T_vl_vor"],
                Trl_vor,
                actual_production_df_vor.loc[hour, f"Erzeuger_{i+1}_vor"],
            )
            for hour in df_input.index
        ]
        Power_df_vor[f"Erzeuger_{i+1}_vor"] = Powerusage_vor

    Power_df_nach.index = df_input.index
    for i, erzeuger in enumerate(erzeugerpark):
        Powerusage_nach = [
            erzeuger.calc_Poweruse(
                hour,
                df_results.loc[hour, "T_vl_nach"],
                Trl_nach,
                actual_production_df_nach.loc[hour, f"Erzeuger_{i+1}_nach"],
            )
            for hour in df_input.index
        ]
        Power_df_nach[f"Erzeuger_{i+1}_nach"] = Powerusage_nach

    print(Power_df_vor)
    print(Power_df_nach)

    # create df of the CO2 Emissions
    CO2_df_vor = pd.DataFrame(index=df_input.index)
    CO2_df_nach = pd.DataFrame(index=df_input.index)

    CO2_df_vor.index = df_input.index
    for i, erzeuger in enumerate(erzeugerpark):
        CO2_vor = [
            erzeuger.calc_co2_emissions(
                Power_df_vor.loc[hour, f"Erzeuger_{i+1}_vor"],
            )
            for hour in df_input.index
        ]
        CO2_df_vor[f"Erzeuger_{i+1}_vor"] = CO2_vor

    CO2_df_nach.index = df_input.index
    for i, erzeuger in enumerate(erzeugerpark):
        CO2_nach = [
            erzeuger.calc_co2_emissions(
                Power_df_nach.loc[hour, f"Erzeuger_{i+1}_nach"],
            )
            for hour in df_input.index
        ]
        CO2_df_nach[f"Erzeuger_{i+1}_nach"] = CO2_nach

    actual_production_df_vor.to_json(
        "results/actual_production_df_vor.json", orient="columns"
    )
    actual_production_df_nach.to_json(
        "results/actual_production_df_ncoach.json", orient="columns"
    )

    Power_df_vor.to_json("results/Power_df_vor.json", orient="columns")
    Power_df_nach.to_json("results/Power_df_nach.json", orient="columns")

    CO2_df_vor.to_json("results/CO2_df_vor.json", orient="columns")
    CO2_df_nach.to_json("results/CO2_df_nach.json", orient="columns")

    st.subheader("Numerische Ergebnisse")
    total_sum = round(actual_production_df_vor.sum().sum() / 1000)
    total_sum2 = round(actual_production_df_nach.sum().sum() / 1000)
    st.write(f"Die gesamte Wärme Produktion:{total_sum} MWh")

    total_sum_power = round(Power_df_vor.sum().sum() / 1000)
    total_sum_power2 = round(Power_df_nach.sum().sum() / 1000)
    st.write(f"Die gesamte Strom Nutzung vor T-Absenkung:     {total_sum_power} MWh")
    st.write(f"Die gesamte Strom Nutzung nach T-Absenkung:    {total_sum_power2} MWh")

    total_sum_co2 = round(CO2_df_vor.sum().sum() / 1000)
    total_sum_co22 = round(CO2_df_nach.sum().sum() / 1000)
    st.write(f"Die gesamte CO2 Emission vor T-Absenkung:     {total_sum_co2} kg")
    st.write(f"Die gesamte CO2 Emission nach T-Absenkung:    {total_sum_co22} kg")

    # Export a DataFrame with the COP of the Heat Pumps as json, in oder to use FlixOpt for Storageoptimization
    COP_df = pd.DataFrame(index=df_input.index)
    COP_df.index = df_input.index
    for i, erzeuger in enumerate(erzeugerpark):
        # Check the type of the 'erzeuger'
        if isinstance(erzeuger, (ep.Waermepumpe1, ep.Waermepumpe2)):
            COP = [
                erzeuger.calc_COP(
                    df_results.loc[hour, "T_vl_vor"],
                )
                for hour in df_input.index
            ]
            COP_df[f"Erzeuger_{i+1}_vor"] = COP

    COP_df.to_json("results/COP_vor_df.json", orient="columns")

    # Define color list
    color_FFE = [erzeuger.color for erzeuger in erzeugerpark]

    # Save color_FFE as a JSON
    with open("results/color_FFE.json", "w") as f:
        json.dump(color_FFE, f)

    # Create and sort sorted_df before plotting it
    sorted_df = actual_production_df_vor.copy()
    for col in sorted_df.columns:
        sorted_df[col] = sorted_df[col].sort_values(ascending=False).values

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

    plot_power_usage(Power_df_vor, Power_df_nach, color_FFE)

    plot_total_change(
        actual_production_df_vor,
        actual_production_df_nach,
        color_FFE,
        "vor",
        "nach",
        "Erzeuger",
        "Change in Production",
        "Erzeuger",
        "Total Production [kWh]",
    )

    plot_total_change(
        CO2_df_vor,
        CO2_df_nach,
        color_FFE,
        "vor",
        "nach",
        "Erzeuger",
        "Change in CO2 Emissions",
        "Erzeuger",
        "Total Emissions [kg CO2]",
    )

    plot_total_change(
        Power_df_vor,
        Power_df_nach,
        color_FFE,
        "vor",
        "nach",
        "Erzeuger",
        "Change in Power Usage",
        "Erzeuger",
        "Total Usage [kWh]",
    )

    st.sidebar.success("Simulation erfolgreich")
