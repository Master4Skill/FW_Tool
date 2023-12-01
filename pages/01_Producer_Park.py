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
from streamlit_extras.stoggle import stoggle
import pickle

with open("results/variables.json", "r") as f:
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

# Load the CSV file
data = pd.read_csv(
    "Zeitreihen/zeitreihen_22.csv", delimiter=";", dtype={"Einstrahlung_22": str}
)

# Convert the 'Einstrahlung_22' column from strings with commas to floats
data["Einstrahlung_22"] = data["Einstrahlung_22"].str.replace(",", ".").astype(float)

# Get the 'Einstrahlung_22' column
irradiation_data = data["Einstrahlung_22"]


color_dict = {
    "♨️ waste heat - limited volume flow of the source (m³/h)": "#639729",
    "🚰 waste heat pump - limited volume flow of the source (m³/h), source temperature constant (°C)": "#1F4E79",
    "🌊 ambient heat pump - limited power (kW), fluctuating source temperature (°C)": "#F7D507",
    "⛰️ geothermal  - maximum power (kW)": "#DD2525",
    "☀️ solarthermal - solar radiation (kW/m²)": "#92D050",
    "🔥 PLB - maximum power (kW)": "#EC9302",
    "🏭 CHP - maximum power (kW)": "#F7D507",
}


st.set_page_config(
    page_title="Producer Park",
    page_icon="🏭",
)
add_logo("resized_image.png")

st.sidebar.header("Temperature Reduction in the Producer Park")

st.sidebar.info("Select the producers and their parameters")

st.markdown("# Producer Park Simulation")

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
erzeugerpark_dict = {}


# Maximum Number of Generator Inputs
max_erzeuger = 7
anzahl_erzeuger = st.slider(
    "How many generators would you like to add?", 1, max_erzeuger
)
expander = st.expander("Additional Solar Parameters")

for i in range(anzahl_erzeuger):
    st.markdown(f"### Generator {i+1}")
    erzeuger_type = st.selectbox(
        f"Please select the type of the generator {i+1}",
        [
            "♨️ waste heat - limited volume flow of the source (m³/h)",
            "🚰 waste heat pump - limited volume flow of the source (m³/h), source temperature constant (°C)",
            "🌊 ambient heat pump - limited power (kW), fluctuating source temperature (°C)",
            "⛰️ geothermal  - maximum power (kW)",
            "☀️ solarthermal - solar radiation (kW/m²)",
            "🔥 PLB - maximum power (kW)",
            "🏭 CHP - maximum power (kW)",
        ],
        key=f"Erzeuger{i}",
    )

    erzeuger_color = color_dict[erzeuger_type]

    if "♨️ waste heat" in erzeuger_type:
        Volumenstrom_quelle = st.number_input(
            "Please enter the volume flow rate of the source ( )",
            value=10,
            key=f"Volumenstrom_quelle{i}",
        )
        Abwärmetemperatur = st.number_input(
            "Please enter the source temperature (°C)",
            value=120,
            key=f"Quelltemperatur{i}",
        )
        Partload = (
            st.number_input(
                "Please enter the partload of the waste heat unit (%)",
                value=0,
                key=f"Partload{i}",
            )
            / 100
        )
        erzeuger = ep.waste_heat(
            Volumenstrom_quelle,
            Abwärmetemperatur,
            Partload,
            color=erzeuger_color,
            co2_emission_factor=0,
        )

    elif "🚰 waste heat pump" in erzeuger_type:
        Volumenstrom_quelle = st.number_input(
            "Please enter the volume flow rate of the source (m³/h)",
            value=200,
            key=f"Volumenstrom_quelle{i}",
        )
        T_q = st.number_input(
            "Please enter the source temperature (°C)",
            value=25,
            key=f"Quelltemperatur{i}",
        )
        Gütegrad = st.number_input(
            "Please specify the efficiency of the water-water heat pump",
            value=0.45,
            key=f"Gütegrad{i}",
        )
        Partload = (
            st.number_input(
                "Please enter the partload of the waste heat pump (%)",
                value=40,
                key=f"Partload{i}",
            )
            / 100
        )

        erzeuger = ep.heatpump_1(
            Volumenstrom_quelle,
            T_q,
            Gütegrad,
            Partload,
            color=erzeuger_color,
            co2_emission_factor=0.468,
        )

    elif "ambient heat pump" in erzeuger_type:
        Leistung_max = st.number_input(
            "Please enter the maximum power (kW)", value=2500, key=f"Leistung_max{i}"
        )
        Gütegrad = st.number_input(
            "Please specify the efficiency of the water-water heat pump",
            value=0.45,
            key=f"Gütegrad{i}",
        )
        Partload = (
            st.number_input(
                "Please enter the partload of the ambient heat pump (%)",
                value=20,
                key=f"Partload{i}",
            )
            / 100
        )

        erzeuger = ep.heatpump_2(
            Leistung_max,
            Gütegrad,
            Partload,
            color=erzeuger_color,
            co2_emission_factor=0.468,
        )

    elif "geothermal" in erzeuger_type:
        Leistung_max = st.number_input(
            "Please enter the maximum power (kW)", value=2000, key=f"Leistung_max{i}"
        )
        Tgeo = st.number_input(
            "Please enter the geothermal temperature (°C)", value=100, key=f"Tgeo{i}"
        )
        h_förder = st.number_input(
            "Please enter the extraction height (m)", value=2000, key=f"h_förder{i}"
        )
        η_geo = st.number_input(
            "Please specify the efficiency of the geothermal pump (%)",
            value=0.8,
            key=f"η_geo{i}",
        )
        Partload = (
            st.number_input(
                "Please enter the partload of the geothermal heat unit (%)",
                value=30,
                key=f"Partload{i}",
            )
            / 100
        )
        erzeuger = ep.geothermal(
            Leistung_max,
            Tgeo,
            h_förder,
            η_geo,
            Partload,
            color=erzeuger_color,
            co2_emission_factor=0.468,
        )

    elif "solarthermal" in erzeuger_type:
        solar_area = st.number_input(
            "Please Enter the total Area of Solarthermal Collectors (m²)",
            value=10000,
            key=f"solar_area{i}",
        )
        Partload = (
            st.number_input(
                "Please enter the partload of the solarthermal collector (%)",
                value=0,
                key=f"Partload{i}",
            )
            / 100
        )

        expander = st.expander("Additional Solar Parameters")
        with expander:
            k_s_1 = st.number_input(
                "Please Enter the first Heat Loss Coefficient of the Solarthermal System (W/m²K)",
                value=1.5,
                format="%.4f",
                key=f"k_s_1{i}",
            )
            k_s_2 = st.number_input(
                "Please Enter the second Heat Loss Coefficient of the Solarthermal System (W/m²K²)",
                value=0.005,
                format="%.4f",
                key=f"k_s_2{i}",
            )

            α = st.number_input(
                "Please Enter the Absorption Coefficient of the Solarthermal System",
                value=0.9,
                key=f"α{i}",
            )

            τ = st.number_input(
                "Please Enter the Transmission Coefficient of the Solarthermal System",
                value=0.9,
                key=f"τ{i}",
            )

        erzeuger = ep.solarthermal(
            solar_area,
            k_s_1 / 1000,
            k_s_2 / 1000,
            α,
            τ,
            Partload,
            color=erzeuger_color,
            co2_emission_factor=0,
        )

    elif "PLB" in erzeuger_type:
        Leistung_max = st.number_input(
            "Please enter the maximum power (kW)", value=10000, key=f"Leistung_max{i}"
        )
        Partload = (
            st.number_input(
                "Please enter the partload of the Peak Load Boiler (%)",
                value=0.1,
                key=f"Partload{i}",
            )
            / 100
        )

        erzeuger = ep.PLB(
            Leistung_max, Partload, color=erzeuger_color, co2_emission_factor=0.201
        )

    elif "CHP" in erzeuger_type:
        Leistung_max = st.number_input(
            "Please enter the maximum power (kW)",
            value=5000,
            key=f"Leistung_max{i}",
        )

        Partload = (
            st.number_input(
                "Please enter the partload of the CHP (%)",
                value=0.1,
                key=f"Partload{i}",
            )
            / 100
        )

        erzeuger = ep.CHP(
            Leistung_max, Partload, color=erzeuger_color, co2_emission_factor=0.201
        )
    else:
        st.write("Please select a valid generator type")

    erzeugerpark.append(erzeuger)

with open("erzeugerpark.pkl", "wb") as file:
    pickle.dump(erzeugerpark_dict, file)


names2 = [obj.__class__.__name__ for obj in erzeugerpark]
teillasten = [obj.Partload for obj in erzeugerpark if hasattr(obj, "Partload")]
# st.write(teillasten)


# my_dict = {f"Erzeuger_{i+1}": name for i, name in enumerate(names2)}

# st.write(my_dict)

name_mapping = {
    "waste_heat": "Waste Heat",
    "heatpump_1": "Waste Heat Pump",
    "heatpump_2": "Ambient\nHeat Pump",
    "solarthermal": "Solar Thermal",
    "geothermal": "Geothermal",
    "PLB": "Peak Load Boiler",
    "CHP": "CHP",
}


names = [
    name_mapping.get(obj.__class__.__name__, obj.__class__.__name__)
    for obj in erzeugerpark
]


my_dict = {f"Erzeuger_{i+1}": name for i, name in enumerate(names)}

# st.write(my_dict)

# print(my_dict)
# print(names)

if st.button("Calculate"):
    # Zeige den Erzeugerpark
    # df_erzeuger = pd.DataFrame([vars(erzeuger) for erzeuger in erzeugerpark])
    # st.dataframe(df_erzeuger)
    # st.write(erzeugerpark)
    df_input = df_input.iloc[:-2]
    # df_input = df_input.reset_index(drop=True)

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

    # st.dataframe(erzeuger_df_vor)
    # st.dataframe(erzeuger_df_nach)

    print(erzeuger_df_vor)

    actual_production_data_vor = []
    actual_production_data_nach = []

    # Iterate over each hour
    for hour in df_input.index:
        # Get the demand for this hour
        # demand_vor = demand_nach = df_input.loc[hour, "Lastgang"]
        demand_vor = df_results.loc[hour, "Wärmelast_vor"]
        demand_nach = df_results.loc[hour, "Wärmelast_nach"]
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

    print(Power_df_vor.iloc[23:])
    print(Power_df_nach.iloc[23:])

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

    # st.dataframe(actual_production_df_vor)
    # st.dataframe(actual_production_df_nach)
    st.subheader("Numerical Results")

    data = {
        "Metric": [
            "Total Heat Generation before Temperature Reduction",
            "Total Heat Generation after Temperature Reduction",
            "Total Power Consumption before Temperature Reduction",
            "Total Power Consumption after Temperature Reduction",
            "Total CO2 Emission before Temperature Reduction",
            "Total CO2 Emission after Temperature Reduction",
        ],
        "Value": [
            f"{round(actual_production_df_vor.sum().sum() / 1000)} MWh",
            f"{round(actual_production_df_nach.sum().sum() / 1000)} MWh",
            f"{round(Power_df_vor.sum().sum() / 1000)} MWh",
            f"{round(Power_df_nach.sum().sum() / 1000)} MWh",
            f"{round(CO2_df_vor.sum().sum() / 1000)} kg",
            f"{round(CO2_df_nach.sum().sum() / 1000)} kg",
        ],
    }

    df_numerical = pd.DataFrame(data)

    st.table(df_numerical)

    # Export a DataFrame with the COP of the Heat Pumps as json, in oder to use FlixOpt for Storageoptimization
    # Pfad zur CSV-Datei
    file_name = "Zeitreihen/zeitreihen_22.csv"

    # Lesen der benötigten Daten aus der CSV-Datei
    df_zeitreihen = pd.read_csv(file_name, sep=";")

    COP_df = pd.DataFrame(index=df_input.index)
    COP_df.index = df_input.index
    for i, erzeuger in enumerate(erzeugerpark):
        # Check the type of the 'erzeuger'
        if isinstance(erzeuger, (ep.heatpump_1, ep.heatpump_2, ep.geothermal)):
            COP = [
                erzeuger.calc_COP(
                    df_results.loc[hour, "T_vl_vor"],
                    Trl_vor,
                    df_zeitreihen.loc[hour, "Isartemp"],
                )
                for hour in df_input.index
            ]
            COP_df[erzeuger.get_class_name()] = COP

    COP_df.to_json("results/COP_vor_df.json", orient="columns")

    for i, erzeuger in enumerate(erzeugerpark):
        # Check the type of the 'erzeuger'
        if isinstance(erzeuger, (ep.heatpump_1, ep.heatpump_2, ep.geothermal)):
            COP = [
                erzeuger.calc_COP(
                    df_results.loc[hour, "T_vl_nach"],
                    Trl_nach,
                    df_zeitreihen.loc[hour, "Isartemp"],
                )
                for hour in df_input.index
            ]
            COP_df[erzeuger.get_class_name()] = COP

    COP_df.to_json("results/COP_nach_df.json", orient="columns")

    # Define color list
    color_FFE = [erzeuger.color for erzeuger in erzeugerpark]

    # Save color_FFE as a JSON
    with open("results/color_FFE.json", "w") as f:
        json.dump(color_FFE, f)

    # Create and sort sorted_df before plotting it
    sorted_df = actual_production_df_vor.copy()
    for col in sorted_df.columns:
        sorted_df[col] = sorted_df[col].sort_values(ascending=False).values

    # st.write(my_dict)

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
    erzeuger_df_vor.fillna(0, inplace=True)
    erzeuger_df_vor_json = erzeuger_df_vor.to_json()
    erzeuger_df_nach.fillna(0, inplace=True)
    erzeuger_df_nach_json = erzeuger_df_nach.to_json()

    # Assuming partloads is a dictionary with producer names as keys and partload numbers as values
    # Example: partloads = {"Erzeuger_1_vor": 100, "Erzeuger_2_vor": 200, ...}

    data1 = {
        "names": names2,
        "partloads": teillasten,  # Moved partloads here
        "erzeuger_df_vor": erzeuger_df_vor_json,
    }

    data2 = {
        "names": names2,
        "partloads": teillasten,  # Moved partloads here
        "erzeuger_df_nach": erzeuger_df_nach_json,
    }

    with open("erzeuger_df_vor.json", "w") as f:
        json.dump(data1, f)

    with open("erzeuger_df_nach.json", "w") as f:
        json.dump(data2, f)
