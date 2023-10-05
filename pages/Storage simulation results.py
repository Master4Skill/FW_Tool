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
from plotting_functions import (
    plot_power_usage_storage,
    plot_actual_production,
    plot_sorted_production,
)
import seaborn as sns
import matplotlib.patches as mpatches
import logging
import math

logging.getLogger("matplotlib.font_manager").disabled = True


with open("results/data.json", "r") as f:
    input_data = json.load(f)


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
# from pages.Erzeugerpark import names, erzeuger_df_vor
file_path = "results/COP_vor_df.json"

# Lesen der JSON-Datei in einen DataFrame
COP_df_imported = pd.read_json(file_path, orient="columns")
COP_df_imported.fillna(COP_df_imported.mean(), inplace=True)

with open("results/df_results.json", "r") as f:
    df_results = json.load(f)

# T_vl_vor = df_results["T_vl_vor"]
T_vl_vor = list(df_results["T_vl_vor"].values())
# Flusstemperatur = list(df_results["Flusstemperatur"].values())

st.set_page_config(page_title="Plotting Demo2", page_icon="📈")
add_logo("resized_image.png")
st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")

# Don't forget to pass a proper DataFrame as an argument when calling the function.
# plot_data3(your_dataframe)
font_properties = {
    "fontsize": 16,
    "color": "#777777",
    "fontfamily": "Segoe UI SemiLight",
}

title_properties = {
    "fontsize": 16,
    "color": "#777777",
    "fontfamily": "Segoe UI SemiLight",
}

legend_properties = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, -0.15),
    "ncol": 2,  # Stack legend items vertically
    "frameon": False,  # Make frame visible
    "fontsize": 16,
    "labelcolor": "#777777",
}

x = 1


def plot_char_values_comparison(df_results, title):
    df_results["Value"] = df_results["Value"] / 100000
    # Define labels, titles, etc.
    title = title
    x_label = " "
    y_label = "Heat Production [in 10²·MWh]"

    # Define color palette, font color, and font family
    palette = ["#1F4E79", "#356CA5", "#8AB5E1"]  # Adjust as necessary
    font_color = "#777777"
    font_family = "Segoe UI SemiLight"

    # Plotting with seaborn
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(
        x="Category",
        y="Value",
        hue="Mode",
        data=df_results,
        palette=palette,
    )

    # Applying the custom styles
    bar_plot.set_xlabel(x_label, fontsize=16, color=font_color, fontfamily=font_family)
    bar_plot.set_ylabel(y_label, fontsize=16, color=font_color, fontfamily=font_family)
    bar_plot.set_title(title, fontsize=16, color=font_color, fontfamily=font_family)

    # Set the tick parameters
    bar_plot.tick_params(axis="both", which="major", labelsize=16, colors=font_color)
    plt.xticks(rotation=45)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=3,  # Adjust as necessary
        frameon=False,
        fontsize=16,
        title_fontsize="16",
        labelcolor=font_color,
    )

    # Set the color and width of the spines
    for spine in ["bottom", "left"]:
        bar_plot.spines[spine].set_edgecolor("#A3A3A3")
        bar_plot.spines[spine].set_linewidth(1)

    # Hide the top and right spines
    for spine in ["top", "right"]:
        bar_plot.spines[spine].set_visible(False)

    # Set the background color
    bar_plot.set_facecolor("white")
    # Annotate bars with absolute numbers

    # Calculate percentages based on the highest value in each category
    max_values = df_results.groupby("Category")["Value"].max()
    percentages = [
        (value / max_values[category]) * 100
        for _, (value, category) in df_results[["Value", "Category"]].iterrows()
    ]

    # Rearrange the list
    rearranged_percentages = []
    for i in range(3):  # Assuming there are 3 modes
        rearranged_percentages.extend(percentages[i::3])

    # Annotate bars with calculated percentages
    # for p, percentage in zip(bar_plot.patches, rearranged_percentages):

    for p in bar_plot.patches:
        height = p.get_height()
        bar_plot.annotate(
            # f"{percentage:.0f}%",  # Annotate with integer percentage
            f"{height:.0f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=12,  # Adjust fontsize as needed
            color="#777777",  # Adjust color as needed
        )

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    configure_style(bar_plot)
    return


def configure_style(ax):
    ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["bottom"].set_linewidth(1)

    ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_linewidth(1)

    ax.xaxis.label.set_color("#A3A3A3")
    ax.yaxis.label.set_color("#A3A3A3")

    ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")
    # ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_comparison_old(df_melted, title, ylabel, percentage_mode):
    plt.figure(figsize=(10, 6))
    palette = ["#1F4E79", "#356CA5", "#8AB5E1", "#FEC05C"]
    ax = sns.barplot(
        x="variable", y="value", hue="Mode", data=df_melted, palette=palette
    )
    plt.xlabel("", **font_properties)
    plt.ylabel(ylabel, **font_properties)
    plt.title(title, **title_properties)
    plt.xticks(rotation=45)
    # plt.ylim(0, 22000000)

    # Calculate percentages based on the highest value in each category
    max_values = df_melted.groupby("variable")["value"].max()
    percentages = [
        (value / max_values[variable]) * 100
        for _, (value, variable) in df_melted[["value", "variable"]].iterrows()
    ]
    # Calculate percentages based on the highest value across all categories

    # max_value = df_melted["value"].max()
    max_value = df_melted["value"].iloc[0]
    percentages2 = [(value / max_value) * 100 for value in df_melted["value"]]

    # Rearrange the list
    rearranged_percentages = []

    # Annotate bars with calculated percentages
    if percentage_mode == "global":
        for i in range(4):
            rearranged_percentages.extend(percentages2[i::4])
    elif percentage_mode == "local":
        for i in range(3):
            rearranged_percentages.extend(percentages[i::3])

    for p, percentage in zip(ax.patches, rearranged_percentages):
        height = p.get_height()
        ax.annotate(
            f"{percentage:.0f}%",  # Annotate with integer percentage
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=12,
            color="#777777",
        )

    # Adapt legend style
    leg = plt.legend(**legend_properties)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=2,  # Adjust as necessary
        frameon=False,
        fontsize=16,
        title_fontsize="16",
        labelcolor="#777777",
    )
    for text in leg.get_texts():
        plt.setp(text, color="#777777")

    # Adapt spine and tick colors
    ax.tick_params(axis="x", colors="#A3A3A3")
    ax.tick_params(axis="y", colors="#A3A3A3")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.xaxis.label.set_color("#A3A3A3")
    ax.yaxis.label.set_color("#A3A3A3")
    configure_style(ax)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    st.pyplot(plt.gcf())
    return

    # Example DataFrame


def plot_comparison(df_melted, title, ylabel, percentage_mode):
    df_melted = df_melted.applymap(
        lambda x: x / 1000 if pd.api.types.is_number(x) else x
    )

    plt.figure(figsize=(10, 6))
    palette = ["#1F4E79", "#356CA5", "#8AB5E1", "#FEC05C"]
    ax = sns.barplot(
        x="variable", y="value", hue="Mode", data=df_melted, palette=palette
    )
    plt.xlabel("", **font_properties)
    plt.ylabel(ylabel, **font_properties)
    plt.title(title, **title_properties)
    plt.xticks(rotation=45)
    # plt.ylim(0, 22000)
    # Adapt legend style
    leg = plt.legend(**legend_properties)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=2,  # Adjust as necessary
        frameon=False,
        fontsize=16,
        title_fontsize="16",
        labelcolor="#777777",
    )
    for text in leg.get_texts():
        plt.setp(text, color="#777777")

    # Adapt spine and tick colors
    ax.tick_params(axis="x", colors="#A3A3A3")
    ax.tick_params(axis="y", colors="#A3A3A3")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.xaxis.label.set_color("#A3A3A3")
    ax.yaxis.label.set_color("#A3A3A3")
    configure_style(ax)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            # f"{percentage:.0f}%",  # Annotate with integer percentage
            f"{height:.0f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=12,  # Adjust fontsize as needed
            color="#777777",  # Adjust color as needed
        )

    st.pyplot(plt.gcf())
    return


def plot_comparison2(df_melted, title, percentage_mode):
    df_melted = df_melted.applymap(
        lambda x: x / 1000 if pd.api.types.is_number(x) else x
    )

    plt.figure(figsize=(10, 6))
    palette = ["#1F4E79", "#356CA5", "#8AB5E1"]
    ax = sns.barplot(
        x="variable", y="value", hue="Mode", data=df_melted, palette=palette
    )
    plt.xlabel("", **font_properties)
    plt.ylabel("tsd. €/MWh/10⁵·g CO₂", **font_properties)
    plt.title(title, **title_properties)
    plt.xticks(rotation=45)
    plt.ylim(0, 7500)

    # Adapt legend style
    leg = plt.legend(**legend_properties)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=3,  # Adjust as necessary
        frameon=False,
        fontsize=16,
        title_fontsize="16",
        labelcolor="#777777",
    )
    for text in leg.get_texts():
        plt.setp(text, color="#777777")

    # Adapt spine and tick colors
    ax.tick_params(axis="x", colors="#A3A3A3")
    ax.tick_params(axis="y", colors="#A3A3A3")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.xaxis.label.set_color("#A3A3A3")
    ax.yaxis.label.set_color("#A3A3A3")
    configure_style(ax)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            # f"{percentage:.0f}%",  # Annotate with integer percentage
            f"{height:.0f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=12,  # Adjust fontsize as needed
            color="#777777",  # Adjust color as needed
        )

    st.pyplot(plt.gcf())
    return


if st.button("Submit"):
    data = {
        "Category": [
            # "Stored Energy",
            # "Stored Energy",
            # "Stored Energy",
            "Waste Heat",
            "Waste Heat",
            "Waste Heat",
            "Heat Pump",
            "Heat Pump",
            "Heat Pump",
            "Geothermal",
            "Geothermal",
            "Geothermal",
            "Solar Thermal",
            "Solar Thermal",
            "Solar Thermal",
            "PLB",
            "PLB",
            "PLB",
        ],
        "Value": [
            # 0,
            #  8362774,
            #  8639260,
            5680000,
            5742986,
            5806461,
            8589566,
            8446942,
            6512327,
            11049200,
            11289913,
            11260068,
            2538100,
            3067500,
            3029031,
            5845000,
            5322816,
            7266945,
        ],
        "Mode": [
            #  "no storage",
            # "avoid PLB",
            # "optimized",
            "no storage",
            "avoid PLB",
            "optimized",
            "no storage",
            "avoid PLB",
            "optimized",
            "no storage",
            "avoid PLB",
            "optimized",
            "no storage",
            "avoid PLB",
            "optimized",
            "no storage",
            "avoid PLB",
            "optimized",
        ],
    }
    df_results = pd.DataFrame(data)

    # Call the function
    plot_char_values_comparison(
        df_results, "Heat Production before the Temperature Reduction cost optimized"
    )

    # Example DataFrame
    data = {
        "Category": [
            #    "Stored Energy",
            #    "Stored Energy",
            #   "Stored Energy",
            "Waste Heat",
            "Waste Heat",
            "Waste Heat",
            "Heat Pump",
            "Heat Pump",
            "Heat Pump",
            "Geothermal",
            "Geothermal",
            "Geothermal",
            "Solar Thermal",
            "Solar Thermal",
            "Solar Thermal",
            "PLB",
            "PLB",
            "PLB",
        ],
        "Value": [
            #   0.00,
            #   8483596.18,
            #   10169576.76,
            5680021.64,
            5757312.58,
            5806461.97,
            8574309.95,
            8441340.94,
            8419549.40,
            11081253.24,
            11299262.71,
            11296422.38,
            2521404.87,
            3050589.33,
            3025382.59,
            5845060.70,
            5323216.77,
            5357625.59,
        ],
        "Mode": [
            #   "no storage",
            #   "avoid PLB",
            #  "optimized",
            "no storage",
            "avoid PLB",
            "optimized",
            "no storage",
            "avoid PLB",
            "optimized",
            "no storage",
            "avoid PLB",
            "optimized",
            "no storage",
            "avoid PLB",
            "optimized",
            "no storage",
            "avoid PLB",
            "optimized",
        ],
    }
    df_results = pd.DataFrame(data)

    # Call the function
    plot_char_values_comparison(
        df_results, "Heat Production after the Temperature Reduction cost optimized"
    )

    import pandas as pd

    # Example data from "before Temperature reduction" table
    data_before = {
        "Category": [
            "Stored Energy",
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "Peak Load Boiler",
        ]
        * 3,  # Each category repeated 3 times for 3 modes
        "Value": [
            0,
            5680334,
            8532500,
            11079560,
            2544056,
            5865598,
            8327406,
            5787330,
            8430616,
            11289658,
            3026637,
            5334356,
            8327406,
            5787330,
            8430616,
            11289658,
            3026637,
            5334356,
        ],
        "Mode": [
            "no Storage",
            "no Storage",
            "no Storage",
            "no Storage",
            "no Storage",
            "no Storage",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "optimized",
            "optimized",
            "optimized",
            "optimized",
            "optimized",
            "optimized",
        ],
    }

    # Convert to DataFrame
    df_before = pd.DataFrame(data_before)

    # Example data from "after Temperature reduction" table
    data_after = {
        "Category": [
            "Stored Energy",
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "Peak Load Boiler",
        ]
        * 3,
        "Value": [
            0,
            5680334,
            8535240,
            11099680,
            2528924,
            5857871,
            8504208,
            5787330,
            8438501,
            11300393,
            3021597,
            5324312,
            8504208,
            5787330,
            8438501,
            11300393,
            3021597,
            5324312,
        ],
        "Mode": [
            "no Storage",
            "no Storage",
            "no Storage",
            "no Storage",
            "no Storage",
            "no Storage",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "optimized",
            "optimized",
            "optimized",
            "optimized",
            "optimized",
            "optimized",
        ],
    }

    # Convert to DataFrame
    df_after = pd.DataFrame(data_after)

    # Assuming plot_char_values_comparison is defined somewhere, you would call:
    plot_char_values_comparison(
        df_before, "Heat Production before the Temperature Reduction emission optimized"
    )
    plot_char_values_comparison(
        df_after, "Heat Production after the Temperature Reduction emission optimized"
    )

    data1 = {  # high temp
        "Mode": ["no Storage", "avoid PLB", "optimized"],
        "Production Costs": [
            19570562.35,  # No Storage
            19072972.41,  # PLB Avoid
            18017060.17,  # Optimized
        ],
        "Stored Energy": [
            0.0,  # No Storage
            8362774.33,  # PLB Avoid
            8639260.98,  # Optimized
        ],
        "Emissions": [
            4263213,  # No Storage
            4136768,  # PLB Avoid
            5141182,  # Optimized
        ],
    }

    data2 = {  # low temp
        "Mode": ["no Storage", "avoid PLB", "optimized"],
        "Production Costs": [
            15137308.55,  # No Storage
            14608701.64,  # PLB Avoid
            13837200.18,  # Optimized
        ],
        "Stored Energy": [
            0.0,  # No Storage
            8483596.18,  # PLB Avoid
            10169576.76,  # Optimized
        ],
        "Emissions": [
            3323432,  # No Storage
            3193394,  # PLB Avoid
            4087538,  # Optimized
        ],
    }

    data3 = {  # only costs
        "Mode": ["before ", "avoid PLB", "optimized"],
        "Costs before cost opt": [
            19570562.35,  # No Storage
            19072972.41,  # PLB Avoid
            18017060.17,  # Optimized
        ],
        "Costs after cost opt": [
            15137308.55,  # No Storage
            14608701.64,  # PLB Avoid
            13837200.18,  # Optimized
        ],
        "Costs before emi-opt": [
            19556281.27,  # No Storage
            19068023.35,  # PLB Avoid
            19068023.35,  # Optimized
        ],
        "Costs after emi-opt": [
            15130807.71,  # No Storage
            14608442.16,  # PLB Avoid
            14608442.16,  # Optimized
        ],
    }
    data5 = {  # only costs
        "Mode": ["No Storage ", "Avoid PLB", "Cost Optimized", "Emission Optimized"],
        "before Temp.\n Reduction": [
            19570562.35,  # No Storage
            19072972.41,  # PLB Avoid
            18017060.17,  # Optimized
            19068023.35,  # Optimized
        ],
        "after Temp.\n Reduction": [
            15137308.55,  # No Storage
            14608701.64,  # PLB Avoid
            13837200.18,  # Optimized
            14608442.16,  # Optimized
        ],
    }
    data4 = {  # only costs
        "Mode": ["No Storage ", "Avoid PLB", "Cost Optimized", "Emission Optimized"],
        "before Temp.\n Reduction": [
            4263213,  # No Storage
            4136768,  # PLB Avoid
            5141182,  # Optimized
            4136339,  # Optimized
        ],
        "after Temp.\n Reduction": [
            3323432,  # No Storage
            3193394,  # PLB Avoid
            4087538,  # Optimized
            3193247,  # Optimized
        ],
    }

    # Modified data from "BA Speicher values high temp" with switched positions
    data_high_temp = {
        "Mode": ["no Storage", "PLB avoid", "optimized"],
        "Production Costs": [
            19556281.27,  # No Storage
            19068023.35,  # PLB Avoid
            19068023.35,  # Optimized
        ],
        "Stored Energy": [
            0.0,  # No Storage
            8327406.020674081,  # PLB Avoid
            8327406.020674081,  # Optimized
        ],
        "Emissions": [
            4261210,  # No Storage
            4136339,  # PLB Avoid
            4136339,  # Optimized
        ],
    }

    # Modified data from "BA Speicher values low temp" with switched positions
    data_low_temp = {
        "Mode": ["no Storage", "PLB avoid", "optimized"],
        "Production Costs": [
            15130807.71,  # No Storage
            14608442.16,  # PLB Avoid
            14608442.16,  # Optimized
        ],
        "Stored Energy": [
            0.0,  # No Storage
            8504208.535650084,  # PLB Avoid
            8504208.535650084,  # Optimized
        ],
        "Emissions": [
            3322645,  # No Storage
            3193247,  # PLB Avoid
            3193247,  # Optimized
        ],
    }

    # Convert to DataFrame
    df_high_temp = pd.DataFrame(data_high_temp)
    df_low_temp = pd.DataFrame(data_low_temp)

    # Melt the DataFrames
    df_melted_high_temp = df_high_temp.melt(
        id_vars="Mode", value_vars=["Production Costs", "Stored Energy", "Emissions"]
    )
    df_melted_low_temp = df_low_temp.melt(
        id_vars="Mode", value_vars=["Production Costs", "Stored Energy", "Emissions"]
    )

    df = pd.DataFrame(data1)
    df_melted1 = df.melt(
        id_vars="Mode", value_vars=["Production Costs", "Stored Energy", "Emissions"]
    )

    df = pd.DataFrame(data2)
    df_melted2 = df.melt(
        id_vars="Mode", value_vars=["Production Costs", "Stored Energy", "Emissions"]
    )

    df = pd.DataFrame(data3)
    df_melted3 = df.melt(
        id_vars="Mode",
        value_vars=[
            "Costs before cost opt",
            "Costs after cost opt",
            "Costs before emi-opt",
            "Costs after emi-opt",
        ],
    )
    df = pd.DataFrame(data5)
    df_melted5 = df.melt(
        id_vars="Mode",
        value_vars=[
            "before Temp.\n Reduction",
            "after Temp.\n Reduction",
        ],
    )
    df = pd.DataFrame(data4)
    df_melted4 = df.melt(
        id_vars="Mode",
        value_vars=[
            "before Temp.\n Reduction",
            "after Temp.\n Reduction",
        ],
    )

    plot_comparison(
        df_melted5,
        "Heat Production Costs",
        "Costs [tsd. €]",
        percentage_mode="global",
    )
    plot_comparison_old(
        df_melted5,
        "Heat Production Costs",
        "Costs [€]",
        percentage_mode="global",
    )
    plot_comparison(
        df_melted4,
        "Heat Production Emissions",
        "Emissions [kg CO₂]",
        percentage_mode="global",
    )
    plot_comparison_old(
        df_melted4,
        "Heat Production Emissions",
        "Emissions [g CO₂]",
        percentage_mode="global",
    )

    plot_comparison(
        df_melted1,
        "Comparison of Operating Modes at original temperatures cost optimized",
        percentage_mode="local",
    )
    plot_comparison(
        df_melted2,
        "Comparison of Operating Modes at reduced temperatures cost optimized",
        percentage_mode="local",
    )

    # Assuming plot_comparison is defined somewhere, you would call:
    plot_comparison(
        df_melted_high_temp,
        "Comparison of Operating Modes at High Temperature emission optimized",
        percentage_mode="local",
    )
    plot_comparison(
        df_melted_low_temp,
        "Comparison of Operating Modes at Low Temperature emission optimized",
        percentage_mode="local",
    )

    plot_comparison(
        df_melted3,
        "Heat Production Costs",
        percentage_mode="global",
    )

    plot_comparison2(
        df_melted4,
        "Heat Production Emissions",
        percentage_mode="global",
    )
