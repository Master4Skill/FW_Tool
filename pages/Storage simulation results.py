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
    y_label = "Heat Generation [in 10⁵·kWh]"

    # Define color palette, font color, and font family
    palette = ["#1F4E79", "#356CA5", "#8AB5E1", "#FEC05C"]  # Adjust as necessary
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


def plot_comparison_old(df_melted, title, ylabel, percentage_mode, divisor):
    df_melted = df_melted.applymap(
        lambda x: x / divisor if pd.api.types.is_number(x) else x
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


def plot_comparison(df_melted, title, ylabel, percentage_mode, divisor):
    df_melted = df_melted.applymap(
        lambda x: x / divisor if pd.api.types.is_number(x) else x
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
    data = {  # high temp, heat pump2, without emission mode
        "Category": [
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
        ],
        "Value": [
            4963599.2947,
            12413761.7175,
            12329891.6835,
            3774551.9972,
            8252491.5149,
            ##
            5871072.9971,
            12048478.6253,
            12671293.6309,
            5142874.9064,
            6899136.1919,
            ##
            5875842.379,
            7568331.941,
            12615419.0339,
            5150420.9785,
            11231372.2946,
        ],
        "Mode": [
            "No storage",
            "No storage",
            "No storage",
            "No storage",
            "No storage",
            "Avoid PLB",
            "Avoid PLB",
            "Avoid PLB",
            "Avoid PLB",
            "Avoid PLB",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
        ],
    }
    df_results = pd.DataFrame(data)
    # Call the function
    plot_char_values_comparison(
        df_results, "Heat Generation before the Temperature Reduction"
    )

    data = {  # high temp, heat pump2, without emission mode
        "Category": [
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
        ],
        "Value": [
            7677750.8197,
            10428000.917,
            11220187.8331,
            3049222.7196,
            7914887.3317,
            #
            7763151.1286,
            10905040.7905,
            12166872.1073,
            4869593.1039,
            5428433.161,
            #
            7771542.651,
            9277068.643,
            12153896.7949,
            4872591.8097,
            6929277.723,
        ],
        "Mode": [
            "No storage",
            "No storage",
            "No storage",
            "No storage",
            "No storage",
            "Avoid PLB",
            "Avoid PLB",
            "Avoid PLB",
            "Avoid PLB",
            "Avoid PLB",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
        ],
    }

    df_results = pd.DataFrame(data)

    # Call the function
    plot_char_values_comparison(
        df_results, "Heat Generation after the Temperature Reduction"
    )

    data5 = {  # only costs heat pump 2
        "Mode": [
            "No Storage ",
            "Avoid PLB",
            "Cost Optimized",
            "Emission Optimized",
        ],
        "before Temp.\n Reduction": [
            3257902.14,  # No Storage
            3070790.48,  # PLB Avoid
            2973137.75,  # Optimized
            3074885.77,  # emission Optimized
        ],
        "after Temp.\n Reduction": [
            2353617.88,  # No Storage
            2186570.10,  # PLB Avoid
            2160435.67,  # Optimized
            2176467.23,  # emission Optimized
        ],
    }

    data4 = {  # only emissions heat pump 2
        "Mode": ["No Storage ", "Avoid PLB", "Cost Optimized", "Emission Optimized"],
        "before Temp.\n Reduction": [
            6278.38,  # No Storage
            5854.03,  # PLB Avoid
            5911.05,  # Optimized
            5870.00,  # emission Optimized
        ],
        "after Temp.\n Reduction": [
            4593.68,  # No Storage
            4163.98,  # PLB Avoid
            4214.61,  # Optimized
            4150.94,  # emission Optimized
        ],
    }

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

    plot_comparison_old(
        df_melted5, "Heat Generation Costs", "Costs [tsd. €]", "global", 1000
    )
    plot_comparison_old(
        df_melted4, "Heat Generation Emissions", "Emissions [t CO₂]", "global", 1
    )

    data = {  # high temp, heat pump2, without emission mode
        "Category": [
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
        ],
        "Value": [
            4963599.2947,
            12413761.7175,
            12329891.6835,
            3774551.9972,
            8252491.5149,
            ##
            5871072.9971,
            12048478.6253,
            12671293.6309,
            5142874.9064,
            6899136.1919,
            ##
            5875842.379,
            7568331.941,
            12615419.0339,
            5150420.9785,
            11231372.2946,
            #
            5641471.5735,
            9291432.3406,
            12670368.5984,
            5393401.8838,
            9424985.8021,
        ],
        "Mode": [
            "No storage",
            "No storage",
            "No storage",
            "No storage",
            "No storage",
            "Avoid PLB",
            "Avoid PLB",
            "Avoid PLB",
            "Avoid PLB",
            "Avoid PLB",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
            "Emission Optimized",
            "Emission Optimized",
            "Emission Optimized",
            "Emission Optimized",
            "Emission Optimized",
        ],
    }
    df_results = pd.DataFrame(data)
    # Call the function
    plot_char_values_comparison(
        df_results, "Heat Generation before the Temperature Reduction"
    )

    data = {  # high temp, heat pump2, without emission mode
        "Category": [
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
            "Waste Heat",
            "Heat Pump",
            "Geothermal",
            "Solar Thermal",
            "PLB",
        ],
        "Value": [
            7677750.8197,
            10428000.917,
            11220187.8331,
            3049222.7196,
            7914887.3317,
            #
            7763151.1286,
            10905040.7905,
            12166872.1073,
            4869593.1039,
            5428433.161,
            #
            7771542.651,
            9277068.643,
            12153896.7949,
            4872591.8097,
            6929277.723,
            #
            7367567.8796,
            10585221.699,
            12166441.309,
            5274663.5298,
            5618558.8653,
        ],
        "Mode": [
            "No storage",
            "No storage",
            "No storage",
            "No storage",
            "No storage",
            "Avoid PLB",
            "Avoid PLB",
            "Avoid PLB",
            "Avoid PLB",
            "Avoid PLB",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
            "Emission Optimized",
            "Emission Optimized",
            "Emission Optimized",
            "Emission Optimized",
            "Emission Optimized",
        ],
    }

    df_results = pd.DataFrame(data)

    # Call the function
    plot_char_values_comparison(
        df_results, "Heat Generation after the Temperature Reduction"
    )
