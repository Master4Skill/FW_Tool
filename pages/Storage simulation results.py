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
            #  "No storage",
            # "Avoid PLB",
            # "Optimized",
            "No storage",
            "Avoid PLB",
            "Optimized",
            "No storage",
            "Avoid PLB",
            "Optimized",
            "No storage",
            "Avoid PLB",
            "Optimized",
            "No storage",
            "Avoid PLB",
            "Optimized",
            "No storage",
            "Avoid PLB",
            "Optimized",
        ],
    }
    df_results = pd.DataFrame(data)
    # Call the function
    # plot_char_values_comparison(df_results, "Heat Generation before the Temperature Reduction Cost Optimized")

    data = {  # high temp, heat pump2, with emission mode
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
            5680334.8929,
            10404008.4005,
            11033953.0234,
            2542281.6065,
            4041472.4766,
            5769388.4565,
            10109952.1803,
            11293256.5945,
            3048652.0883,
            3644837.609,
            5807737.4171,
            6799374.2404,
            11295159.8072,
            3031289.3835,
            6945737.5658,
            5769388.4565,
            10109952.1803,
            11293256.5945,
            3048652.0883,
            3644837.609,
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
            "Emission",
            "Emission",
            "Emission",
            "Emission",
            "Emission",
        ],
    }

    df_results = pd.DataFrame(data)
    # Call the function
    plot_char_values_comparison(
        df_results, "Heat Generation before the Temperature Reduction"
    )
    # Additional code for plotting (use as per your requirement)

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
            5680334.8929,
            10404008.4005,
            11033953.0234,
            2542281.6065,
            4041472.4766,
            5769388.4565,
            10109952.1803,
            11293256.5945,
            3048652.0883,
            3644837.609,
            5807737.4171,
            6799374.2404,
            11295159.8072,
            3031289.3835,
            6945737.5658,
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

    # Example DataFrame
    data = {
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
            6978120.3423,
            8627075.946,
            10811127.0739,
            2909733.5161,
            4375993.5217,
            7240996.1054,
            9297714.2773,
            11005738.3309,
            3283257.2996,
            3024014.9014,
            7247829.2837,
            8275386.1462,
            11010023.5804,
            3275001.9973,
            4047421.9206,
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
    # Example DataFrame
    data = {
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
            6978120.3423,
            8627075.946,
            10811127.0739,
            2909733.5161,
            4375993.5217,
            7240996.1054,
            9297714.2773,
            11005738.3309,
            3283257.2996,
            3024014.9014,
            7247829.2837,
            8275386.1462,
            11010023.5804,
            3275001.9973,
            4047421.9206,
            7221179.5228,
            9264088.2893,
            11007367.8469,
            3302641.2544,
            3048631.0693,
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
            "Emission",
            "Emission",
            "Emission",
            "Emission",
            "Emission",
        ],
    }
    df_results = pd.DataFrame(data)

    # Call the function
    plot_char_values_comparison(
        df_results, "Heat Generation after the Temperature Reduction"
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
            "No Storage",
            "No Storage",
            "No Storage",
            "No Storage",
            "No Storage",
            "No Storage",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
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
            "No Storage",
            "No Storage",
            "No Storage",
            "No Storage",
            "No Storage",
            "No Storage",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "PLB avoid",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
            "Optimized",
        ],
    }

    # Convert to DataFrame
    df_after = pd.DataFrame(data_after)

    # Assuming plot_char_values_comparison is defined somewhere, you would call:
    # plot_char_values_comparison(df_before, "Heat Generation before the Temperature Reduction Emission Optimized")
    # plot_char_values_comparison(df_after, "Heat Generation after the Temperature Reduction Emission Optimized")

    data5 = {  # only costs heat pump 1
        "Mode": ["No Storage ", "Avoid PLB", "Cost Optimized", "Emission Optimized"],
        "before Temp.\n Reduction": [
            19570562.35,  # No Storage
            19072972.41,  # PLB Avoid
            18734300.17,  # Optimized
            19068023.35,  # emission Optimized
        ],
        "after Temp.\n Reduction": [
            15137308.55,  # No Storage
            14608701.64,  # PLB Avoid
            14554351.63,  # 13837200.18,  # Optimized
            14608442.16,  # emission Optimized
        ],
    }
    data4 = {  # only emissions heat pump 1
        "Mode": ["No Storage ", "Avoid PLB", "Cost Optimized", "Emission Optimized"],
        "before Temp.\n Reduction": [
            4263213,  # No Storage
            4136768,  # PLB Avoid
            4518870,  # Optimized
            4136339,  # emission Optimized
        ],
        "after Temp.\n Reduction": [
            3323432,  # No Storage
            3193394,  # PLB Avoid
            3465210,  # Optimized
            3193247,  # emission Optimized
        ],
    }

    data51 = {  # only costs heat pump 2
        "Mode": [
            "No Storage ",
            "Avoid PLB",
            "Emission Optimized",
            "Cost Optimized",
        ],
        "before Temp.\n Reduction": [
            2412533.79,  # No Storage
            2350658.56,  # PLB Avoid  2350777.91
            2329541.37,  # emission Optimized 2329542.21
            2268594.59,  # Optimized
        ],
        "after Temp.\n Reduction": [
            1723795.01,  # No Storage
            1666798.07,  # PLB Avoid
            1666063.89,  # emission Optimized
            1653304.62,  # Optimized
        ],
    }
    data41 = {  # only emissions heat pump 2
        "Mode": ["No Storage ", "Avoid PLB", "Cost Optimized", "Emission Optimized"],
        "before Temp.\n Reduction": [
            4634.58,  # No Storage
            4504.04,  # PLB Avoid
            4523.51,  # Optimized
            4494.04,  # emission Optimized
        ],
        "after Temp.\n Reduction": [
            3365.81,  # No Storage
            3188.82,  # PLB Avoid
            3238.38,  # Optimized
            3188.33,  # emission Optimized
        ],
    }

    data5 = {  # only costs heat pump 2
        "Mode": [
            "No Storage ",
            "Avoid PLB",
            "Cost Optimized",
            "Emission Optimized",
        ],
        "before Temp.\n Reduction": [
            2412533.79,  # No Storage
            2350777.56,  # PLB Avoid  2350777.91
            2268594.59,  # Optimized
            2329541.37,  # emission Optimized 2329542.21
        ],
        "after Temp.\n Reduction": [
            1723795.01,  # No Storage
            1666798.07,  # PLB Avoid
            1666063.89,  # emission Optimized
            1653304.62,  # Optimized
        ],
    }

    data4 = {  # only emissions heat pump 2
        "Mode": ["No Storage ", "Avoid PLB", "Cost Optimized", "Emission Optimized"],
        "before Temp.\n Reduction": [
            4634.58,  # No Storage
            4504.04,  # PLB Avoid
            4523.51,  # Optimized
            4493.98,  # emission Optimized
        ],
        "after Temp.\n Reduction": [
            3365.81,  # No Storage
            3188.82,  # PLB Avoid
            3238.38,  # Optimized
            3188.33,  # emission Optimized
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

    st.write("## old results")
    plot_comparison(
        df_melted5,
        "Heat Generation Costs",
        "Costs [tsd. €]",
        "global",
        1000,
    )
    plot_comparison(
        df_melted4,
        "Heat Generation Emissions",
        "Emissions [t CO₂]",
        "global",
        1,
    )
