import gurobipy as gp
from gurobipy import GRB
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, PULP_CBC_CMD, GUROBI_CMD
import streamlit as st
from streamlit_extras.app_logo import add_logo
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from io import StringIO
import json as json
import numpy as np
import ErzeugerparkClasses as ep
from plotting_functions import (
    plot_power_usage_storage,
    plot_actual_production,
    plot_sorted_production,
)
import seaborn as sns
import matplotlib.patches as mpatches
import logging
import math
import pickle
from pulp import PULP_CBC_CMD

logging.getLogger("matplotlib.font_manager").disabled = True


with open("results/variables.json", "r") as f:
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
ηSpitzenkessel = input_data["ηSpitzenkessel"]
ηBHKW_el = input_data["ηBHKW_el"]
ηBHKW_therm = input_data["ηBHKW_therm"]

# from pages.Erzeugerpark import names, erzeuger_df_vor

# Flusstemperatur = list(df_results["Flusstemperatur"].values())

st.set_page_config(page_title="Plotting Demo2", page_icon="📈")
add_logo("resized_image.png")
st.markdown("# Storage Simulation")
st.sidebar.header("Storage Simulation")

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


def plot_data3(df, title):
    fig, axs = plt.subplots(1, figsize=(10, 6))

    names_map = {
        "g1": "Waste Heat",
        "g2": "Waste\nHeat Pump",
        "g3": "Ambient\nHeat Pump",
        "g4": "Geothermal",
        "g5": "Solar Thermal",
        "g6": "Peak Load Boiler",
        "g71": "CHP",
    }

    # Updated color mapping
    color_dict = {
        "Waste Heat": "#639729",
        "Waste\nHeat Pump": "#1F4E79",
        "Ambient\nHeat Pump": "#DD2525",
        "Geothermal": "#92D050",
        "Solar Thermal": "#EC9302",
        "Peak Load Boiler": "#EC9302",
        "CHP": "#F7D507",  # Assuming the same color for CHP, update if different
    }

    # Define the columns to check
    columns_to_check = ["g1", "g4", "g3", "g2", "g5", "g6", "g71"]
    non_zero_columns = [col for col in columns_to_check if df[col].sum() != 0]

    cumulative = 0
    previous_cumulative = 0
    patches = []  # For shaded areas in legend

    for i, column in enumerate(non_zero_columns):
        label = names_map.get(column, column)
        color = color_dict.get(
            label, "#000000"
        )  # Fallback color, if not found in color_dict

        cumulative += df[column]

        axs.plot(
            df.index,
            cumulative,
            linewidth=1,
            color="grey",
            alpha=0.7,
        )

        axs.fill_between(
            df.index,
            previous_cumulative,
            cumulative,
            color=color,
            alpha=0.7,
        )
        patches.append(mpatches.Patch(color=color, label=label, alpha=0.7))
        previous_cumulative = cumulative.copy()

    # Plotting P_to_dem
    axs.plot(
        df.index,
        df["P_to_dem"],
        label="P_to_dem",
        color="black",
        linewidth=1,
        alpha=1,
    )

    # Handling E_stored
    df["E_stored"].fillna(0, inplace=True)

    # Create a second y-axis for the stored energy
    axs2 = axs.twinx()

    # Plot the stored energy on the second y-axis
    axs2.plot(
        df.index,
        -df["E_stored"],
        color="#F7D507",  # Choose a color
        linewidth=1,
    )

    # Label the second y-axis

    axs.set_ylim(-4400, None)

    # Add a legend for the second y-axis
    axs2.legend(
        loc="upper right",
        fontsize=16,
        frameon=False,
        facecolor="white",
        edgecolor="white",
    )

    # Label the second y-axis
    axs2.set_ylabel(
        "Stored Energy [kWh]",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI SemiLight",
    )
    axs2.yaxis.label.set_color("#777777")

    # Add a legend for the second y-axis
    axs2.legend(
        loc="upper right",
        fontsize=16,
        frameon=False,
        facecolor="white",
        edgecolor="white",
    )

    # Ensure the tick parameters and spine colors of the second y-axis match the first
    axs2.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    axs2.spines["right"].set_edgecolor("#A3A3A3")
    axs2.spines["right"].set_linewidth(1)
    axs2.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

    # Set the limits of the second y-axis to be the same as the first
    axs2.set_ylim(axs.get_ylim())

    # Turn off the tick labels for the second y-axis
    axs2.set_yticklabels([])

    axs.plot(
        df.index,
        -df["s_out"],
        label="Storage Outflow (negative)",
        color="#AB2626",
        linewidth=1,
    )
    # Change color of storage inflow from black to, e.g., cyan
    axs.plot(
        df.index,
        -df["s_in"],
        label="Storage Inflow (negative)",
        color="#1F4E79",  # Changed color here
        linewidth=1,
    )
    # Set labels and title with style configuration
    font_properties = {
        "fontsize": 16,
        "color": "#777777",
        "fontfamily": "Segoe UI SemiLight",
    }
    axs.set_xlabel("Time [h]", **font_properties)
    axs.set_ylabel("Power [kW]", **font_properties)

    axs2.yaxis.label.set_color(font_properties["color"])
    axs2.yaxis.label.set_fontsize(font_properties["fontsize"])
    # axs[1].set_ylabel("Electricity Prices [ct/kWh]", **font_properties)

    title_properties = {
        "fontsize": 16,
        "color": "#777777",
        "fontfamily": "Segoe UI SemiLight",
    }

    axs.set_title(title, **title_properties)

    # Add legend with style configuration
    legend_properties = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.15),
        "ncol": 2,
        "frameon": False,
        "fontsize": 16,
        "facecolor": "white",
        "edgecolor": "white",
        "title_fontsize": "16",
        "labelcolor": "#777777",
    }

    # Adding line objects for the lines in the legend
    lines = [
        mlines.Line2D([], [], color="black", label="Load Profile"),
        mlines.Line2D([], [], color="#F7D507", label="Stored Energy (Negative) [kWh]"),
        mlines.Line2D([], [], color="#AB2626", label="Storage Outflow (Negative)"),
        mlines.Line2D(
            [], [], color="#1F4E79", label="Storage Inflow (Negative)"
        ),  # Adjusted color
    ]

    # Ensure all items (patches and lines) have labels to appear in the legend
    axs.legend(handles=patches + lines, **legend_properties)

    # X and Y axis properties with style configuration

    axs.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
    axs.spines["bottom"].set_edgecolor("#A3A3A3")
    axs.spines["bottom"].set_linewidth(1)

    axs.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    axs.spines["left"].set_edgecolor("#A3A3A3")
    axs.spines["left"].set_linewidth(1)

    axs.xaxis.label.set_color("#A3A3A3")
    axs.yaxis.label.set_color("#A3A3A3")
    axs2.yaxis.label.set_color("#A3A3A3")

    axs.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

    axs.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

    # Background and other spines color
    axs.set_facecolor("white")
    axs.spines["top"].set_visible(False)
    axs2.spines["top"].set_visible(False)
    axs2.spines["left"].set_visible(False)
    axs2.spines["bottom"].set_visible(False)

    # Show the plot
    st.pyplot(fig)


def plot_char_values_comparison(df_results):
    df_results["Value"] = df_results["Value"] / 1000
    # Define labels, titles, etc.
    title = "Comparison of Modes"
    x_label = " "
    y_label = "Values (in units)"

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
    for p, percentage in zip(bar_plot.patches, rearranged_percentages):
        height = p.get_height()
        bar_plot.annotate(
            f"{percentage:.0f}%",  # Annotate with integer percentage
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=9,  # Adjust fontsize as needed
            color="#777777",  # Adjust color as needed
        )

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    configure_style(bar_plot)
    return


def plot_storage_data(df):
    # Plot s_in and s_out individually
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(df.index, df["s_in"], label="storage inflow", linewidth=1, color="blue")
    ax1.plot(df.index, df["s_out"], label="storage outflow", linewidth=1, color="green")

    # Configure Labels, Title, Legend and Style for the First Plot
    ax1.set_xlabel("Time [h]", **font_properties)
    ax1.set_ylabel("s_in and s_out values", **font_properties)
    ax1.set_title("Storage in- and outflow over time", **title_properties)
    ax1.legend(**legend_properties)
    configure_style(ax1)

    # Show the first plot in Streamlit
    st.pyplot(fig1)

    # Plot E_stored individually
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(df.index, df["E_stored"], label="Stored Heat", linewidth=1, color="red")

    # Configure Labels, Title, Legend and Style for the Second Plot
    ax2.set_xlabel("Time [h]", **font_properties)
    ax2.set_ylabel("E_stored values", **font_properties)
    ax2.set_title("Stored energy over time", **title_properties)
    ax2.legend(**legend_properties)
    configure_style(ax2)

    # Show the second plot in Streamlit
    st.pyplot(fig2)


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


def plot_preise_flusstemp(strompreise_export, strompreise, gaspreise, flusstemperatur):
    # First Plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        strompreise_export, label="Electricity Export Price", linewidth=1, color="blue"
    )
    ax1.plot(strompreise, label="Electricity Import Price", linewidth=1, color="green")
    ax1.set_ylabel("Electricity Prices [€/MWh]", **font_properties)
    ax1.set_title("Electricity Prices", **title_properties)
    ax1.legend(**legend_properties)
    configure_style(ax1)
    st.pyplot(fig1)

    # Second Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(gaspreise, label="Gas Price", linewidth=1, color="red")
    ax2.set_ylabel("Gas Price [€/MWh]", **font_properties)
    ax2.set_title("Gas Price", **title_properties)
    ax2.legend(**legend_properties)
    configure_style(ax2)
    st.pyplot(fig2)

    # Third Plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(flusstemperatur, label="Flusstemperatur", linewidth=1, color="purple")
    ax3.set_xlabel("Time [h]", **font_properties)
    ax3.set_ylabel("Flusstemperatur [°C]", **font_properties)
    ax3.set_title("Flusstemperatur", **title_properties)
    ax3.legend(**legend_properties)
    configure_style(ax3)
    st.pyplot(fig3)


def plot_preise_flusstemp2(
    strompreise_export, strompreise, gaspreise, flusstemperatur
):  # flusstemperatur als neuer Parameter
    fig, axs = plt.subplots(
        3, 1, figsize=(10, 18), sharex=False
    )  # von 2 auf 3 Subplots geändert

    # Definieren Sie font_properties und andere Konfigurationen bevor Sie sie verwenden.
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
        "ncol": 2,
        "frameon": False,
        "fontsize": 16,
        "facecolor": "white",
        "edgecolor": "white",
        "title_fontsize": "16",
        "labelcolor": "#777777",
    }
    # Erster und zweiter Plot bleiben gleich.
    axs[0].plot(
        strompreise_export, label="Strompreis Export", linewidth=1, color="blue"
    )
    axs[0].plot(strompreise, label="Strompreis", linewidth=1, color="green")
    axs[0].legend(**legend_properties)

    axs[1].plot(gaspreise, label="Gaspreise", linewidth=1, color="red")

    # Neuer Plot für Flusstemperatur im dritten Subplot.
    axs[2].plot(
        flusstemperatur, label="Flusstemperatur", linewidth=1, color="purple"
    )  # Neue Farbe und Label
    axs[2].set_ylabel("Flusstemperatur [°C]", **font_properties)  # Neues ylabel
    axs[2].set_title("Flusstemperatur", **title_properties)  # Neuer Titel
    axs[2].legend(**legend_properties)  # Legende hinzufügen

    # Labels und Titel für die anderen Subplots und x-Achse
    axs[0].set_ylabel("Electricity Prices [€/MWh]", **font_properties)
    axs[0].set_title("Strompreise", **title_properties)
    axs[1].set_ylabel("Gas Prices [€/MWh]", **font_properties)
    axs[1].set_title("Gaspreise", **title_properties)
    axs[1].legend(**legend_properties)
    axs[2].set_xlabel(
        "Time [h]", **font_properties
    )  # x-Achse wird nur im letzten Subplot gesetzt

    # Setting x and y axis properties with style configuration
    for ax in axs:
        ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
        ax.spines["bottom"].set_edgecolor("#A3A3A3")
        ax.spines["bottom"].set_linewidth(1)

        ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
        ax.spines["left"].set_edgecolor("#A3A3A3")
        ax.spines["left"].set_linewidth(1)

        ax.xaxis.label.set_color("#A3A3A3")
        ax.yaxis.label.set_color("#A3A3A3")

        ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

        ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

        # Setting background and other spines color
        ax.set_facecolor("white")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # Show the plot
    st.pyplot(fig)


def plot_char_values(df_results):
    # Step 1: Sum up the necessary values
    e_stored_sum = df_results["E_stored"].sum()
    g_columns_sum = df_results[["g1", "g2", "g4", "g5", "g6"]].sum().sum()

    # Step 2: Create a new data structure to hold these summed values
    sums = {"E_stored": e_stored_sum, "g_columns": g_columns_sum}
    sum_df = pd.DataFrame(list(sums.items()), columns=["Category", "Value"])

    # Define labels, titles, etc.
    label1 = "E_stored"
    label2 = "g_columns"
    column_name = "Category"
    title = "Sum of Values"
    x_label = "operating Mode"
    y_label = "Values (in units)"

    # Define color palette, font color, and font family
    palette = {label1: "#3795D5", label2: "#D7E6F5"}
    font_color = "#777777"
    font_family = "Segoe UI SemiLight"

    # Assigning status labels to the entries
    sum_df["Status"] = sum_df["Category"].apply(
        lambda x: label1 if x == "E_stored" else label2
    )

    # Plotting with seaborn
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(
        x=column_name,
        y="Value",
        hue="Status",
        data=sum_df,
        palette=palette,
    )

    # Applying the custom styles
    bar_plot.set_xlabel(x_label, fontsize=16, color=font_color, fontfamily=font_family)
    bar_plot.set_ylabel(y_label, fontsize=16, color=font_color, fontfamily=font_family)
    bar_plot.set_title(title, fontsize=16, color=font_color, fontfamily=font_family)

    # Set the tick parameters
    bar_plot.tick_params(axis="both", which="major", labelsize=16, colors=font_color)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,  # Adjust as necessary
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

    # Adding percentage above the bars
    total_1 = sum_df[sum_df["Status"] == label1]["Value"].sum()
    total_2 = sum_df[sum_df["Status"] == label2]["Value"].sum()

    for p in bar_plot.patches:
        height = p.get_height() * 1e6  # Getting the original value
        bar_plot.text(
            p.get_x() + p.get_width() / 2.0,
            p.get_height(),
            f"{height:.0f}",  # Formatting as an integer (remove .0f for exact value)
            ha="center",
            va="bottom",
            fontsize=10,  # Adjust fontsize to fit the values properly
            color=font_color,
        )

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    return


def plot_single_values(df_results, mode):
    names_map = {
        "E_stored": "Stored Energy",
        "g1": "Waste Heat",
        "g2": "Waste\nHeat Pump",
        "g3": "Ambient\nHeat Pump",
        "g4": "Geothermal",
        "g5": "Solar Thermal",
        "g6": "Peak Load Boiler",
        "g71": "CHP",
    }

    if mode == 0:
        # Calculating the sums for each category
        sums = df_results[["E_stored", "g1", "g3", "g4", "g5", "g6", "g71"]].sum()
    elif mode == 1:
        sums = df_results[["f1", "f2", "f3", "f4", "f5", "f6", "f7"]].sum()

    # Remove categories with zero sum
    sums = sums[sums > 0]

    # Creating a new DataFrame to hold the sums
    sum_df = pd.DataFrame({"Category": sums.index, "Sum": sums.values})
    sum_df["Category"] = sum_df["Category"].replace(names_map)
    st.dataframe(sum_df)

    # Color mapping - ensure all categories in sum_df are included here
    color_dict = {
        "Stored Energy": "#777777",  # Example color, change as needed
        "Waste Heat": "#639729",
        "Waste\nHeat Pump": "#1F4E79",
        "Ambient\nHeat Pump": "#F7D507",
        "Geothermal": "#DD2525",
        "Solar Thermal": "#92D050",
        "Peak Load Boiler": "#EC9302",
        "CHP": "#F7D507",
        # Add other categories if needed
    }

    # Assign colors to each category, using a default color if not found in color_dict
    default_color = "#777777"  # Default color, change as needed
    colors = [
        color_dict.get(category, default_color) for category in sum_df["Category"]
    ]

    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x="Category", y="Sum", data=sum_df, palette=colors)

    # Defining custom styles
    font_color = "#777777"
    font_family = "Segoe UI SemiLight"

    # Applying custom styles
    bar_plot.set_xlabel(
        "Category", fontsize=16, color=font_color, fontfamily=font_family
    )
    bar_plot.set_ylabel(
        "Heat [kWh]",
        fontsize=16,
        color=font_color,
        fontfamily=font_family,
    )
    bar_plot.set_title(
        "Sum of the Stored and Generated Heat",
        fontsize=16,
        color=font_color,
        fontfamily=font_family,
    )
    plt.xticks(rotation=45, fontsize=12)

    # Displaying the sums on top of the bars
    for p in bar_plot.patches:
        height = p.get_height()
        bar_plot.text(
            p.get_x() + p.get_width() / 2.0,
            height,
            f"{height:.2f}",  # Adjust the number of decimal places as needed
            ha="center",
            va="bottom",
            fontsize=10,  # Adjust fontsize to fit the values properly
            color=font_color,
        )

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    return


def plot_comparison(df_melted, title, percentage_mode):
    plt.figure(figsize=(10, 6))
    palette = ["#1F4E79", "#356CA5", "#8AB5E1"]
    ax = sns.barplot(
        x="variable", y="value", hue="Mode", data=df_melted, palette=palette
    )
    plt.xlabel("", **font_properties)
    plt.ylabel("€/kWh²/hg CO₂", **font_properties)
    plt.title(title, **title_properties)
    plt.xticks(rotation=45)
    plt.ylim(0, 22000000)

    # Calculate percentages based on the highest value in each category
    max_values = df_melted.groupby("variable")["value"].max()
    percentages = [
        (value / max_values[variable]) * 100
        for _, (value, variable) in df_melted[["value", "variable"]].iterrows()
    ]
    # Calculate percentages based on the highest value across all categories

    max_value = df_melted["value"].max()
    percentages2 = [(value / max_value) * 100 for value in df_melted["value"]]

    # Rearrange the list
    rearranged_percentages = []

    # Annotate bars with calculated percentages
    if percentage_mode == "global":
        for i in range(3):
            rearranged_percentages.extend(percentages2[i::3])
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

    st.pyplot(plt.gcf())


# Continue with your existing data and call the modified function


def main_pulp(graphtitle, K_s_en):
    if checkbox4 == 0:
        file_path = "results/COP_vor_df.json"
    elif checkbox4 == 1:
        file_path = "results/COP_nach_df.json"

    # Lesen der JSON-Datei in einen DataFrame
    COP_df_imported = pd.read_json(file_path, orient="columns")
    COP_df_imported.fillna(COP_df_imported.mean(), inplace=True)

    # print the average values of each column in the dataframe to streamlit
    # st.write("Average values of each column in the dataframe:")
    # st.write(COP_df_imported.mean())

    # st.dataframe(COP_df_imported.iloc[start_hour : start_hour + hours, :])

    df_results = pd.DataFrame()
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

    def read_demand(start_hour, hours, column):
        # Read the results JSON file
        with open("results/df_results.json", "r") as f:
            df_results = pd.read_json(f)

        # Select the required column from the df_results dataframe
        if column not in df_results.columns:
            raise ValueError(f"Column {column} not found in df_results")

        demand_values = (
            df_results[column].iloc[start_hour : start_hour + hours].tolist()
        )
        return demand_values

    def read_data(file_name, start_hour, hours):
        # read csv file into a dataframe
        df = pd.read_csv(file_name, skiprows=range(1, start_hour), nrows=hours, sep=";")

        # Convert strings with comma as decimal separator to float
        df["gaspreis_22"] = df["gaspreis_22"].str.replace(",", ".").astype(float)

        # convert each column to a list
        strompreise_export = df["strompreis_22_ex"].tolist()
        strompreise = df["strompreis_22_in"].tolist()
        gaspreise = df["gaspreis_22"].tolist()
        Flusstemperatur = df["Isartemp"].tolist()

        return strompreise_export, strompreise, gaspreise, Flusstemperatur

    def read_data2(filename, start_hour, hours):
        data = pd.read_csv(filename, sep=";", header=None)

        # Convert the strompreise (electricity import prices) column to float
        data.iloc[:, 0] = (
            data.iloc[:, 0].astype(str).str.replace(",", ".").astype(float)
        )

        # Convert the strompreise_export (electricity export prices) column to float
        data.iloc[:, 1] = (
            data.iloc[:, 1].astype(str).str.replace(",", ".").astype(float)
        )

        # For gaspreise, take each value for 24 hours and only use the first 365 of them
        gaspreise_raw = (
            data.iloc[:365, 2].astype(str).str.replace(",", ".").astype(float)
        )
        gaspreise = []
        for price in gaspreise_raw:
            gaspreise.extend([price] * 24)  # Each gas price is repeated 24 times

        # Slice the dataframe and the gaspreise list according to the start_hour and hours
        strompreise = data.iloc[start_hour : start_hour + hours, 0].tolist()
        strompreise_export = data.iloc[start_hour : start_hour + hours, 1].tolist()
        gaspreise = gaspreise[start_hour : start_hour + hours]

        return (
            strompreise,
            strompreise_export,
            gaspreise,
        )  # Add other return variables as needed

    # Add other return variables as needed

    file_name = "Input_Netz.csv"

    # start_hour = st.number_input("Enter start hour", min_value=1, value=1)
    # hours = st.number_input("Enter hours", min_value=1, value=2000)
    if checkbox4 == 0:
        P_to_dem = read_demand(start_hour, hours, "Wärmelast_vor")
    elif checkbox4 == 1:
        P_to_dem = read_demand(start_hour, hours, "Wärmelast_nach")

    file_name = "Zeitreihen/zeitreihen_22.csv"

    (
        strompreise_export_2,
        strompreise_2,
        gaspreise_2,
        Flusstemperatur,
    ) = read_data(file_name, start_hour, hours)

    file_name = "Zeitreihen/preise2.csv"
    strompreise, strompreise_export, gaspreise = read_data2(
        file_name, start_hour, hours
    )

    preise_null = [0] * len(strompreise_export)
    preise_constant_high = [1000] * len(strompreise_export)
    preise_constant_low = [0.01] * len(strompreise_export)
    with open("erzeuger_df_vor.json") as f:
        data1 = json.load(f)

    with open("erzeuger_df_nach.json") as f:
        data2 = json.load(f)

    with open("results/color_FFE.json", "r") as f:
        color_FFE = json.load(f)

    names1 = data1["names"]
    names2 = data2["names"]
    partloads1 = data1["partloads"]
    partloads2 = data2["partloads"]
    erzeuger_df_vor = pd.read_json(data1["erzeuger_df_vor"])
    erzeuger_df_nach = pd.read_json(data2["erzeuger_df_nach"])
    Tiefentemperatur = 120

    # average_strompreise = sum(strompreise) / len(strompreise)
    I = list(range(len(P_to_dem)))

    m = LpProblem("Fernwärme_Erzeugungslastgang", LpMinimize)
    ###Parameters
    T = 24  # timesteps
    ΔT = 1  # hours
    # comodities
    K_elec_imp = 100000
    K_elec_exp = 0
    # storage
    K_s_pow_in = 3000  # kW
    K_s_pow_out = 3000
    Y_s_self = 0.02
    big_M = 1.2 * K_s_en

    # Masterarbeit Therese Farber Formel/Prozent für Speicherverluste

    K_p = {}
    if checkbox4 == 0:
        names = names1
        for i, name in enumerate(names):
            K_p[name] = [
                0 if v is None else v
                for v in erzeuger_df_vor[f"Erzeuger_{i+1}_vor"].tolist()
            ]
        partload_map = dict(zip(names, partloads1))
    elif checkbox4 == 1:
        names = names2
        for i, name in enumerate(names):
            K_p[name] = [
                0 if v is None else v
                for v in erzeuger_df_nach[f"Erzeuger_{i+1}_nach"].tolist()
            ]
        partload_map = dict(zip(names, partloads2))

    # st.dataframe(erzeuger_df_vor)

    # Abwärme
    K_p1 = K_p["waste_heat"][start_hour:] if "waste_heat" in names else [0] * 8761
    PL_p1 = partload_map.get("waste_heat", 0)
    # Waermepumpe1
    K_p2 = K_p["heatpump_1"][start_hour:] if "heatpump_1" in names else [0] * 8761
    PL_p2 = partload_map.get("heatpump_1", 0)

    # Waermepumpe2
    K_p3 = K_p["heatpump_2"][start_hour:] if "heatpump_2" in names else [0] * 8761
    PL_p3 = partload_map.get("heatpump_2", 0)

    # Geothermie
    K_p4 = K_p["geothermal"][start_hour:] if "geothermal" in names else [0] * 8761
    PL_p4 = partload_map.get("geothermal", 0)

    # Solar
    K_p5 = K_p["solarthermal"][start_hour:] if "solarthermal" in names else [0] * 8761
    PL_p5 = partload_map.get("solarthermal", 0)

    # Spitzenlastkessel
    K_p6 = K_p["PLB"][start_hour:] if "PLB" in names else [0] * 8761
    PL_p6 = partload_map.get("PLB", 0)

    # BHKW
    K_p7 = K_p["CHP"][start_hour:] if "CHP" in names else [0] * 8761
    PL_p7 = partload_map.get("CHP", 0)

    # st.write(K_p)
    ###decision variables
    e_imp = LpVariable.dicts(
        "ρ_imp_elec_t", I, lowBound=0, upBound=K_elec_imp, cat="Continuous"
    )
    e_imp_real = LpVariable.dicts(
        "ρ_imp_elec_t_real", I, lowBound=0, upBound=K_elec_imp, cat="Continuous"
    )
    e_exp = LpVariable.dicts(
        "ρ_exp_elec_t", I, lowBound=0, upBound=K_elec_exp, cat="Continuous"
    )
    e_imag = LpVariable.dicts(
        "ρ_imag_elec_t", I, lowBound=0, upBound=K_elec_imp, cat="Continuous"
    )
    g_imp = LpVariable.dicts("ρ_imp_gas_t", I, lowBound=0, cat="Continuous")
    s_in = LpVariable.dicts(
        "ρ_in_s_t", I, lowBound=0, upBound=K_s_pow_in, cat="Continuous"
    )
    s_out = LpVariable.dicts(
        "ρ_out_s_t", I, lowBound=0, upBound=K_s_pow_out, cat="Continuous"
    )
    # Define your new variables to represent the products
    z_in = LpVariable.dicts("z_in", I, lowBound=0, upBound=big_M, cat="Continuous")
    z_out = LpVariable.dicts("z_out", I, lowBound=0, upBound=big_M, cat="Continuous")

    E_stored = LpVariable.dicts("E_stored", I, lowBound=0, cat="Continuous")
    f2 = LpVariable.dicts("ρ_in_WP1_elec_t", I, lowBound=0, cat="Continuous")
    # f22 = LpVariable.dicts("ρ_in_WP1_water_t", I, lowBound=0, cat="Continuous")
    f3 = LpVariable.dicts("ρ_in_WP2_elec_t", I, lowBound=0, cat="Continuous")
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
    Elec_sum = LpVariable("Elec_sum")
    Gas_sum = LpVariable("Gas_sum")

    E_real = LpVariable("total_actual_Euros")
    E_tot = LpVariable("total_theoretical_Euros")
    E_imp = LpVariable("Euros_for_Import")
    E_exp = LpVariable("Euros_for_Export")
    E_emissions = LpVariable("Emissions")
    E_imag = LpVariable("Euros_for_Import_imaginary")

    if checkbox == 1:
        strompreise_export_c = preise_constant_low
        strompreise_c = preise_constant_low
        gaspreise_c = preise_constant_high
    elif checkbox == 3:
        strompreise_export_c = preise_constant_low
        strompreise_c = preise_constant_low
        gaspreise_c = preise_constant_high
    else:
        strompreise_export_c = strompreise_export
        strompreise_c = strompreise
        gaspreise_c = gaspreise

    if checkbox2 == 1:
        K_s_en = 0

    ###Constraints
    if checkbox3 == 0:
        m += E_tot
        m += E_tot == E_imp + E_exp
    elif checkbox3 == 1:
        if checkbox == 0:
            m += E_tot
            m += E_tot == E_emissions + 0.0001 * E_imag
        if checkbox == 1:
            m += E_tot
            m += E_tot == E_imp + E_exp

    m += (
        E_imp
        == sum(strompreise_c[i] * e_imp[i] for i in I) / 100
        + sum(gaspreise_c[i] * g_imp[i] for i in I) / 100
    )
    # e imag
    m += E_imag == sum(e_imag[i] for i in I) / 100
    m += E_exp == sum(strompreise_export_c[i] * e_exp[i] for i in I) / 100
    m += (
        E_real
        == sum(strompreise[i] * e_imp_real[i] for i in I) / 1000
        + sum(gaspreise[i] * g_imp[i] for i in I) / 1000
        + sum(strompreise_export[i] * e_exp[i] for i in I) / 1000
    )
    m += (
        E_emissions
        == sum(0.468 * e_imp_real[i] for i in I) / 1000
        + sum(0.201 * g_imp[i] for i in I) / 1000
    )
    m += Elec_sum == sum(e_imp_real[i] for i in I)
    m += Gas_sum == sum(g_imp[i] for i in I)

    # m += X == sum(average_strompreise * x[i] for i in I) / 100

    for i in range(len(P_to_dem)):
        P = P_to_dem[i]

        # Check if column exists in DataFrame, if not, assign 0
        try:
            A_p2_in = COP_df_imported.get(
                "heatpump_1", pd.Series(0, index=COP_df_imported.index)
            ).loc[i + start_hour]
            A_p3_in = COP_df_imported.get(
                "heatpump_2", pd.Series(0, index=COP_df_imported.index)
            ).loc[i + start_hour]
            A_p4_in = COP_df_imported.get(
                "geothermal", pd.Series(0, index=COP_df_imported.index)
            ).loc[i + start_hour]
        except KeyError:
            A_p2_in = 0
            A_p3_in = 0
            A_p4_in = 0

        # st.write(i, A_p3_in)

        # 0.45 * (T_vl_vor[i] + 273.15) / (T_vl_vor[i] - Flusstemperatur[i]) #compare to ε = self.Gütegrad * ((Tvl + 273.15) / (Tvl - self.T_q))
        # A_p3_in = 0.45 * (60 + 273.15) / (60 - Flusstemperatur[i]) #        ε = self.Gütegrad * (Tvl + 273.15) / (Tvl - self.T_q)
        # A_p4_in = -0.45 * (60 + 273.15) / (60 - Tiefentemperatur) #        # Placeholder calculation:
        # V = current_last / (self.Tgeo - (Trl + T_Wü_delta_r) * ρ_water * cp_water)
        # P = V / 3600 * ρ_water * 9.81 * self.h_förder / self.η_geo
        # return P / 1000
        A_p5_in = 100  # Very high COP for solar thermal, since it is only used to prioritize WH over Solarthermalheat, as none of them is actually considered in the prices as they are modelled without electrical input
        A_p6_in = ηSpitzenkessel  # PLB
        A_p7_in = 1
        A_p7_out_heat = ηBHKW_therm  # BHKW heat
        A_p7_out_elec = ηBHKW_el  # BHKW elec

        # Storage
        if i == 0:  # for the first period, E_stored_prev is 0
            m += E_stored[i] == 0
            m += s_in[i] == 0
            m += s_out[i] == 0
        else:  # for other periods, E_stored_prev is E_stored from last period
            m += E_stored[i] == (1 - Y_s_self) * E_stored[i - 1] + s_in[i] - s_out[i]
        m += E_stored[i] <= K_s_en
        # m += s_in <= x_s[i] * s_in[i]
        # m += s_out <= (1 - x_s[i]) * s_out[i]
        m += z_in[i] <= big_M * x_s[i]
        m += z_in[i] >= s_in[i] - big_M * (1 - x_s[i])
        m += z_in[i] <= s_in[i] + big_M * (1 - x_s[i])
        m += z_in[i] >= 0

        m += z_out[i] <= big_M * (1 - x_s[i])
        m += z_out[i] >= s_out[i] - big_M * x_s[i]
        m += z_out[i] <= s_out[i] + big_M * x_s[i]
        m += z_out[i] >= 0
        # Replace your original constraints with constraints in terms of z_in and z_out
        m += s_in[i] <= z_in[i]
        m += s_out[i] <= z_out[i]
        # Process Flows
        # p1 WH
        m += g1[i] <= K_p1[i]
        # p2 WP1
        m += g2[i] == f2[i] * A_p2_in
        m += g2[i] >= x2[i] * K_p2[i] * PL_p2
        m += g2[i] <= x2[i] * K_p2[i]

        # m += g2[i] <= K_p2[i]
        # m += g2[i] - K_p2[i] * PL_p2 >= -(1 - x2[i]) * K_p2[i]

        # p3 WP2
        m += g3[i] == f3[i] * A_p3_in
        m += g3[i] >= x3[i] * K_p3[i] * PL_p3
        m += g3[i] <= x3[i] * K_p3[i]
        # p4 Geothermal
        m += g4[i] == f4[i] * A_p4_in
        m += g4[i] >= x4[i] * K_p4[i] * PL_p4
        m += g4[i] <= x4[i] * K_p4[i]
        # p5 solar
        m += g5[i] == f5[i] * A_p5_in
        m += g5[i] <= x5[i] * K_p5[i]
        # p6
        m += g6[i] == f6[i] * A_p6_in
        m += g6[i] >= x6[i] * K_p6[i] * PL_p6
        m += g6[i] <= x6[i] * K_p6[i]
        # p7
        m += g71[i] == f7[i] * A_p7_out_heat
        m += g72[i] == f7[i] * A_p7_out_elec
        m += g71[i] >= x7[i] * K_p7[i] * PL_p7
        m += g71[i] <= x7[i] * K_p7[i]

        # Commodities
        # heat
        m += (
            g1[i] + g2[i] + g3[i] + g4[i] + g5[i] + g6[i] + g71[i] + s_out[i]
            == s_in[i] + P
        )
        # electricity
        m += e_imp[i] == f2[i] + f3[i] + f4[i] + f5[i] + g72[i] + e_exp[i]
        m += e_imp_real[i] == f2[i] + f3[i] + f4[i] + e_exp[i]
        # e_imag

        m += e_imag[i] == f5[i]

        # gas
        m += g_imp[i] == f6[i] + f7[i]

    # solver = GUROBI_CMD(options=[("MIPGap", 0.05)])
    # solver = GUROBI_CMD(options=[("TimeLimit", 100)])  # 5 minutes time limit
    # solver = GUROBI_CMD(options=[("LogFile", "gurobi.log")])

    # solver = GUROBI_CMD()

    # m.solve(solver)
    # With the following CBC solver code:
    solver = PULP_CBC_CMD(msg=True, timeLimit=100, fracGap=0.05)  # CBC with options
    m.solve(solver)

    # Extract solution
    e_imp_m = {i: e_imp[i].varValue for i in I}
    e_exp_m = {i: e_exp[i].varValue for i in I}
    s_in_m = {i: s_in[i].varValue for i in I}
    s_out_m = {i: s_out[i].varValue for i in I}
    E_stored_m = {i: E_stored[i].varValue for i in I}
    f2_m = {i: f2[i].varValue for i in I}
    f3_m = {i: f3[i].varValue for i in I}
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
    E_real_m = E_real.varValue
    E_emissions_m = E_emissions.varValue
    Elec_sum = Elec_sum.varValue
    Gas_sum = Gas_sum.varValue

    df_results = pd.DataFrame()

    for key in e_imp_m.keys():
        data = {
            # "K_s_en": K_s_en_m,
            "E_real": E_real_m,
            "e_imp": e_imp_m[key],
            "e_exp": e_exp_m[key],
            "s_in": s_in_m[key],
            "s_out": s_out_m[key],
            "E_stored": E_stored_m[key],
            "x_elec": strompreise[key],
            # "x_elec_export": strompreise_export[key + start_hour],
            "x_gas": gaspreise[key],
            # "f2": f2_m[key],
            # "f3": f3_m[key],
            # "f4": f4_m[key],
            # "f5": f5_m[key],
            # "f6": f6_m[key],
            # "f7": f7_m[key],
            "g1": g1_m[key],
            "g2": g2_m[key],
            "g3": g3_m[key],
            "g4": g4_m[key],
            "g5": g5_m[key],
            "g6": g6_m[key],
            "g71": g71_m[key],
            "g72": g72_m[key],
            "P_to_dem": P_to_dem_m[key],
        }
        df_results = pd.concat(
            [df_results, pd.DataFrame(data, index=[key])], ignore_index=True
        )

    print(df_results)

    st.dataframe(df_results)
    # print the sum od s_out
    # st.write(f"Der gesamte Storage outflow beträgt {sum(s_out_m.values()):.2f} kWh")
    # print the sum of P_to_dem
    # st.write(f"Der gesamte Wärmebedarf beträgt {sum(P_to_dem_m.values()):.2f} kWh")
    # st.write(Elec_sum)
    # st.write(Gas_sum)

    # Creating a dictionary to map the old column names to the new column names
    column_mapping = {
        "g1": "Erzeuger_1_nach",
        "g2": "Erzeuger_2_nach",
        "g3": "Erzeuger_3_nach",
        "g4": "Erzeuger_4_nach",
        "g5": "Erzeuger_5_nach",
        "g6": "Erzeuger_6_nach",
        "g71": "Erzeuger_7_nach",
    }

    # Creating a new dataframe with the renamed columns
    df_production_storage = df_results.rename(columns=column_mapping)

    # Adding a new column called 'storage' which is the difference of 's_out' and 's_in'
    # df_production_storage["storage"] = ( df_production_storage["s_out"] - df_production_storage["s_in"])

    # Keeping only the necessary columns in the new dataframe
    df_production_storage = df_production_storage[
        list(column_mapping.values())  # + ["storage"]
    ]
    st.write("# Numerical Results")
    st.write(f"Das Ergebnis der Kostenfunktion im Modell ist {E_tot_m:.2f} ")

    st.write(f"Die tatsächlichen Energiekosten betragen {E_real_m:.2f} €")

    total = df_results["E_stored"].sum()

    # Display the total sum of the column in Streamlit
    st.write(f"The sum of the stored energy is: {total/1000:.2f} MWh")
    st.write(f"Die gesamten Emissionen betragen {E_emissions_m:.2f} t CO₂")
    plot_data3(df_results, graphtitle)
    # plot_power_usage_storage(df_production_storage, color_FFE)

    my_dict = {
        "Erzeuger_1": "waste_heat",
        "Erzeuger_2": "heatpump_1",
        "Erzeuger_3": "solarthermal",
        "Erzeuger_4": "geothermal",
        "Erzeuger_5": "PLB",
    }

    plot_single_values(df_results, 0)
    # plot_single_values(df_results, 1)

    # plot_char_values(df_results)
    # plot_storage_data(df_results)
    plot_preise_flusstemp(strompreise_export, strompreise, gaspreise, Flusstemperatur)

    # plot_actual_production(df_input, df_production_storage, color_FFE, "Test", my_dict, start_hour)
    # plot_sorted_production(df_input, df_production_storage, color_FFE, "Test", my_dict)
    # Step 1: Sum up the necessary values
    # plot_char_values(df_results)


start_hour = st.number_input("Enter start hour", min_value=1, value=1)
hours = st.number_input("Enter hours", min_value=1, max_value=8760, value=2000)
if hours == 8760:
    hours = 8759

expander = st.expander("Storage Parameters")
with expander:
    K_s_en = st.number_input("Enter storage size [kWh]", min_value=0, value=4000)
    K_s_pow_in = st.number_input(
        "Enter the storage inflow capacity [kW]", min_value=0, value=3000
    )
    K_s_pow_out = st.number_input(
        "Enter storage outflow capacity [kW]", min_value=0, value=3000
    )
    Y_s_self = (
        st.number_input(
            "Enter the Percentage of stored energy lost at every time step [%]",
            min_value=0,
            value=2,
        )
        / 100
    )


st.markdown("## Selection Options")
st.markdown("Please select your preferences:")

col1, col2 = st.columns(2)

with col1:
    checkbox = st.checkbox(
        "Use linear prioritization",
        help="Opt for linear prioritization by applying constant prices and prioritizing renewables.",
    )
    checkbox2 = st.checkbox(
        "Use no storage", help="Select this to avoid using any storage solutions."
    )

with col2:
    checkbox3 = st.checkbox(
        "Minimize emissions instead of costs",
        help="Focus on minimizing operational emissions instead of operational costsin the optimization process.",
    )
    checkbox4 = st.checkbox(
        "Use reduced network temperatures",
        help="Apply lower temperature values in network calculations and storage simulation.",
    )


if st.button("Submit"):
    st.write(f"You entered {hours} hours.")
    if checkbox4:
        if checkbox3:
            if checkbox2:
                main_pulp(
                    "Emission Optimized Heat Generation at lower Temperatures", K_s_en
                )
            else:
                main_pulp(
                    "Emission Optimized Heat Generation and Storage at lower Temperatures",
                    K_s_en,
                )
        else:
            if checkbox:
                if checkbox2:
                    main_pulp(
                        "Linear Prioritized Heat Generation at lower Temperatures",
                        K_s_en,
                    )
                else:
                    main_pulp(
                        "Linear Prioritized Heat Generation and Storage at lower Temperatures",
                        K_s_en,
                    )
            else:
                main_pulp(
                    "Cost Optimized Heat Generation and Storage at lower Temperatures",
                    K_s_en,
                )
    else:
        if checkbox3:
            if checkbox2:
                main_pulp("Emission Optimized Heat Generation", K_s_en)
            else:
                main_pulp("Emission Optimized Heat Generation and Storage", K_s_en)
        else:
            if checkbox:
                if checkbox2:
                    main_pulp("Linear Prioritized Heat Generation", K_s_en)
                else:
                    main_pulp("Linear Prioritized Heat Generation and Storage", K_s_en)
            else:
                main_pulp("Cost Optimized Heat Generation and Storage", K_s_en)
