import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.colors as mcolors
import colorsys
from typing import Generator, Sequence
from contextlib import contextmanager

color_dict = {
    "🚰 Wärmepumpe1 - begrenzter Volumenstrom der Quelle (m³/h), Quelltemperatur konstant (°C)": "#1F4E79",
    "🌊 Wärmepumpe2 - Begrenzte Leistung (kW), bei schwankender Quelltemperatur (°C)": "#F7D507",
    "⛰️ Geothermie - Maximale Leistung (kW)": "#DD2525",
    "☀️ Solarthermie - Sonneneinstrahlung (kW/m²)": "#92D050",
    "🔥 Spitzenlastkessel - Maximale Leistung (kW)": "#EC9302",
    "🏭 BHKW - Maximale Leistung (kW)": "#639729",
}


def plot_actual_production(df_input, actual_production_df, color_FFE, title):
    fig, ax = plt.subplots(figsize=(12, 6))

    lastgang_plot = df_input["Lastgang"].plot.area(
        ax=ax, color="#E0E0E0", linewidth=0, zorder=1, alpha=0.5, label="Lastgang"
    )

    area_plot = actual_production_df.plot.area(
        ax=ax, color=color_FFE, linewidth=0, zorder=2, alpha=0.7
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(lastgang_plot.get_legend_handles_labels()[0][0])
    ax.legend(handles=handles, labels=labels)

    plt.xlabel("hours")
    plt.ylabel("Erzeugung")
    plt.title(title)
    plt.grid(True)

    st.pyplot(fig)

    sorted_df = actual_production_df.copy()
    for col in sorted_df.columns:
        sorted_df[col] = sorted_df[col].sort_values(ascending=False).values

    return sorted_df


def plot_sorted_production(df_input, sorted_df, actual_production_df, color_FFE, title):
    fig, ax = plt.subplots(figsize=(12, 6))

    lastgang_plot = (
        df_input["Lastgang"]
        .sort_values(ascending=False)
        .reset_index(drop=True)
        .plot.area(ax=ax, color="#E0E0E0", linewidth=0, zorder=1, alpha=0.5)
    )

    area_plot = sorted_df.plot.area(
        ax=ax, color=color_FFE, linewidth=0, zorder=2, alpha=0.7
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(lastgang_plot.get_legend_handles_labels()[0][0])
    ax.legend(handles=handles, labels=labels)

    plt.xlabel("hours")
    plt.ylabel("kWh")
    plt.title(title)
    plt.grid(True)

    st.pyplot(fig)

    plot_df = actual_production_df.copy()
    plot_df["Lastgang"] = df_input["Lastgang"]

    return plot_df


def lighten_color(color, amount=0.5):
    """
    Lightens the given color.

    Parameters:
    color : str
        The color to lighten.
    amount : float, default=0.5
        The amount to lighten the color. The higher the amount, the lighter the color.

    Returns:
    color : str
        The lightened color.
    """
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_power_usage(Power_df_vor, Power_df_nach, color_FFE):
    num_of_erzeuger = len(color_FFE)

    for i in range(num_of_erzeuger):
        # Check if all entries in the respective columns are zero
        if (
            Power_df_vor[f"Erzeuger_{i+1}_vor"].sum() == 0
            and Power_df_nach[f"Erzeuger_{i+1}_nach"].sum() == 0
        ):
            continue

        plt.figure(figsize=(12, 6))

        plt.plot(
            Power_df_vor.index,
            Power_df_vor[f"Erzeuger_{i+1}_vor"],
            color=color_FFE[i],
            label="vor",
        )
        plt.plot(
            Power_df_nach.index,
            Power_df_nach[f"Erzeuger_{i+1}_nach"],
            color=lighten_color(color_FFE[i]),
            label="nach",
        )

        plt.title(f"Erzeuger {i+1} Power Usage")
        plt.xlabel("Time")
        plt.ylabel("Power Usage")
        plt.grid(True)
        plt.legend()

        st.pyplot(plt.gcf())


def plot_total_change(
    df1, df2, color_FFE, label1, label2, column_name, title, x_label, y_label
):
    df1_sum = pd.DataFrame(df1.sum(), columns=["Value"])
    df1_sum["Status"] = label1
    df1_sum[column_name] = df1_sum.index.str.replace("_" + label1.lower(), "")

    df2_sum = pd.DataFrame(df2.sum(), columns=["Value"])
    df2_sum["Status"] = label2
    df2_sum[column_name] = df2_sum.index.str.replace("_" + label2.lower(), "")

    # Concatenate the two DataFrames
    sum_df = pd.concat([df1_sum, df2_sum])

    # Remove rows with value 0
    sum_df = sum_df[sum_df["Value"] != 0]
    # Sort the DataFrame
    sum_df.sort_values(by=[column_name, "Status"], inplace=True)

    # Reset index
    sum_df.reset_index(drop=True, inplace=True)

    # Calculate the total production for each status
    total_1 = sum_df[sum_df["Status"] == label1]["Value"].sum()
    total_2 = sum_df[sum_df["Status"] == label2]["Value"].sum()

    # Define color palette
    palette = {label1: "#E6E6E6", label2: "#0033A0"}

    # Plotting with seaborn
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(
        x=column_name, y="Value", hue="Status", data=sum_df, palette=palette
    )

    # Loop over the bars, and adjust the height to add the text label
    for p in bar_plot.patches:
        total = total_1 if label1 in p.get_label() else total_2
        percentage = "{:.1f}%".format(100 * p.get_height() / total)
        bar_plot.text(
            p.get_x() + p.get_width() / 2.0,
            p.get_height(),
            percentage,
            ha="center",
            va="bottom",
        )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())


def plot_total_emissions(
    df1, df2, erzeugerpark, column_name, df1_status="Vor", df2_status="Nach", ylabel=""
):
    df1_sum = df1.sum().reset_index()
    df1_sum.columns = [column_name, "Value"]
    df1_sum["Status"] = df1_status

    df2_sum = df2.sum().reset_index()
    df2_sum.columns = [column_name, "Value"]
    df2_sum["Status"] = df2_status

    # Convert power usage to CO2 emissions
    for erzeuger in erzeugerpark:
        if hasattr(erzeuger, "co2_emission_factor") and erzeuger.co2_emission_factor:
            df1_sum.loc[
                df1_sum[column_name] == erzeuger.__class__.__name__, "Value"
            ] *= erzeuger.co2_emission_factor
            df2_sum.loc[
                df2_sum[column_name] == erzeuger.__class__.__name__, "Value"
            ] *= erzeuger.co2_emission_factor

    # Concatenate the two DataFrames
    sum_df = pd.concat([df1_sum, df2_sum])

    # Remove rows where Value equals to 0
    sum_df = sum_df[sum_df["Value"] != 0]

    # Combine 'Erzeuger' and 'Status' into one column
    sum_df["combined"] = sum_df[column_name] + " " + sum_df["Status"]

    # Calculate the total sum for each dataframe
    total_df1 = df1_sum["Value"].sum()
    total_df2 = df2_sum["Value"].sum()

    # Define color palette for combined column
    palette = {
        f"{name} {df1_status}": "#E6E6E6" for name in sum_df[column_name].unique()
    }
    palette.update(
        {f"{name} {df2_status}": "#0033A0" for name in sum_df[column_name].unique()}
    )

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x="combined", y="Value", data=sum_df, palette=palette)

    for p in bar_plot.patches:
        total = total_df1 if df1_status in p.get_label() else total_df2
        percentage = "{:.1f}%".format(100 * p.get_height() / total)
        bar_plot.text(
            p.get_x() + p.get_width() / 2.0,
            p.get_height(),
            percentage,
            ha="center",
            va="bottom",
        )

    plt.xlabel(column_name)
    plt.ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(plt.gcf())
