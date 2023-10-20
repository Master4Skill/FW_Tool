import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.colors as mcolors
import colorsys
from typing import Generator, Sequence
from contextlib import contextmanager


def plot_actual_production(
    df_input, actual_production_df, color_FFE, title, my_dict, start_hour
):
    fig, ax = plt.subplots(figsize=(12, 6))

    lastgang_plot = df_input["Lastgang"].plot.area(
        ax=ax, color="#E0E0E0", linewidth=0, zorder=1, alpha=0.5, label="Load profile"
    )
    actual_production_df.index = range(
        start_hour, start_hour + len(actual_production_df)
    )
    area_plot = actual_production_df.plot.area(
        ax=ax, color=color_FFE, linewidth=0, zorder=2, alpha=0.7
    )

    # Get the current labels
    handles, labels = ax.get_legend_handles_labels()
    handles.append(lastgang_plot.get_legend_handles_labels()[0][0])

    # Create a new list of labels using a mapping dictionary
    new_labels = []
    for label in labels:
        split_label = label.split("_")
        if len(split_label) >= 2:
            key = split_label[0] + "_" + split_label[1]
            new_labels.append(my_dict.get(key, label))
        else:
            new_labels.append(label)

    # Set the new labels
    ax.legend(
        handles=handles,
        labels=new_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=16,
        facecolor="white",
        edgecolor="white",
        title_fontsize="16",
        labelcolor="#777777",
    )

    # Set labels with specified font properties
    ax.set_xlabel(
        "Time [h]", fontsize=16, color="#777777", fontfamily="Segoe UI SemiLight"
    )
    ax.set_ylabel(
        "Production [kW]", fontsize=16, color="#777777", fontfamily="Segoe UI SemiLight"
    )

    # Set title with specified font properties
    ax.set_title(title, fontsize=16, color="#777777", fontfamily="Segoe UI SemiLight")

    # Set tick parameters
    ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

    # Set spine colors and widths
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_linewidth(1)

    # Set grid
    ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

    # Set background color and hide the top and right spines
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Display the plot
    st.pyplot(fig)

    sorted_df = actual_production_df.copy()
    for col in sorted_df.columns:
        sorted_df[col] = sorted_df[col].sort_values(ascending=False).values

    return sorted_df


def plot_sorted_production(
    df_input, sorted_df, actual_production_df, color_FFE, title, my_dict
):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Your existing plot commands
    lastgang_plot = (
        df_input["Lastgang"]
        .sort_values(ascending=False)
        .reset_index(drop=True)
        .plot.area(
            ax=ax,
            color="#E0E0E0",
            linewidth=0,
            zorder=1,
            alpha=0.5,
            label="Load profile",
        )
    )
    area_plot = sorted_df.plot.area(
        ax=ax, color=color_FFE, linewidth=0, zorder=2, alpha=0.7
    )

    # Adjusting legends as in your script
    handles, labels = ax.get_legend_handles_labels()
    handles.append(lastgang_plot.get_legend_handles_labels()[0][0])
    new_labels = [
        my_dict.get(label.split("_")[0] + "_" + label.split("_")[1], label)
        if len(label.split("_")) >= 2
        else label
        for label in labels
    ]

    # Applying similar styles as in your first plot
    ax.set_xlabel(
        "Time [h]", fontsize=16, color="#777777", fontfamily="Segoe UI SemiLight"
    )
    ax.set_ylabel(
        "Production [kW]", fontsize=16, color="#777777", fontfamily="Segoe UI SemiLight"
    )

    ax.xaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["bottom"].set_linewidth(1)

    ax.yaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_linewidth(1)

    ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

    ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

    ax.set_title(title, fontsize=16, color="#777777", fontfamily="Segoe UI SemiLight")

    ax.legend(
        handles=handles,
        labels=new_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=16,
        facecolor="white",
        edgecolor="white",
        title_fontsize="16",
        labelcolor="#777777",
    )

    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Adjusting y-limit as in your script
    plt.ylim(0, 20000)

    # Display the plot (assuming you're using Streamlit)
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


def plot_power_usage2(Power_df_vor, Power_df_nach, color_FFE):
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
            label="before",
        )
        plt.plot(
            Power_df_nach.index,
            Power_df_nach[f"Erzeuger_{i+1}_nach"],
            color=lighten_color(color_FFE[i]),
            label="after",
        )

        plt.title(f"Erzeuger {i+1} Power Consumption")
        plt.xlabel("Time")
        plt.ylabel("Power Consumption [kW]")
        plt.grid(True)
        plt.legend()

        st.pyplot(plt.gcf())


def plot_power_usage(Power_df_vor, Power_df_nach, color_FFE):
    num_of_erzeuger = len(color_FFE)

    for i in range(num_of_erzeuger):
        # Check if all entries in the respective columns are zero
        if (
            Power_df_vor[f"Erzeuger_{i+1}_vor"].sum() == 0
            and Power_df_nach[f"Erzeuger_{i+1}_nach"].sum() == 0
        ):
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            Power_df_vor.index,
            Power_df_vor[f"Erzeuger_{i+1}_vor"],
            linestyle="-",
            linewidth=2,
            color=color_FFE[i],
            label="before Temp. reduction",
        )
        ax.plot(
            Power_df_nach.index,
            Power_df_nach[f"Erzeuger_{i+1}_nach"],
            linestyle="-",
            linewidth=2,
            color=lighten_color(color_FFE[i]),  # Adjust the color as needed
            label="after Temp. reduction",
        )

        ax.set_xlabel(
            "Time",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
        ax.set_ylabel(
            "Power Consumption [kW]",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )

        ax.xaxis.label.set_color("#A3A3A3")
        ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
        ax.spines["bottom"].set_edgecolor("#A3A3A3")
        ax.spines["bottom"].set_linewidth(1)

        ax.yaxis.label.set_color("#A3A3A3")
        ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
        ax.spines["left"].set_edgecolor("#A3A3A3")
        ax.spines["left"].set_linewidth(1)

        ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

        ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

        ax.set_title(
            f"Generator {i+1} Power Consumption",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            frameon=False,
            fontsize=16,
            facecolor="white",
            edgecolor="white",
            title_fontsize="16",
            labelcolor="#777777",
        )

        ax.set_facecolor("white")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        st.pyplot(fig)


import matplotlib.pyplot as plt


def plot_power_usage_storage(Power_df_nach, color_FFE):
    num_of_erzeuger = len(color_FFE)

    for i in range(num_of_erzeuger):
        # Check if all entries in the respective columns are zero
        if Power_df_nach[f"Erzeuger_{i+1}_nach"].sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            Power_df_nach.index,
            Power_df_nach[f"Erzeuger_{i+1}_nach"],
            linestyle="-",
            linewidth=2,
            color=color_FFE[i],
            label="nach",
        )

        ax.set_xlabel(
            "Time",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
        ax.set_ylabel(
            "Power Consumption [kW]",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )

        ax.xaxis.label.set_color("#A3A3A3")
        ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
        ax.spines["bottom"].set_edgecolor("#A3A3A3")
        ax.spines["bottom"].set_linewidth(1)

        ax.yaxis.label.set_color("#A3A3A3")
        ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
        ax.spines["left"].set_edgecolor("#A3A3A3")
        ax.spines["left"].set_linewidth(1)

        ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

        ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

        ax.set_title(
            f"Erzeuger {i+1} Power Consumption",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=1,  # Adjusted ncol to 1, as we only have one series to plot now
            frameon=False,
            fontsize=16,
            facecolor="white",
            edgecolor="white",
            title_fontsize="16",
            labelcolor="#777777",
        )

        ax.set_facecolor("white")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        # Replace with your plot display function if not using streamlit
        # st.pyplot(fig)
        st.pyplot(fig)


# Don't forget to import lighten_color function or remove it if it's not required.


def plot_total_change(
    df1,
    df2,
    color_FFE,
    label1,
    label2,
    column_name,
    title,
    x_label,
    y_label,
    my_dict,
    box_x,
    box_y,
):
    df1_sum = pd.DataFrame(df1.sum(), columns=["Value"])
    df1_sum["Status"] = label1
    df1_sum["Original_Index"] = range(len(df1_sum))
    # df1_sum[column_name] = df1_sum.index.str.replace("_" + label1.lower(), "")

    df2_sum = pd.DataFrame(df2.sum(), columns=["Value"])
    df2_sum["Status"] = label2
    df2_sum["Original_Index"] = range(len(df2_sum))
    # df2_sum[column_name] = df2_sum.index.str.replace("_" + label2.lower(), "")

    def get_new_label(old_label):
        split_label = old_label.split("_")
        if len(split_label) >= 2:
            key = split_label[0] + "_" + split_label[1]
            return my_dict.get(
                key, old_label
            )  # Use original label if key not found in my_dict
        else:
            return old_label  # Use original label if it doesn't contain at least two underscores

    df1_sum[column_name] = df1_sum.index.to_series().apply(get_new_label)
    df2_sum[column_name] = df2_sum.index.to_series().apply(get_new_label)

    df1_sum["Value"] = df1_sum["Value"] / 1e6
    df2_sum["Value"] = df2_sum["Value"] / 1e6

    # Concatenate the two DataFrames
    sum_df = pd.concat([df1_sum, df2_sum])
    # st.dataframe(sum_df)

    total_value_sum = df1_sum["Value"].sum()
    # st.write(f"Total sum og GWh before: {total_value_sum}")

    total_value_sum = df2_sum["Value"].sum()
    # st.write(f"Total sum of values: {total_value_sum}")

    # st.dataframe(df2_sum)
    # Remove rows with value 0
    sum_df = sum_df[sum_df["Value"] != 0]
    # Sort the DataFrame
    sum_df.sort_values(by=["Original_Index", "Status"], inplace=True)
    # sum_df.sort_values(by=[column_name, "Status"], inplace=True)

    # Reset index
    sum_df.reset_index(drop=True, inplace=True)

    # Calculate the total for each status
    total_1 = sum_df[sum_df["Status"] == label1]["Value"].sum()
    total_2 = sum_df[sum_df["Status"] == label2]["Value"].sum()
    # st.write(color_FFE)
    # Define color palette
    palette = {label1: "#3795D5", label2: "#D7E6F5"}

    # palette = {label1: color_FFE[0], label2: color_FFE[1]}

    font_color = "#777777"
    font_family = "Segoe UI SemiLight"

    # Plotting with seaborn
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(
        x=column_name,
        y="Value",
        hue="Status",
        data=sum_df,
        palette=palette,
        hue_order=[label1, label2],
    )

    # Applying the custom styles
    bar_plot.set_xlabel(x_label, fontsize=16, color=font_color, fontfamily=font_family)
    bar_plot.set_ylabel(y_label, fontsize=16, color=font_color, fontfamily=font_family)
    bar_plot.set_title(title, fontsize=16, color=font_color, fontfamily=font_family)

    # Set the tick parameters
    bar_plot.tick_params(axis="both", which="major", labelsize=16, colors=font_color)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.55),
        ncol=2,  # Adjust as necessary
        frameon=False,
        fontsize=16,
        title_fontsize="16",
        labelcolor=font_color,
    )
    explanation_text = (
        "% - represent share of all producers \n before or after temp. reduction "
    )
    plt.gcf().text(
        0.5,
        -0.3,
        explanation_text,
        ha="center",
        fontsize=12,
        bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=0.5"),
    )

    plt.xticks(rotation=45)
    # Set the color and width of the spines
    for spine in ["bottom", "left"]:
        bar_plot.spines[spine].set_edgecolor("#A3A3A3")
        bar_plot.spines[spine].set_linewidth(1)

    # Hide the top and right spines
    for spine in ["top", "right"]:
        bar_plot.spines[spine].set_visible(False)

    # Set the background color
    bar_plot.set_facecolor("white")

    # Get the number of unique x-values (i.e., the number of groups of bars)
    num_groups = len(sum_df[column_name].unique())

    # Create a list of labels corresponding to each bar, repeating for each group of bars
    status_labels = [label1] * num_groups + [label2] * num_groups

    # Then in your loop, use these labels to determine the total
    for p, status in zip(bar_plot.patches, status_labels):
        total = total_1 if status == label1 else total_2
        percentage = "{:.0f}%".format(100 * p.get_height() / total)
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
