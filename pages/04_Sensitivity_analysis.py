import streamlit as st
import pandas as pd
from streamlit_extras.app_logo import add_logo
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import ErzeugerparkClasses as ep
import numpy as np
from plotting_functions import (
    plot_actual_production,
    plot_sorted_production,
    plot_power_usage,
    plot_total_change,
    plot_total_emissions,
)
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from CoolProp.CoolProp import PropsSI
from streamlit_extras.app_logo import add_logo
from PIL import Image
import sys
import json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from CoolProp.CoolProp import PropsSI
import ErzeugerparkClasses as ep

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
ηVerdichter = input_data["ηVerdichter"]
T_q_diffmax = input_data["T_q_diffmax"]
p_network = input_data["p_network"]
T_Wü_delta_f = input_data["T_Wü_delta_f"]
T_Wü_delta_r = input_data["T_Wü_delta_r"]

st.set_page_config(
    page_title="Sensitivity Analysis",
    page_icon="📊",
)
# Inject custom CSS to move the Streamlit logo to the top
st.markdown("""
    <style>
        [data-testid="stSidebarNav"]::before {
            content: "";
            display: block;
            margin: 20px auto;
            height: 50px;
            width: 20px;
            background-image: url('resized_image.png');
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
        }
    </style>
    """, unsafe_allow_html=True)

add_logo("resized_image.png")

st.sidebar.header("Sensitivity Analysis")

st.sidebar.info("identify the general impact of a temperature reduction")

st.markdown("# Sensitivity Analysis")

df_input = pd.read_csv("Input_Netz.csv", delimiter=",", decimal=",")
df_input.columns = df_input.columns.str.strip()
# # st.write(df_input.columns)
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


# Calculate Netzverluste
def calc_verlust(T_vl, T_b, T_rl):
    term1 = 4 * np.pi * ((T_vl + T_rl) / 2 - T_b)
    term2 = (1 / λD) * np.log(rM / rR)
    term3 = (1 / λB) * np.log(4 * (hÜ + rM) / rM)
    term4 = (1 / λB) * np.log(((2 * (hÜ + rM) / a + 2 * rM) ** 2 + 1) ** 0.5)
    return l_Netz / 1000 * term1 / (term2 + term3 + term4)


# Calculate Wärmelast
def calc_totalLast(Netzverluste, Lastgang):
    return (Lastgang / ηWüHüs + Netzverluste) / ηWüE


def calc_VerlustProzentual(Netzverluste, Wärmelast):
    return Netzverluste / Wärmelast


# Calculate Volumenstrom und Strömungsgeschwindigkeit
def calc_flowRate(T_vl, Wärmelast, T_rl):
    return Wärmelast / (ρ_water * cp_water * (T_vl - T_rl))


def calc_flowSpeed(flowRate):
    return (flowRate / 3600) / (((np.pi) * rR**2) / 4)


# calculate Pumpleistung


def water_viscosity_CoolProp(T, p_network):
    """
    Calculate water viscosity given temperature in Celsius using CoolProp.
    T : temperature (degrees Celsius)
    """
    if pd.isna(T) or T < 0 or T > 150:  # Additional check for temperature range
        return None  # Return None or some other default value if out of range or NaN

    T_K = T + 273.15  # Convert temperature to Kelvin
    P = 101325  # Assume atmospheric pressure in Pa

    # Get viscosity (in Pa.s)
    viscosity = PropsSI("VISCOSITY", "P", p_network, "T", T_K, "Water")
    # # st.write("T:", T)
    # # st.write("viscosity:", viscosity)
    return viscosity


def calc_Reynolds(flowSpeed, T):
    return flowSpeed * rR * ρ_water / water_viscosity_CoolProp(T, p_network)


def calc_pressureloss(Reynolds, flowSpeed):
    λ = 64 / Reynolds if Reynolds < 2300 else 0.3164 / Reynolds**0.25
    # # st.write("Reynolds:", Reynolds)
    # # st.write("λ:", λ)
    return λ * ρ_water / 2 * (flowSpeed**2) / rR * (l_Netz + ζ * rR / λ)


def calc_pumpleistung(flowSpeed, flowRate, Tvl, Trl):
    Rey_vl = calc_Reynolds(flowSpeed, Tvl)
    # # st.write("Rey_vl:", Rey_vl)
    Rey_rl = calc_Reynolds(flowSpeed, Trl)
    # # st.write("Rey_rl:", Rey_rl)
    p_loss_vl = calc_pressureloss(Rey_vl, flowSpeed)
    p_loss_rl = calc_pressureloss(Rey_rl, flowSpeed)
    return ((p_loss_vl + p_loss_rl) * (flowRate / 3600)) / (ηPump * 1000)


def returnreduction(T_rl_range, T_vl):
    Netzverluste_list = []
    Wärmelast_list = []
    flowRate_list = []
    flowSpeed_list = []
    pumpleistung_list = []
    for T_rl in T_rl_range:
        Netzverluste = calc_verlust(T_vl, T_b, T_rl)
        # # st.write("Netzverluste:", Netzverluste)
        Wärmelast = calc_totalLast(Netzverluste, Lastgang)
        flowRate = calc_flowRate(T_vl, Wärmelast, T_rl)
        flowSpeed = calc_flowSpeed(flowRate)
        pumpleistung = calc_pumpleistung(flowSpeed, flowRate, T_vl, T_rl)

        Netzverluste_list.append(Netzverluste)
        Wärmelast_list.append(Wärmelast)
        flowRate_list.append(flowRate)
        flowSpeed_list.append(flowSpeed)
        pumpleistung_list.append(pumpleistung)

    # # st.write("T_rl_range:", T_rl_range)
    # # st.write("Netzverluste_list:", Netzverluste_list)
    # # st.write("Wärmelast_list:", Wärmelast_list)
    # # st.write("flowRate_list:", flowRate_list)
    # # st.write("flowSpeed_list:", flowSpeed_list)
    # # st.write("pumpleistung_list:", pumpleistung_list)

    Netzverluste_list = [x / Netzverluste_list[0] for x in Netzverluste_list]
    Wärmelast_list = [x / Wärmelast_list[0] for x in Wärmelast_list]
    flowRate_list = [x / flowRate_list[0] for x in flowRate_list]
    flowSpeed_list = [x / flowSpeed_list[0] for x in flowSpeed_list]
    pumpleistung_list = [x / pumpleistung_list[0] for x in pumpleistung_list]

    # st.write("pumpleistung_list[-1]:", 1 - pumpleistung_list[-1])
    # st.write("Netzverluste_list[-1]:", 1 - Netzverluste_list[-1])

    fig, ax = plt.subplots(
        figsize=(800 / 80, 600 / 80)
    )  # figsize needs to be in inches, dpi is usually 80

    # Plotting
    ax.plot(
        T_rl_range,
        Netzverluste_list,
        label="Network Losses",
        lw=2,
        color="#DD2525",  # Bright red color
    )
    ax.plot(
        T_rl_range,
        Wärmelast_list,
        label="Total Heat Load",
        lw=2,
        color="#EC9302",  # Warm orange color
    )
    ax.plot(
        T_rl_range,
        flowRate_list,
        label="Flow Rate",
        lw=2,
        color="#92D050",  # Fresh green color
    )
    ax.plot(
        T_rl_range,
        flowSpeed_list,
        label="Flow Speed",
        linestyle="--",
        dashes=(10, 10),
        lw=2,
        color="#3795D5",  # Light blue color
    )
    ax.plot(
        T_rl_range,
        pumpleistung_list,
        label="Pump Power",
        lw=2,
        color="#515151",  # Dark grey color
    )

    ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=1.3)

    # X and Y labels with style configuration
    ax.set_xlabel(
        "Return Temperature[°C]",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI SemiLight",
    )
    ax.set_ylabel(
        "Relative Change", fontsize=16, color="#777777", fontfamily="Segoe UI SemiLight"
    )

    # X-axis properties
    ax.xaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["bottom"].set_linewidth(1)

    # Y-axis properties
    ax.yaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_linewidth(1)

    # Setting x-ticks with style configuration
    ax.set_xticks(T_rl_range)
    ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

    # Horizontal grid lines
    ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

    # Title with style configuration
    ax.set_title(
        r"Sensitivity of the Network for $T_f$ = Constant (95°C) ",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI SemiLight",
    )

    # Legend with style configuration
    ax.legend(
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

    # Background and other spines color
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Display the plot (assuming you're using Streamlit)
    st.pyplot(fig)

    return


def returnandflowreduction(T_rl_range, T_vl_range):
    Netzverluste_list = []
    Wärmelast_list = []
    flowRate_list = []
    flowSpeed_list = []
    pumpleistung_list = []
    for T_rl, T_vl in zip(T_rl_range, T_vl_range):
        # # st.write("T_rl:", T_rl)
        # # st.write("T_vl:", T_vl)
        Netzverluste = calc_verlust(T_vl, T_b, T_rl)
        # # st.write("Netzverluste:", Netzverluste)
        Wärmelast = calc_totalLast(Netzverluste, Lastgang)
        flowRate = calc_flowRate(T_vl, Wärmelast, T_rl)
        flowSpeed = calc_flowSpeed(flowRate)
        pumpleistung = calc_pumpleistung(flowSpeed, flowRate, T_vl, T_rl)

        Netzverluste_list.append(Netzverluste)
        Wärmelast_list.append(Wärmelast)
        flowRate_list.append(flowRate)
        flowSpeed_list.append(flowSpeed)
        pumpleistung_list.append(pumpleistung)

    Netzverluste_list = [x / Netzverluste_list[0] for x in Netzverluste_list]
    Wärmelast_list = [x / Wärmelast_list[0] for x in Wärmelast_list]
    flowRate_list = [x / flowRate_list[0] for x in flowRate_list]
    flowSpeed_list = [x / flowSpeed_list[0] for x in flowSpeed_list]
    pumpleistung_list = [x / pumpleistung_list[0] for x in pumpleistung_list]

    fig, ax = plt.subplots(
        figsize=(800 / 80, 600 / 80)
    )  # figsize needs to be in inches, dpi is usually 80

    # st.write("pumpleistung_list[-1]:", 1 - pumpleistung_list[-1])
    # st.write("Netzverluste_list[-1]:", 1 - Netzverluste_list[-1])

    # Plotting
    ax.plot(
        T_vl_range,
        Netzverluste_list,
        label="Network Losses",
        lw=2,
        color="#DD2525",  # Red color for losses
    )
    ax.plot(
        T_vl_range,
        Wärmelast_list,
        label="Total Heat Load",
        lw=2,
        color="#EC9302",  # Orange color for heat
    )
    ax.plot(
        T_vl_range,
        flowRate_list,
        label="Flow Rate",
        lw=2,
        color="#92D050",  # Gold color for flow
    )
    ax.plot(
        T_vl_range,
        flowSpeed_list,
        label="Flow Speed",
        linestyle="--",
        dashes=(10, 10),
        lw=2,
        color="#3795D5",  # Blue color for speed
    )
    ax.plot(
        T_vl_range,
        pumpleistung_list,
        label="Pump Power",
        lw=2,
        color="#515151",  # Grey color for mechanical power
    )

    ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=1.3)

    # X and Y labels with style configuration
    ax.set_xlabel(
        "Flow Temperature[°C]",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI SemiLight",
    )
    ax.set_ylabel(
        "Relative Change", fontsize=16, color="#777777", fontfamily="Segoe UI SemiLight"
    )
    ax.set_ylim(None, 1.5)

    # X-axis properties
    ax.xaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["bottom"].set_linewidth(1)

    # Y-axis properties
    ax.yaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_linewidth(1)

    # Setting x-ticks with style configuration
    ax.set_xticks(T_vl_range)
    ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

    # Horizontal grid lines
    ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

    # Title with style configuration
    ax.set_title(
        r"Sensitivity of the Network for $\Delta T$ = Constant (30°C) ",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI SemiLight",
    )

    # Legend with style configuration
    ax.legend(
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

    # Background and other spines color
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Display the plot (assuming you're using Streamlit)
    st.pyplot(fig)

    return


def flowreduction(T_vl_range, T_rl):
    Netzverluste_list = []
    Wärmelast_list = []
    flowRate_list = []
    flowSpeed_list = []
    pumpleistung_list = []

    for T_vl in T_vl_range:
        Netzverluste = calc_verlust(T_vl, T_b, T_rl)
        # # st.write("Netzverluste:", Netzverluste)
        Wärmelast = calc_totalLast(Netzverluste, Lastgang)
        flowRate = calc_flowRate(T_vl, Wärmelast, T_rl)
        flowSpeed = calc_flowSpeed(flowRate)
        pumpleistung = calc_pumpleistung(flowSpeed, flowRate, T_vl, T_rl)

        Netzverluste_list.append(Netzverluste)
        Wärmelast_list.append(Wärmelast)
        flowRate_list.append(flowRate)
        flowSpeed_list.append(flowSpeed)
        pumpleistung_list.append(pumpleistung)

    Netzverluste_list = [x / Netzverluste_list[0] for x in Netzverluste_list]
    Wärmelast_list = [x / Wärmelast_list[0] for x in Wärmelast_list]
    flowRate_list = [x / flowRate_list[0] for x in flowRate_list]
    flowSpeed_list = [x / flowSpeed_list[0] for x in flowSpeed_list]
    pumpleistung_list = [x / pumpleistung_list[0] for x in pumpleistung_list]

    # st.write("pumpleistung_list[-1]:", 1 - pumpleistung_list[-1])
    # st.write("Netzverluste_list[-1]:", 1 - Netzverluste_list[-1])

    fig, ax = plt.subplots(
        figsize=(800 / 80, 600 / 80)
    )  # figsize needs to be in inches, dpi is usually 80

    # Plotting
    ax.plot(
        T_vl_range,
        Netzverluste_list,
        label="Network Losses",
        lw=2,
        color="#DD2525",  # Bright Red for Network Losses
    )
    ax.plot(
        T_vl_range,
        Wärmelast_list,
        label="Total Heat Load",
        lw=2,
        color="#EC9302",  # Warm Orange for Heat Load
    )
    ax.plot(
        T_vl_range,
        flowRate_list,
        label="Flow Rate",
        lw=2,
        color="#92D050",  # Lively gold for Flow Rate#92D050
    )
    ax.plot(
        T_vl_range,
        flowSpeed_list,
        label="Flow Speed",
        linestyle="--",
        dashes=(10, 10),
        lw=2,
        color="#3795D5",  # Sky Blue for Flow Speed
    )
    ax.plot(
        T_vl_range,
        pumpleistung_list,
        label="Pump Power",
        lw=2,
        color="#777777",  # Darker Grey for Pump Power
    )

    ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=4.5)

    # X and Y labels with style configuration
    ax.set_xlabel(
        "Flow Temperature[°C]",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI SemiLight",
    )
    ax.set_ylabel(
        "Relative Change", fontsize=16, color="#777777", fontfamily="Segoe UI SemiLight"
    )

    # X-axis properties
    ax.xaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["bottom"].set_linewidth(1)

    # Y-axis properties
    ax.yaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_linewidth(1)

    # Setting x-ticks with style configuration
    ax.set_xticks(T_vl_range)
    ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

    # Horizontal grid lines
    ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

    # Title with style configuration
    ax.set_title(
        r"Sensitivity of the Network for $T_r$ = Constant (45°C) ",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI SemiLight",
    )

    # Legend with style configuration
    ax.legend(
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

    # Background and other spines color
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Display the plot (assuming you're using Streamlit)
    st.pyplot(fig)

    return


# Extracted style configurations
def style_plot(ax, T_range, option):
    ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=1.5)
    ax.set_xlabel(
        "Return Temperature[°C]" if option == 2 else "Flow Temperature[°C]",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI SemiLight",
    )
    ax.set_ylabel(
        "Relative Change", fontsize=16, color="#777777", fontfamily="Segoe UI SemiLight"
    )
    ax.xaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["bottom"].set_linewidth(1)
    ax.yaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_linewidth(1)
    ax.set_xticks(T_range)
    ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")
    ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)
    title_options = {
        1: ["Sensitivity of the Waste Heat Pump", "for $T_r$ = Constant (45°C)"],
        2: ["Sensitivity of the Waste Heat Pump", "for $T_f$ = Constant (95°C)"],
        3: ["Sensitivity of the Waste Heat Pump", "for $\Delta T$ = Constant (30°C)"],
    }

    # Join the list with a newline character to create a multiline title
    title_text = "\n".join(title_options.get(option, ""))

    ax.set_title(
        title_text,
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
    ax.invert_xaxis()


def producer_sensitivitywp1(producer, T_range, T_rl, T_vl, current_last, option):
    COP_list = []
    flowrate_list = []  # assuming this is Volumenstrom
    poweruse_list = []
    powerout_list = []

    if option == 1:
        for T_vl in T_range:
            # Get the values
            # st.write("T_vl1:", T_vl)
            COP = producer.calc_COP(T_vl, None, None)
            # st.write("COP:", COP)
            poweruse = producer.calc_Poweruse(None, T_vl, T_rl, current_last)

            flowrate = producer.Volumenstrom_quelle
            powerout = producer.calc_output(None, T_vl, T_rl)

            # Append to lists
            COP_list.append(COP)
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)
            powerout_list.append(powerout)

    elif option == 2:
        for T_rl in T_range:
            # Get the values
            COP = producer.calc_COP(T_vl, None, None)
            poweruse = producer.calc_Poweruse(None, T_vl, T_rl, current_last)
            # Here, I'm assuming that the flow rate for the Waermepumpe1 is Constant and given by Volumenstrom_quelle.
            # If it's not Constant, you'd need a method in your class to compute it.
            flowrate = producer.Volumenstrom_quelle
            powerout = producer.calc_output(None, T_vl, T_rl)

            # Append to lists
            COP_list.append(COP)
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)
            powerout_list.append(powerout)

    elif option == 3:
        for T_vl in T_range:
            # Get the values
            # st.write("T_vl:", T_vl)
            COP = producer.calc_COP(T_vl, None, None)
            # st.write("COP:", COP)
            poweruse = producer.calc_Poweruse(None, T_vl, T_vl - 30, current_last)
            # Here, I'm assuming that the flow rate for the Waermepumpe1 is Constant and given by Volumenstrom_quelle.
            # If it's not Constant, you'd need a method in your class to compute it.
            flowrate = producer.Volumenstrom_quelle
            powerout = producer.calc_output(None, T_vl, T_rl)

            # Append to lists
            COP_list.append(COP)
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)
            powerout_list.append(powerout)

    COP_list = [x / COP_list[0] for x in COP_list]
    poweruse_list = [x / poweruse_list[0] for x in poweruse_list]
    flowrate_list = [x / flowrate_list[0] for x in flowrate_list]
    powerout_list = [x / powerout_list[0] for x in powerout_list]
    # write the last value of poweruse to streamlit
    # st.write("poweruse_list[-1]:", (1 - poweruse_list[-1]) * 100)
    # powerout
    # st.write("powerout_list[-1]:", (1 - powerout_list[-1]) * 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(T_range, COP_list, label="COP", lw=2, color="#DD2525")  # red
    ax.plot(
        T_range,
        poweruse_list,
        label="Electrical Power Consumption",
        lw=2,
        color="#1F4E79",  # dark blue
    )
    ax.plot(
        T_range,
        flowrate_list,
        label="Heat from the Source",
        lw=2,
        color="#41641A",  # dark green
    )
    ax.plot(
        T_range, powerout_list, label="Heat Output", lw=2, color="#EC9302"
    )  # orange

    ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=1.5)

    # X and Y labels with style configuration
    if option == 2:
        ax.set_xlabel(
            "Return Temperature[°C]",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    else:
        ax.set_xlabel(
            "Flow Temperature[°C]",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    ax.set_ylabel(
        "Relative Change", fontsize=16, color="#777777", fontfamily="Segoe UI SemiLight"
    )

    # X-axis properties
    ax.xaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["bottom"].set_linewidth(1)

    # Y-axis properties
    ax.yaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_linewidth(1)

    # Setting x-ticks with style configuration
    ax.set_xticks(T_range)
    ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

    # Horizontal grid lines
    ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

    # Title with style configuration
    if option == 2:
        ax.set_title(
            r"Sensitivity of the Waste Heat Pump for $T_f$ = Constant (95°C)",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    elif option == 1:
        ax.set_title(
            r"Sensitivity of the Waste Heat Pump for $T_r$ = Constant (45°C)",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    else:
        ax.set_title(
            r"Sensitivity of the Waste Heat Pump for $\Delta T$ = Constant (30°C)",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )

    # Legend with style configuration
    ax.legend(
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

    # Background and other spines color
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    style_plot(ax, T_range, option)

    # Display the plot (assuming you're using Streamlit)
    st.pyplot(fig)

    return


def producer_sensitivitywp2(producer, T_range, T_rl, T_vl, T_q, current_last, option):
    COP_list = []
    flowrate_list = []  # assuming this is Volumenstrom
    poweruse_list = []

    if option == 1:
        for T_vl in T_range:
            # Get the values
            COP = producer.calc_COP(T_vl, None, T_q)
            poweruse = producer.calc_Poweruse(5000, T_vl, T_rl, current_last)
            # Here, I'm assuming that the flow rate for the Waermepumpe1 is Constant and given by Volumenstrom_quelle.
            # If it's not Constant, you'd need a method in your class to compute it.
            flowrate = producer.calc_flowrate(T_vl)

            # Append to lists
            COP_list.append(COP)
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)

    elif option == 2:
        for T_rl in T_range:
            # Get the values
            COP = producer.calc_COP(T_vl, None, T_q)
            poweruse = producer.calc_Poweruse(5000, T_vl, T_rl, current_last)
            # Here, I'm assuming that the flow rate for the Waermepumpe1 is Constant and given by Volumenstrom_quelle.
            # If it's not Constant, you'd need a method in your class to compute it.
            flowrate = producer.calc_flowrate(T_vl)

            # Append to lists
            COP_list.append(COP)
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)

    elif option == 3:
        for T_vl in T_range:
            # Get the values
            COP = producer.calc_COP(T_vl, None, T_q)
            poweruse = producer.calc_Poweruse(5000, T_vl, T_vl - 30, current_last)
            # Here, I'm assuming that the flow rate for the Waermepumpe1 is Constant and given by Volumenstrom_quelle.
            # If it's not Constant, you'd need a method in your class to compute it.
            flowrate = producer.calc_flowrate(T_vl)

            # Append to lists
            COP_list.append(COP)
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)

    COP_list = [x / COP_list[0] for x in COP_list]
    poweruse_list = [x / poweruse_list[0] for x in poweruse_list]
    flowrate_list = [x / flowrate_list[0] for x in flowrate_list]

    # write the last value of poweruse to streamlit
    # st.write("cop_list[-1]:", (1 - COP_list[-1]) * 100)
    # st.write("powerout_list[-1]:", 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(T_range, COP_list, label="COP", lw=2, color="#DD2525")  # red
    ax.plot(
        T_range, poweruse_list, label="Power Consumption", lw=2, color="#1F4E79"
    )  # dark blue
    ax.plot(
        T_range, flowrate_list, label="Flow Rate", lw=2, color="#92D050"
    )  # light gold

    ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=1.5)

    ax.set_xlabel(
        "Return Temperature[°C]" if option == 2 else "Flow Temperature[°C]",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI SemiLight",
    )
    ax.set_ylabel(
        "Relative Change", fontsize=16, color="#777777", fontfamily="Segoe UI SemiLight"
    )
    ax.xaxis.label.set_color("#A3A3A3")
    ax.yaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
    ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_linewidth(1)

    # ... (keep other settings)

    # Setting x-ticks with style configuration
    ax.set_xticks(T_range)
    ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

    # Horizontal grid lines
    ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

    # Title with style configuration
    title_options = {
        1: ["Sensitivity of the Ambient Heat Pump", "for $T_r$ = Constant (45°C)"],
        2: ["Sensitivity of the Ambient Heat Pump", "for $T_f$ = Constant (95°C)"],
        3: ["Sensitivity of the Ambient Heat Pump", "for $\Delta T$ = Constant (30°C)"],
    }

    # Join the list with a newline character to create a multiline title
    title_text = "\n".join(title_options.get(option, ""))

    ax.set_title(
        title_text,
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI SemiLight",
    )

    # Legend with style configuration
    ax.legend(
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
    # Background and other spines color
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Display the plot (assuming you're using Streamlit)
    st.pyplot(fig)
    return


def producer_sensitivitygeo(producer, T_range, T_rl, T_vl, current_last, option):
    T_inject_list = []
    flowrate_list = []  # assuming this is Volumenstrom
    poweruse_list = []
    cop_list = []

    def calc_Poweruse(self, hour, Tvl, Trl, current_last):
        # Placeholder calculation:
        V = current_last / (self.Tgeo - (Trl + 2) * ρ_water * cp_water)
        P = V / 3600 * ρ_water * 9.81 * self.h_förder / self.η_geo
        return P

    if option == 1:
        for T_vl in T_range:
            # Get the values
            T_inject = T_rl + 2
            poweruse = producer.calc_Poweruse(None, T_vl, T_rl, current_last)
            flowrate = current_last / (producer.Tgeo - (T_rl + 2) * ρ_water * cp_water)
            cop = producer.calc_COP(None, T_rl, None)

            # Append to lists
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)
            T_inject_list.append(T_inject)
            cop_list.append(cop)

    elif option == 2:
        for T_rl in T_range:
            # Get the values
            T_inject = T_rl + 2
            poweruse = producer.calc_Poweruse(None, T_vl, T_rl, current_last)
            flowrate = current_last / (producer.Tgeo - (T_rl + 2) * ρ_water * cp_water)
            cop = producer.calc_COP(None, T_rl, None)

            # Append to lists
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)
            T_inject_list.append(T_inject)
            cop_list.append(cop)

    elif option == 3:
        for T_vl in T_range:
            # Get the values
            T_inject = T_vl - 30 + 2
            poweruse = producer.calc_Poweruse(None, T_vl, T_vl - 30, current_last)
            flowrate = current_last / (
                producer.Tgeo - (T_vl - 30 + 2) * ρ_water * cp_water
            )
            cop = producer.calc_COP(None, T_vl - 30, None)

            # Append to lists
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)
            T_inject_list.append(T_inject)
            cop_list.append(cop)

    poweruse_list = [x / poweruse_list[0] for x in poweruse_list]
    flowrate_list = [x / flowrate_list[0] for x in flowrate_list]
    T_inject_list = [x / T_inject_list[0] for x in T_inject_list]
    cop_list = [x / cop_list[0] for x in cop_list]

    # write the last value of poweruse to streamlit
    # st.write("cop_list[-1]:", (1 - cop_list[-1]) * 100)
    # powerout
    # st.write("powerout_list[-1]:", 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(
        T_range,
        poweruse_list,
        label="Electrical Power Consumption",
        lw=2,
        color="#B77201",  # Purple-like color
    )
    ax.plot(
        T_range,
        flowrate_list,
        label="Flow Rate",
        lw=2,
        color="#92D050",  # Golden yellow color
        linestyle="--",
        dashes=(10, 10),
    )
    ax.plot(
        T_range,
        T_inject_list,
        label="Injection Temperature",
        lw=2,
        color="#3795D5",  # Light blue color
    )

    ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=1.5)

    # X and Y labels with style configuration
    if option == 2:
        ax.set_xlabel(
            "Return Temperature[°C]",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    else:
        ax.set_xlabel(
            "Flow Temperature[°C]",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    ax.set_ylabel(
        "Performance Metrics",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI SemiLight",
    )

    # X-axis properties
    ax.xaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["bottom"].set_linewidth(1)

    # Y-axis properties
    ax.yaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_linewidth(1)

    # Setting x-ticks with style configuration
    ax.set_xticks(T_range)
    ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

    # Horizontal grid lines
    ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

    # Title with style configuration
    if option == 2:
        ax.set_title(
            r"Sensitivity of The Geothermal Unit for $T_f$ = Constant (95°C)",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    elif option == 1:
        ax.set_title(
            r"Sensitivity of The Geothermal Unit for $T_r$ = Constant (45°C)",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    else:
        ax.set_title(
            r"Sensitivity of The Geothermal Unit for $\Delta T$ = Constant (30°C)",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )

    # Legend with style configuration
    ax.legend(
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

    # Background and other spines color
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Display the plot (assuming you're using Streamlit)
    st.pyplot(fig)

    return


def producer_sensitivitysolar(
    producer, T_range, T_rl, T_vl, solar_area, k_s_1, k_s_2, α, τ, option
):
    powerout_list = []
    T_m_list = []
    # st.write(f"k_s_1={k_s_1}")
    # st.write(f"k_s_2={k_s_2}")
    # st.write(f"α={α}")
    # st.write(f"τ={τ}")

    def calc_output(Tvl, Trl):
        T_u = 20
        T_m = (Tvl + T_Wü_delta_f + Trl + T_Wü_delta_r) / 2
        # st.write(f"T_m = {T_m}")
        a = 0.3 * α * τ
        b = k_s_1 * (T_m - T_u)
        c = k_s_2 * (T_m - T_u) ** 2
        # st.write(f"a = {a}")
        # st.write(f"b = {b}")
        # st.write(f"c = {c}")
        # st.write(f"prelim = {(a - b - c)}")
        x = (a - b - c) * solar_area
        # st.write(f"x = {x}")
        # T_m = 77
        return [x if x > 0 else 0][0]

    def calc_T_m(Tvl, Trl):
        T_m = (Tvl + T_Wü_delta_f + Trl + T_Wü_delta_r) / 2
        return T_m

    if option == 1:
        for T_vl in T_range:
            # Get the values
            poweruse = calc_output(T_vl, T_rl)
            T_m = calc_T_m(T_vl, T_rl)
            # Append to lists
            powerout_list.append(poweruse)
            T_m_list.append(T_m)

    elif option == 2:
        for T_rl in T_range:
            # Get the values
            poweruse = calc_output(T_vl, T_rl)
            T_m = calc_T_m(T_vl, T_rl)
            # Append to lists
            powerout_list.append(poweruse)
            T_m_list.append(T_m)

    elif option == 3:
        for T_vl in T_range:
            # Get the values
            poweruse = calc_output(T_vl, T_vl - 30)
            T_m = calc_T_m(T_vl, T_vl - 30)
            # Append to lists
            powerout_list.append(poweruse)
            T_m_list.append(T_m)

    powerout_list = [x / powerout_list[0] for x in powerout_list]
    fig, ax = plt.subplots(figsize=(10, 6))

    # write the last value of poweruse to streamlit
    # st.write("powerout_list[-1]:", (1 - powerout_list[-1]) * 100)
    # powerout
    # # st.write("powerout_list[-1]:", (1 - power_list[-1])*100)

    # Assuming the setup of the figure and axis (fig, ax) has been done previously

    # Plotting
    ax.plot(
        T_range,
        powerout_list,
        label="Heat Output",
        lw=2,
        color="#EC9302",
    )

    ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=1.5)

    # X and Y labels with style configuration
    if option == 2:
        ax.set_xlabel(
            "Return Temperature[°C]",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    else:
        ax.set_xlabel(
            "Flow Temperature[°C]",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    ax.set_ylabel(
        "Relative Change", fontsize=16, color="#777777", fontfamily="Segoe UI SemiLight"
    )

    # X-axis properties
    ax.xaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["bottom"].set_linewidth(1)

    # Y-axis properties
    ax.yaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_linewidth(1)

    # Setting x-ticks with style configuration
    ax.set_xticks(T_range)
    ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

    # Horizontal grid lines
    ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

    # Title with style configuration
    if option == 2:
        ax.set_title(
            r"Sensitivity of the Solarthermal Collector for $T_f$ = Constant (95°C)",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    elif option == 1:
        ax.set_title(
            r"Sensitivity of the Solarthermal Collector for $T_r$ = Constant (45°C)",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    else:
        ax.set_title(
            r"Sensitivity of the Solarthermal Collector for $\Delta T$ = Constant (30°C)",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )

    # Legend with style configuration
    ax.legend(
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

    # Background and other spines color
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Display the plot (assuming you're using Streamlit)
    st.pyplot(fig)

    return


def producer_sensitivitywh(producer, T_range, T_rl, T_vl, current_last, option):
    powerout_list = []
    resttemp_list = []

    if option == 1:
        for T_vl in T_range:
            # Get the values
            powerout = producer.calc_output(None, None, T_rl)

            # Append to lists
            powerout_list.append(powerout)

    elif option == 2:
        for T_rl in T_range:
            # Get the values
            powerout = producer.calc_output(None, None, T_rl)
            # # st.write(powerout)
            # Append to lists
            powerout_list.append(powerout)

    elif option == 3:
        for T_vl in T_range:
            # Get the values
            powerout = producer.calc_output(None, None, T_vl - 30)

            # Append to lists
            powerout_list.append(powerout)

    powerout_list = [x / powerout_list[0] for x in powerout_list]

    # st.write("powerout_list[-1]:", (1 - powerout_list[-1]) * 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(
        T_range, powerout_list, label="Heat Output", lw=2, color="#EB8585"
    )  # orange color

    ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=1.7)

    # X and Y labels with style configuration
    if option == 2:
        ax.set_xlabel(
            "Return Temperature[°C]",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    else:
        ax.set_xlabel(
            "Flow Temperature[°C]",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )

    ax.set_ylabel(
        "Heat Output [kW]",
        fontsize=16,
        color="#777777",
        fontfamily="Segoe UI SemiLight",
    )

    # X-axis properties
    ax.xaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="x", colors="#A3A3A3", direction="out", which="both")
    ax.spines["bottom"].set_edgecolor("#A3A3A3")
    ax.spines["bottom"].set_linewidth(1)

    # Y-axis properties
    ax.yaxis.label.set_color("#A3A3A3")
    ax.tick_params(axis="y", colors="#A3A3A3", direction="out", which="both")
    ax.spines["left"].set_edgecolor("#A3A3A3")
    ax.spines["left"].set_linewidth(1)

    # Setting x-ticks with style configuration
    ax.set_xticks(T_range)
    ax.tick_params(axis="both", which="major", labelsize=16, colors="#777777")

    # Horizontal grid lines
    ax.yaxis.grid(color="#C4C4C4", linestyle="--", linewidth=0.5)

    # Title with style configuration
    if option == 2:
        ax.set_title(
            r"Sensitivity of the waste heat unit for $T_f$ = Constant (95°C)",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    elif option == 1:
        ax.set_title(
            r"Sensitivity of the waste heat unit for $T_r$ = Constant (45°C)",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    else:
        ax.set_title(
            r"Sensitivity of the waste heat unit for $\Delta T$ = Constant (30°C)",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )

    # Legend with style configuration
    ax.legend(
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

    # Background and other spines color
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Display the plot (assuming you're using Streamlit)
    st.pyplot(fig)

    return


T_b = 10
Lastgang = 15000
T_rl_range = np.arange(65, 40, -5)  # (65, 30, -5)
# # st.write("T_rl_range:", T_rl_range)
T_vl = 95
T_vl_range = np.arange(95, 70, -5)  # (95, 60, -5)
# # st.write("T_rl_range:", T_rl_range)
T_rl = 45
returnandflowreduction(T_rl_range, T_vl_range)
flowreduction(T_vl_range, T_rl)
returnreduction(T_rl_range, T_vl)


Volumenstrom_quelle_value = 85
T_q_value = 25
Gütegrad_value = 0.8
Lastgang = 15000
T_vl = 95
T_rl = 45

Leistung_max_value = 5000

h_förder = 2000
T_geo = 100
η_geo = 0.8

wp1 = ep.heatpump_1(Volumenstrom_quelle_value, T_q_value, Gütegrad_value, None)
wp2 = ep.heatpump_2(Leistung_max_value, T_q_value, Gütegrad_value, None)
geo = ep.geothermal(
    Leistung_max_value,
    T_geo,
    h_förder,
    η_geo,
    None,
)

wh = ep.waste_heat(10, 120, None)

producer_sensitivitywp1(wp1, T_vl_range, T_rl, T_vl, Lastgang, option=3)
producer_sensitivitywp1(wp1, T_vl_range, T_rl, T_vl, Lastgang, option=1)
producer_sensitivitywp1(wp1, T_rl_range, T_rl, T_vl, Lastgang, option=2)

producer_sensitivitywp2(wp2, T_vl_range, T_rl, T_vl, 25, Lastgang, option=3)
producer_sensitivitywp2(wp2, T_vl_range, T_rl, T_vl, 25, Lastgang, option=1)
producer_sensitivitywp2(wp2, T_rl_range, T_rl, T_vl, 25, Lastgang, option=2)

producer_sensitivitygeo(geo, T_vl_range, T_rl, T_vl, Lastgang, option=3)
producer_sensitivitygeo(geo, T_vl_range, T_rl, T_vl, Lastgang, option=1)
producer_sensitivitygeo(geo, T_rl_range, T_rl, T_vl, Lastgang, option=2)

Lastgang = 800
producer_sensitivitywh(wh, T_vl_range, T_rl, T_vl, Lastgang, option=3)
producer_sensitivitywh(wh, T_vl_range, T_rl, T_vl, Lastgang, option=1)
producer_sensitivitywh(wh, T_rl_range, T_rl, T_vl, Lastgang, option=2)

solar_area = 5000

k_s_1 = 1.5 / 1000  # Dividing by 1000 to convert to the appropriate unit
k_s_2 = 0.005 / 1000  # Dividing by 1000 to convert to the appropriate unit
α = 0.9
τ = 0.9
erzeuger_color = "black"  # Assuming no color is set by default, you might want to provide an actual default color

solar = ep.solarthermal(
    solar_area,
    k_s_1,
    k_s_2,
    α,
    τ,
    None,
    color=erzeuger_color,
    co2_emission_factor=0,
)
producer_sensitivitysolar(
    solar, T_vl_range, T_rl, T_vl, solar_area, k_s_1, k_s_2, α, τ, option=3
)
producer_sensitivitysolar(
    solar, T_rl_range, T_rl, T_vl, solar_area, k_s_1, k_s_2, α, τ, option=1
)
producer_sensitivitysolar(
    solar, T_vl_range, T_rl, T_vl, solar_area, k_s_1, k_s_2, α, τ, option=2
)
