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
ηVerdichter = input_data["ηVerdichter"]
T_q_diffmax = input_data["T_q_diffmax"]
p_network = input_data["p_network"]


st.set_page_config(
    page_title="Sensitivity Analysis",
    page_icon="📊",
)
add_logo("resized_image.png")

st.sidebar.header("Sensitivity Analysis")

st.sidebar.info("identify the general impact of a temperature reduction")

st.markdown("# Sensitivity Analysis")

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
    # st.write("T:", T)
    # st.write("viscosity:", viscosity)
    return viscosity


def calc_Reynolds(flowSpeed, T):
    return flowSpeed * rR * ρ_water / water_viscosity_CoolProp(T, p_network)


def calc_pressureloss(Reynolds, flowSpeed):
    λ = 64 / Reynolds if Reynolds < 2300 else 0.3164 / Reynolds**0.25
    # st.write("Reynolds:", Reynolds)
    # st.write("λ:", λ)
    return λ * ρ_water / 2 * (flowSpeed**2) / rR * (l_Netz + ζ * rR / λ)


def calc_pumpleistung(flowSpeed, flowRate, Tvl, Trl):
    Rey_vl = calc_Reynolds(flowSpeed, Tvl)
    # st.write("Rey_vl:", Rey_vl)
    Rey_rl = calc_Reynolds(flowSpeed, Trl)
    # st.write("Rey_rl:", Rey_rl)
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
        # st.write("Netzverluste:", Netzverluste)
        Wärmelast = calc_totalLast(Netzverluste, Lastgang)
        flowRate = calc_flowRate(T_vl, Wärmelast, T_rl)
        flowSpeed = calc_flowSpeed(flowRate)
        pumpleistung = calc_pumpleistung(flowSpeed, flowRate, T_vl, T_rl)

        Netzverluste_list.append(Netzverluste)
        Wärmelast_list.append(Wärmelast)
        flowRate_list.append(flowRate)
        flowSpeed_list.append(flowSpeed)
        pumpleistung_list.append(pumpleistung)

    # st.write("T_rl_range:", T_rl_range)
    # st.write("Netzverluste_list:", Netzverluste_list)
    # st.write("Wärmelast_list:", Wärmelast_list)
    # st.write("flowRate_list:", flowRate_list)
    # st.write("flowSpeed_list:", flowSpeed_list)
    # st.write("pumpleistung_list:", pumpleistung_list)

    Netzverluste_list = [x / Netzverluste_list[0] for x in Netzverluste_list]
    Wärmelast_list = [x / Wärmelast_list[0] for x in Wärmelast_list]
    flowRate_list = [x / flowRate_list[0] for x in flowRate_list]
    flowSpeed_list = [x / flowSpeed_list[0] for x in flowSpeed_list]
    pumpleistung_list = [x / pumpleistung_list[0] for x in pumpleistung_list]

    fig, ax = plt.subplots(
        figsize=(800 / 80, 600 / 80)
    )  # figsize needs to be in inches, dpi is usually 80

    # Plotting
    ax.plot(T_rl_range, Netzverluste_list, label="Networklosses", lw=2)
    ax.plot(T_rl_range, Wärmelast_list, label="Total heat load", lw=2)
    ax.plot(T_rl_range, flowRate_list, label="Flow rate", lw=2)
    ax.plot(T_rl_range, flowSpeed_list, label="Flow speed", linestyle=":", lw=2)
    ax.plot(T_rl_range, pumpleistung_list, label="Pump power", lw=2)

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
        r"Sensitivity of the Network for $T_f$ = constant (95°C) ",
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
        # st.write("T_rl:", T_rl)
        # st.write("T_vl:", T_vl)
        Netzverluste = calc_verlust(T_vl, T_b, T_rl)
        # st.write("Netzverluste:", Netzverluste)
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

    # Plotting
    ax.plot(T_vl_range, Netzverluste_list, label="Networklosses", lw=2)
    ax.plot(T_vl_range, Wärmelast_list, label="Total heat load", lw=2)
    ax.plot(T_vl_range, flowRate_list, label="Flow rate", lw=2)
    ax.plot(T_vl_range, flowSpeed_list, label="Flow speed", linestyle=":", lw=2)
    ax.plot(T_vl_range, pumpleistung_list, label="Pump power", lw=2)

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
        r"Sensitivity of the Network for $\Delta T$ = constant (30°C) ",
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
        # st.write("Netzverluste:", Netzverluste)
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

    # Plotting
    ax.plot(T_vl_range, Netzverluste_list, label="Networklosses", lw=2)
    ax.plot(T_vl_range, Wärmelast_list, label="Total heat load", lw=2)
    ax.plot(T_vl_range, flowRate_list, label="Flow rate", lw=2)
    ax.plot(T_vl_range, flowSpeed_list, label="Flow speed", linestyle=":", lw=2)
    ax.plot(T_vl_range, pumpleistung_list, label="Pump power", lw=2)

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
        r"Sensitivity of the Network for $T_r$ = constant (45°C) ",
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
T_rl_range = np.arange(65, 40, -5)
# st.write("T_rl_range:", T_rl_range)
T_vl = 95
T_vl_range = np.arange(95, 70, -5)
# st.write("T_rl_range:", T_rl_range)
T_rl = 45
returnreduction(T_rl_range, T_vl)
flowreduction(T_vl_range, T_rl)
returnandflowreduction(T_rl_range, T_vl_range)
Lastgang = 10
returnreduction(T_rl_range, T_vl)
flowreduction(T_vl_range, T_rl)
returnandflowreduction(T_rl_range, T_vl_range)


def producer_sensitivitywp1(producer, T_range, T_rl, T_vl, current_last, option):
    COP_list = []
    flowrate_list = []  # assuming this is Volumenstrom
    poweruse_list = []
    powerout_list = []

    if option == 1:
        for T_vl in T_range:
            # Get the values
            COP = producer.calc_COP(T_vl)
            poweruse = producer.calc_Poweruse(None, T_vl, T_rl, current_last)
            # Here, I'm assuming that the flow rate for the Waermepumpe1 is constant and given by Volumenstrom_quelle.
            # If it's not constant, you'd need a method in your class to compute it.
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
            COP = producer.calc_COP(T_vl)
            poweruse = producer.calc_Poweruse(None, T_vl, T_rl, current_last)
            # Here, I'm assuming that the flow rate for the Waermepumpe1 is constant and given by Volumenstrom_quelle.
            # If it's not constant, you'd need a method in your class to compute it.
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
            COP = producer.calc_COP(T_vl)
            poweruse = producer.calc_Poweruse(None, T_vl, T_vl - 30, current_last)
            # Here, I'm assuming that the flow rate for the Waermepumpe1 is constant and given by Volumenstrom_quelle.
            # If it's not constant, you'd need a method in your class to compute it.
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

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(T_range, COP_list, label="COP", lw=2)
    ax.plot(T_range, poweruse_list, label="Electrical power use", lw=2, color="purple")
    ax.plot(
        T_range,
        flowrate_list,
        label="Heat delivered by the source",
        lw=2,
        color="green",
    )
    ax.plot(T_range, powerout_list, label="Power out", lw=2, color="orange")

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
            r"Sensitivity of the heat pump for $T_f$ = constant (95°C) ",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    elif option == 1:
        ax.set_title(
            r"Sensitivity of the heat pump for $T_r$ = constant (45°C) ",
            fontsize=16,
            color="#777777",
            fontfamily="Segoe UI SemiLight",
        )
    else:
        ax.set_title(
            r"Sensitivity of the heat pump for $\Delta T$ = constant (30°C) ",
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


def producer_sensitivitywp2(producer, T_range, T_rl, T_vl, current_last, option):
    COP_list = []
    flowrate_list = []  # assuming this is Volumenstrom
    poweruse_list = []

    if option == 1:
        for T_vl in T_range:
            # Get the values
            COP = producer.calc_COP(T_vl)
            poweruse = producer.calc_Poweruse(None, T_vl, T_rl, current_last)
            # Here, I'm assuming that the flow rate for the Waermepumpe1 is constant and given by Volumenstrom_quelle.
            # If it's not constant, you'd need a method in your class to compute it.
            flowrate = producer.calc_flowrate(T_vl)

            # Append to lists
            COP_list.append(COP)
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)

    elif option == 2:
        for T_rl in T_range:
            # Get the values
            COP = producer.calc_COP(T_vl)
            poweruse = producer.calc_Poweruse(None, T_vl, T_rl, current_last)
            # Here, I'm assuming that the flow rate for the Waermepumpe1 is constant and given by Volumenstrom_quelle.
            # If it's not constant, you'd need a method in your class to compute it.
            flowrate = producer.calc_flowrate(T_vl)

            # Append to lists
            COP_list.append(COP)
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)

    elif option == 3:
        for T_vl in T_range:
            # Get the values
            COP = producer.calc_COP(T_vl)
            poweruse = producer.calc_Poweruse(None, T_vl, T_vl - 30, current_last)
            # Here, I'm assuming that the flow rate for the Waermepumpe1 is constant and given by Volumenstrom_quelle.
            # If it's not constant, you'd need a method in your class to compute it.
            flowrate = producer.calc_flowrate(T_vl)

            # Append to lists
            COP_list.append(COP)
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)

    COP_list = [x / COP_list[0] for x in COP_list]
    poweruse_list = [x / poweruse_list[0] for x in poweruse_list]
    flowrate_list = [x / flowrate_list[0] for x in flowrate_list]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(T_range, COP_list, label="COP", lw=2)
    ax.plot(T_range, poweruse_list, label="Power use", lw=2, color="purple")
    ax.plot(T_range, flowrate_list, label="Flow rate", lw=2, color="green")

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
    ax.set_title(
        f"Sensitivity of the heat pump{' 2' if 'wp2' in globals()['__name__'] else ''} for "
        f"${'T_f' if option == 2 else 'T_r' if option == 1 else 'Delta T'}$ = constant "
        f"({95 if option == 2 else 45 if option == 1 else 30}°C) ",
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


def producer_sensitivitygeo(producer, T_range, T_rl, T_vl, current_last, option):
    T_inject_list = []
    flowrate_list = []  # assuming this is Volumenstrom
    poweruse_list = []

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

            # Append to lists
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)
            T_inject_list.append(T_inject)

    elif option == 2:
        for T_rl in T_range:
            # Get the values
            T_inject = T_rl + 2
            poweruse = producer.calc_Poweruse(None, T_vl, T_rl, current_last)
            flowrate = current_last / (producer.Tgeo - (T_rl + 2) * ρ_water * cp_water)

            # Append to lists
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)
            T_inject_list.append(T_inject)

    elif option == 3:
        for T_vl in T_range:
            # Get the values
            T_inject = T_vl - 30 + 2
            poweruse = producer.calc_Poweruse(None, T_vl, T_vl - 30, current_last)
            flowrate = current_last / (
                producer.Tgeo - (T_vl - 30 + 2) * ρ_water * cp_water
            )

            # Append to lists
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)
            T_inject_list.append(T_inject)

    poweruse_list = [x / poweruse_list[0] for x in poweruse_list]
    flowrate_list = [x / flowrate_list[0] for x in flowrate_list]
    T_inject_list = [x / T_inject_list[0] for x in T_inject_list]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(T_range, poweruse_list, label="Power use", lw=2, color="purple")
    ax.plot(
        T_range, flowrate_list, label="Flow rate", lw=2, color="yellow", linestyle="--"
    )
    ax.plot(T_range, T_inject_list, label="Injection temperature", lw=2)

    ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=1.5)

    # X and Y labels with larger font
    if option == 2:
        ax.set_xlabel("Return Temperature[°C]", fontsize=14)
    else:
        ax.set_xlabel("Flow Temperature[°C]", fontsize=14)

    # Setting x-ticks to the values in T_vl_range with larger font
    ax.set_xticks(T_range)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Adding horizontal lines for each y-tick
    for y in ax.get_yticks():
        ax.axhline(y, color="grey", linestyle="--", linewidth=0.5)

    # Removing the frame (spines)
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    # Title with larger font
    if option == 2:
        ax.set_title(
            r"Sensitivity of the geothermal unit for $T_f$ = constant (95°C) ",
            fontsize=16,
        )
    elif option == 1:
        ax.set_title(
            r"Sensitivity of the geothermal unit for $T_r$ = constant (45°C) ",
            fontsize=16,
        )
    else:
        ax.set_title(
            r"Sensitivity of the geothermal unit for $\Delta T$ = constant (30°C) ",
            fontsize=16,
        )

    # Legend with larger font
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=12,
    )

    # Display the plot (assuming you're using Streamlit)
    st.pyplot(fig)

    return


def producer_sensitivitysolar(producer, T_range, T_rl, T_vl, current_last, option):
    T_inject_list = []
    flowrate_list = []  # assuming this is Volumenstrom
    poweruse_list = []

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

            # Append to lists
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)
            T_inject_list.append(T_inject)

    elif option == 2:
        for T_rl in T_range:
            # Get the values
            T_inject = T_rl + 2
            poweruse = producer.calc_Poweruse(None, T_vl, T_rl, current_last)
            flowrate = current_last / (producer.Tgeo - (T_rl + 2) * ρ_water * cp_water)

            # Append to lists
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)
            T_inject_list.append(T_inject)

    elif option == 3:
        for T_vl in T_range:
            # Get the values
            T_inject = T_vl - 30 + 2
            poweruse = producer.calc_Poweruse(None, T_vl, T_vl - 30, current_last)
            flowrate = current_last / (
                producer.Tgeo - (T_vl - 30 + 2) * ρ_water * cp_water
            )

            # Append to lists
            poweruse_list.append(poweruse)
            flowrate_list.append(flowrate)
            T_inject_list.append(T_inject)

    poweruse_list = [x / poweruse_list[0] for x in poweruse_list]
    flowrate_list = [x / flowrate_list[0] for x in flowrate_list]
    T_inject_list = [x / T_inject_list[0] for x in T_inject_list]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(T_range, poweruse_list, label="Power use", lw=2, color="purple")
    ax.plot(
        T_range, flowrate_list, label="Flow rate", lw=2, color="yellow", linestyle="--"
    )
    ax.plot(T_range, T_inject_list, label="Injection temperature", lw=2)

    ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=1.5)

    # X and Y labels with larger font
    if option == 2:
        ax.set_xlabel("Return Temperature[°C]", fontsize=14)
    else:
        ax.set_xlabel("Flow Temperature[°C]", fontsize=14)

    # Setting x-ticks to the values in T_vl_range with larger font
    ax.set_xticks(T_range)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Adding horizontal lines for each y-tick
    for y in ax.get_yticks():
        ax.axhline(y, color="grey", linestyle="--", linewidth=0.5)

    # Removing the frame (spines)
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    # Title with larger font
    if option == 2:
        ax.set_title(
            r"Sensitivity of the geothermal unit for $T_f$ = constant (95°C) ",
            fontsize=16,
        )
    elif option == 1:
        ax.set_title(
            r"Sensitivity of the geothermal unit for $T_r$ = constant (45°C) ",
            fontsize=16,
        )
    else:
        ax.set_title(
            r"Sensitivity of the geothermal unit for $\Delta T$ = constant (30°C) ",
            fontsize=16,
        )

    # Legend with larger font
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=12,
    )

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
            # st.write(powerout)
            # Append to lists
            powerout_list.append(powerout)

    elif option == 3:
        for T_vl in T_range:
            # Get the values
            powerout = producer.calc_output(None, None, T_vl - 30)

            # Append to lists
            powerout_list.append(powerout)

    powerout_list = [x / powerout_list[0] for x in powerout_list]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(T_range, powerout_list, label="Power out", lw=2, color="orange")

    ax.invert_xaxis()
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=1.7)

    # X and Y labels with larger font
    if option == 2:
        ax.set_xlabel("Return Temperature[°C]", fontsize=14)
    else:
        ax.set_xlabel("Flow Temperature[°C]", fontsize=14)

    # Setting x-ticks to the values in T_vl_range with larger font
    ax.set_xticks(T_range)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Adding horizontal lines for each y-tick
    for y in ax.get_yticks():
        ax.axhline(y, color="grey", linestyle="--", linewidth=0.5)

    # Removing the frame (spines)
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    # Title with larger font
    if option == 2:
        ax.set_title(
            r"Sensitivity of the wast heat unit for $T_f$ = constant (95°C) ",
            fontsize=16,
        )
    elif option == 1:
        ax.set_title(
            r"Sensitivity of the waste heat unit for $T_r$ = constant (45°C) ",
            fontsize=16,
        )
    else:
        ax.set_title(
            r"Sensitivity of the waste heat unit for $\Delta T$ = constant (30°C) ",
            fontsize=16,
        )

    # Legend with larger font
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=12,
    )

    # Display the plot (assuming you're using Streamlit)
    st.pyplot(fig)

    return


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

wp1 = ep.heatpump_1(Volumenstrom_quelle_value, T_q_value, Gütegrad_value)
wp2 = ep.heatpump_2(Leistung_max_value, T_q_value, Gütegrad_value)
geo = ep.geothermal(
    Leistung_max_value,
    T_geo,
    h_förder,
    η_geo,
)

wh = ep.waste_heat(10, 120)

producer_sensitivitywp1(wp1, T_vl_range, T_rl, T_vl, Lastgang, option=1)
producer_sensitivitywp1(wp1, T_rl_range, T_rl, T_vl, Lastgang, option=2)
producer_sensitivitywp1(wp1, T_vl_range, T_rl, T_vl, Lastgang, option=3)

producer_sensitivitywp2(wp2, T_vl_range, T_rl, T_vl, Lastgang, option=1)
producer_sensitivitywp2(wp2, T_rl_range, T_rl, T_vl, Lastgang, option=2)
producer_sensitivitywp2(wp2, T_vl_range, T_rl, T_vl, Lastgang, option=3)

producer_sensitivitygeo(geo, T_vl_range, T_rl, T_vl, Lastgang, option=1)
producer_sensitivitygeo(geo, T_rl_range, T_rl, T_vl, Lastgang, option=2)
producer_sensitivitygeo(geo, T_vl_range, T_rl, T_vl, Lastgang, option=3)

Lastgang = 800
producer_sensitivitywh(wh, T_vl_range, T_rl, T_vl, Lastgang, option=1)
producer_sensitivitywh(wh, T_rl_range, T_rl, T_vl, Lastgang, option=2)
producer_sensitivitywh(wh, T_vl_range, T_rl, T_vl, Lastgang, option=3)

solar = ep.solarthermal(Leistung_max_value, T_q_value, Gütegrad_value)
producer_sensitivitysolar(solar, T_vl_range, T_rl, T_vl, Lastgang, option=1)
producer_sensitivitysolar(solar, T_rl_range, T_rl, T_vl, Lastgang, option=2)
producer_sensitivitysolar(solar, T_vl_range, T_rl, T_vl, Lastgang, option=3)
