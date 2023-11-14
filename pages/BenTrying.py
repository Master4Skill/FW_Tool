"""

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



# Create a list of dictionaries for each instance of WP1
# Define parameters for each instance
p2_params = [{'K': 1000, 'PL': 0.2, 'name': 'WP1_1'}, {'K': 900, 'PL': 0.25, 'name': 'WP1_2'}, {'K': 800, 'PL': 0.3, 'name': 'WP1_3'}]

# Create variables for each instance
f2 = {p['name']: LpVariable.dicts("ρ_in_"+p['name']+"_elec_t", I, lowBound=0, cat="Continuous") for p in p2_params}
g2 = {p['name']: LpVariable.dicts("ρ_out_"+p['name']+"_heat_t", I, lowBound=0, cat="Continuous") for p in p2_params}
x2 = {p['name']: LpVariable.dicts(p['name']+"_ON_OFF", I, lowBound=0, upBound=1, cat="Integer") for p in p2_params}

# Define constraints for each instance
for i in range(len(P_to_dem)):
    for p in p2_params:
        A_p2_in = 0.45 * (60 + 273.15) / (60 - Flusstemperatur[i])
        m += g2[p['name']][i] == f2[p['name']][i] * A_p2_in
        m += g2[p['name']][i] >= x2[p['name']][i] * p['K'] * p['PL']
        m += g2[p['name']][i] <= x2[p['name']][i] * p['K']

    m += e_imp[i] == sum(f2[p['name']][i] for p in p2_params) + f31[i] + f4[i] + e_exp[i]
    m += g1[i] + sum(g2[p['name']][i] for p in p2_params) + g3[i] + g4[i] + g5[i] + g6[i] + g72[i] + s_out[i] == s_in[i] + P

""" """
Currently, your optimization model is designed to handle fixed parameters for each "Erzeuger" (generator) type. 
If you want to allow multiple instances of the same Erzeuger type but with different parameters, you need to add a new dimension to your model. 

In this case, instead of treating each Erzeuger type as a separate entity, you could create a list of dictionaries for each type of Erzeuger, 
where each dictionary contains the parameters for a particular instance of that Erzeuger type. Then, you would need to adjust your decision variables 
and constraints accordingly to handle this new structure.

Here is a simplified example of how you might approach this. The example just demonstrates creating multiple instances for a single type of Erzeuger, 
but the idea could be extended to the other types.
In the code snippet above, I create separate dictionaries for the decision variables and parameters of each instance of WP1. 
The constraints are also updated to handle the new structure. This approach can be extended to all the other Erzeuger types.

Remember that this modification will increase the size of your optimization model, which may increase the computational time. 
You'll need to adjust the `p2_params` list according to your specific use case. 
You might want to consider creating a more systematic way of generating these lists if the current method is too manual.




    # Abwärme
    K_p1 = 700

    # Waermepumpe1
    K_p2 = 1000
    PL_p2 = 0.2

    # Waermepumpe2
    K_p3 = 0
    PL_p3 = 0.2

    # Geothermie
    K_p4 = 5000
    PL_p4 = 0.2

    # Solar
    K_p5 = 0
    PL_p5 = 0

    # Spizenkessel
    K_p6 = 10000
    PL_p6 = 0.01

    # BHKW
    K_p7 = 0
    PL_p7 = 0.1

        df_results["pressure_loss_vl"] = df_results.apply(
        lambda row: calc_pressureloss(
            calc_Reynolds(row["Strömungsgeschwindigkeit_vor"], row["T_vl_vor"]),
            row["Strömungsgeschwindigkeit_vor"],
        ),
        axis=1,
    )
    df_results["pressure_loss_rl"] = df_results.apply(
        lambda row: calc_pressureloss(
            calc_Reynolds(row["Strömungsgeschwindigkeit_vor"], Trl_vor),
            row["Strömungsgeschwindigkeit_vor"],
        ),
        axis=1,
    )
    df_results["Reynolds_vl"] = df_results.apply(
        lambda row: calc_Reynolds(row["Strömungsgeschwindigkeit_vor"], row["T_vl_vor"]),
        axis=1,
    )
    df_results["Reynolds_rl"] = df_results.apply(
        lambda row: calc_Reynolds(row["Strömungsgeschwindigkeit_nach"], Trl_vor),
        axis=1,
    )
    df_results["viscosity_vor"] = df_results.apply(
        lambda row: water_viscosity_CoolProp(row["T_vl_vor"], p_network), axis=1
    )
    print(df_results.loc[0, "viscosity_vor"])
    df_results["viscosity_nach"] = df_results.apply(
        lambda row: water_viscosity_CoolProp(row["T_vl_nach"], p_network), axis=1
    )

"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from CoolProp.CoolProp import PropsSI
from streamlit_extras.app_logo import add_logo
from PIL import Image
import sys
import json
from scipy.interpolate import interp1d

# Load the data from the json file
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
p_network = input_data["p_network"]

Tvl_max_vor = input_data["Tvl_max_vor"]
Tvl_min_vor = input_data["Tvl_min_vor"]
Trl_vor = input_data["Trl_vor"]
Tvl_max_nach = input_data["Tvl_max_nach"]
Tvl_min_nach = input_data["Tvl_min_nach"]
Trl_nach = input_data["Trl_nach"]

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


def plot_char_values(df_results):
    # Ensure your DataFrame is structured with a 'Mode' column to differentiate the three modes
    # Example:
    #    Category        Value  Mode
    # 0  E_stored        1000  Mode1
    # 1  E_stored        1200  Mode2
    # 2  E_stored        1100  Mode3
    # 3  g1              1500  Mode1
    # 4  g1              1600  Mode2
    # 5  g1              1400  Mode3
    # ...

    # Define labels, titles, etc.
    title = "Comparison of Modes"
    x_label = "Categories"
    y_label = "Values (in units)"

    # Define color palette, font color, and font family
    palette = sns.color_palette("viridis", n_colors=3)  # Adjust as necessary
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

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
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

    # Adding values above the bars
    for p in bar_plot.patches:
        height = p.get_height()
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


# Example percentages
percentages = [100, 97, 92, 0, 97, 100, 83, 80, 100]

# Rearrange the list
rearranged_percentages = []
for i in range(3):  # Assuming there are 3 modes
    rearranged_percentages.extend(percentages[i::3])

# Output
# print("Original: ", percentages)
# print("Rearranged: ", rearranged_percentages)


def read_data2(filename, start_hour, hours):
    data = pd.read_csv(filename, sep=";", header=None)

    # Convert the strompreise (electricity import prices) column to float
    data.iloc[:, 0] = data.iloc[:, 0].astype(str).str.replace(",", ".").astype(float)

    # Convert the strompreise_export (electricity export prices) column to float
    data.iloc[:, 1] = data.iloc[:, 1].astype(str).str.replace(",", ".").astype(float)

    # For gaspreise, take each value for 24 hours and only use the first 365 of them
    gaspreise_raw = data.iloc[:365, 2].astype(str).str.replace(",", ".").astype(float)
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


file_name = "Zeitreihen/preise2.csv"
strompreise, strompreise_export, gaspreise = read_data2(file_name, 1, 8759)

# calculate the average value
average_strompreise = sum(strompreise) / len(strompreise)
average_strompreise_export = sum(strompreise_export) / len(strompreise_export)
average_gaspreise = sum(gaspreise) / len(gaspreise)


file_path = "results/COP_vor_df.json"


# Lesen der JSON-Datei in einen DataFrame
COP_df_imported = pd.read_json(file_path, orient="columns")
COP_df_imported.fillna(COP_df_imported.mean(), inplace=True)
# find minimum COP for each producer in there
min_cop = COP_df_imported.mean(axis=0)
# print it
print(min_cop)

# print averages
# print(average_strompreise)
# print(average_strompreise_export)
# print(average_gaspreise)


import pandas as pd

# Your data
data = {
    "Category": ["Endenergiebedarf", "Kosten", "Emissionen"],
    "Strom": [6481772, 10017060, 29551],
    "Gas": [6697031, 8034763, 22411],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame as a JSON file
df.to_json("FW_Tool_Results.json", orient="split", indent=4)

print(df)

# read in df
df = pd.read_json("FW_Tool_Results.json", orient="split")
