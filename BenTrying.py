"""
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

from matplotlib import font_manager

fonts = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")

font_names = []
for font in fonts:
    try:
        font_names.append(font_manager.FontProperties(fname=font).get_name())
    except RuntimeError:
        print(f"Could not retrieve the name for font: {font}")
        continue

for font_name in font_names:
    print(font_name)
