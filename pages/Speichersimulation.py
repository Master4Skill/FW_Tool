import streamlit as st
from streamlit_extras.app_logo import add_logo
import pandas as pd
from io import StringIO
import json as json
import numpy as np


st.set_page_config(page_title="Plotting Demo", page_icon="📈")
add_logo("resized_image.png")
st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")


def load_data(file):
    df = pd.read_csv(file, sep=";", decimal=",")
    print(df)
    return df


df_Zeitreihen = load_data("Zeitreihen/zeitreihen_22.csv")

df_input = pd.read_json("results/Input_Netz.json")


# Select the first 169 rows
df_Zeitreihen = df_Zeitreihen.head(24)
df_input = df_input.head(24)
# print(df_Zeitreihen)
# print(df_input)
