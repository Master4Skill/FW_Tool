import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Assuming the following is the path to your file
DATA_URL = 'input_oekotool.csv'

# Load the data
@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data = load_data()

# Select cluster
cluster_id = st.selectbox("Select a cluster id:", sorted(data['cluster'].unique()))

# Filter data based on the cluster id
filtered_data = data[data['cluster'] == cluster_id]

# Generate a plot for this specific cluster id
fig, ax = plt.subplots()
ax.plot(filtered_data['time'], filtered_data['value'])
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title(f'Values over Time for Cluster {cluster_id}')

# Show the plot
st.pyplot(fig)
