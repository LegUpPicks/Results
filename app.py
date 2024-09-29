import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import calendar
import numpy as np
from streamlit_gsheets import GSheetsConnection

st.set_page_config(layout="wide")
url = "https://docs.google.com/spreadsheets/d/1XoVHcy6qqwKKT7HiIb5CKwv32_1Ce1fhl5XoPW-lREI/edit?usp=sharing"

conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read(spreadsheet=url, usecols=[0, 1, 2, 3, 4, 5, 6, 7])

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sidebar for sport filtering
sports = df['Sport'].unique()
selected_sport = st.sidebar.selectbox('Select Sport', options=['All'] + list(sports))

# Filter data based on selected sport
if selected_sport != 'All':
    df = df[df['Sport'] == selected_sport]

# Calculations for summary statistics
w_count = (df['Win_Loss_Push'] == 'w').sum()
l_count = (df['Win_Loss_Push'] == 'l').sum()
p_count = (df['Win_Loss_Push'] == 'p').sum()
total_records = w_count + l_count + p_count  # Total wins, losses, and pushes

# Calculate win percentage based on total records
win_percentage = (w_count / total_records) * 100 if total_records > 0 else 0
total_units = df['Units_W_L'].sum()

# Display summary statistics with rounding
st.header("Summary Statistics")
st.write(f"Total Wins: {w_count}, Total Losses: {l_count}, Total Pushes: {p_count}")
st.write(f"Win Percentage: {win_percentage:.2f}%")
st.write(f"Total Units: {total_units:.2f}")

# Cumulative units line chart
df_cumulative = df.groupby('Date').agg({'Units_W_L': 'sum'}).cumsum().reset_index()
df_cumulative.rename(columns={'Units_W_L': 'Units'}, inplace=True)

fig = px.line(df_cumulative, x='Date', y='Units', title='Cumulative Units Over Time')
st.plotly_chart(fig)

# Summary table by sport
summary_table = df.groupby('Sport')['Units_W_L'].sum().reset_index()
summary_table.rename(columns={'Units_W_L': 'Units'}, inplace=True)
summary_table['Units'] = summary_table['Units'].round(2)

# Sort summary table by Units descending
summary_table = summary_table.sort_values(by='Units', ascending=False)

st.subheader("Units Summary by Sport")
st.table(summary_table)

# Calendar for daily units
calendar_data = df.groupby(df['Date'].dt.date)['Units_W_L'].sum().reset_index()
calendar_data['Date'] = pd.to_datetime(calendar_data['Date'])

# Sort the dataframe by Date descending
df = df.sort_values(by='Date', ascending=False)

st.header('Full Data')
st.dataframe(df)
