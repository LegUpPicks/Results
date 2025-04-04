import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
import numpy as np
from streamlit_gsheets import GSheetsConnection

st.set_page_config(layout="wide")

# Load the data from Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)
url_all = "https://docs.google.com/spreadsheets/d/1AEiuEVQFHetOawGOWqpl2MbuPXcWsb6eeK01oRE8gGQ/edit?usp=sharing"
df_all = conn.read(spreadsheet=url_all, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])

# Reset the DataFrame index
df = df_all.reset_index(drop=True)

# Filter the DataFrame based on POTD option
option = st.sidebar.radio(
    'Choose to view only POTD or All Plays',
    ('All Picks', 'POTD')
)

if option == 'POTD':
    df = df[df['POTD'] == 1]

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter by Sport
sports = df['Sport'].unique()
selected_sport = st.sidebar.selectbox('Select Sport', options=['All'] + list(sports))

if selected_sport != 'All':
    df = df[df['Sport'] == selected_sport]

# Filter by Date Range
date_range = st.sidebar.date_input("Select Date Range", [df['Date'].min(), df['Date'].max()])

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
else:
    st.sidebar.warning("Please select both start and end dates.")

# Summary stats (No user column, just global stats)
w_count = (df['Win_Loss_Push'] == 'w').sum()
l_count = (df['Win_Loss_Push'] == 'l').sum()
p_count = (df['Win_Loss_Push'] == 'p').sum()
total_records = w_count + l_count + p_count  # Total wins, losses, and pushes

win_percentage = (w_count / total_records) * 100 if total_records > 0 else 0
total_units = df['Units Gained'].sum()

# Display metrics
st.header("Summary Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Wins", w_count)
    
with col2:
    st.metric("Total Losses", l_count)

with col3:
    st.metric("Total Pushes", p_count)

with col4:
    st.metric("Win Percentage", f"{win_percentage:.2f}%")

with col5:
    st.metric("Total Units", f"{total_units:.2f}")

# Cumulative Units Calculation
df_cumulative = df.groupby('Date').agg({'Units Gained': 'sum'}).cumsum().reset_index()
df_cumulative.rename(columns={'Units Gained': 'Units'}, inplace=True)

# Create Daily Bar Chart using Plotly
df_daily_sum = df.groupby('Date')['Units Gained'].sum().reset_index()
fig_daily = go.Figure()

fig_daily.add_trace(go.Bar(
    x=df_daily_sum['Date'],
    y=df_daily_sum['Units Gained'],
    marker=dict(color=df_daily_sum['Units Gained'].apply(lambda x: 'green' if x > 0 else 'red')),
    text=df_daily_sum['Units Gained'].round(2),
    textposition='auto',
    hoverinfo='x+y+text',
))

fig_daily.update_layout(
    title='Daily Units Won / Lost',
    xaxis_title='Date',
    yaxis_title='Units Won / Lost',
    showlegend=False,
    template='plotly_white',
    xaxis_tickangle=-45,
)

# Create Weekly Bar Chart using Plotly
df['Week'] = df['Date'].dt.to_period('W-SUN').dt.start_time  # Convert dates to start of the week
df_weekly_sum = df.groupby('Week')['Units Gained'].sum().reset_index()

fig_weekly = go.Figure()

fig_weekly.add_trace(go.Bar(
    x=df_weekly_sum['Week'],
    y=df_weekly_sum['Units Gained'],
    marker=dict(color=df_weekly_sum['Units Gained'].apply(lambda x: 'green' if x > 0 else 'red')),
    text=df_weekly_sum['Units Gained'].round(2),
    textposition='auto',
    hoverinfo='x+y+text',
))

fig_weekly.update_layout(
    title='Weekly Units Won / Lost',
    xaxis_title='Week',
    yaxis_title='Units Won / Lost',
    showlegend=False,
    template='plotly_white',
    xaxis_tickangle=-45,
)

# Display the charts
st.plotly_chart(fig_daily, key='daily_chart')

# Display the filtered DataFrame
st.header("All Data")
st.dataframe(df.sort_values(by='Date', ascending=False), hide_index=True)

#st.plotly_chart(fig_weekly, key='weekly_chart')