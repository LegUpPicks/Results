import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
import numpy as np
from streamlit_gsheets import GSheetsConnection

# Setup
st.set_page_config(layout="wide")

url = "https://docs.google.com/spreadsheets/d/1XoVHcy6qqwKKT7HiIb5CKwv32_1Ce1fhl5XoPW-lREI/edit?usp=sharing"

conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read(spreadsheet=url, usecols=[0, 1, 2, 3, 4, 5, 6, 7,8])

df['Date'] = pd.to_datetime(df['Date'])

# Sidebar
sports = df['Sport'].unique()
selected_sport = st.sidebar.selectbox('Select Sport', options=['All'] + list(sports))

# Filter by sport
if selected_sport != 'All':
    df = df[df['Sport'] == selected_sport]

# Date filter
date_range = st.sidebar.date_input("Select Date Range", [df['Date'].min(), df['Date'].max()])

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
else:
    st.sidebar.warning("Please select both start and end dates.")

# Filter data for POTD == 1
df_potd = df[df['POTD'] == 1]

# Summary stats for POTD == 1
w_count_potd = (df_potd['Win_Loss_Push'] == 'w').sum()
l_count_potd = (df_potd['Win_Loss_Push'] == 'l').sum()
p_count_potd = (df_potd['Win_Loss_Push'] == 'p').sum()
total_records_potd = w_count_potd + l_count_potd + p_count_potd  # Total wins, losses, and pushes for POTD == 1

win_percentage_potd = (w_count_potd / total_records_potd) * 100 if total_records_potd > 0 else 0
total_units_potd = df_potd['Units_W_L'].sum()


# Continue with existing code to display overall summary stats
w_count = (df['Win_Loss_Push'] == 'w').sum()
l_count = (df['Win_Loss_Push'] == 'l').sum()
p_count = (df['Win_Loss_Push'] == 'p').sum()
total_records = w_count + l_count + p_count  # Total wins, losses, and pushes

win_percentage = (w_count / total_records) * 100 if total_records > 0 else 0
total_units = df['Units_W_L'].sum()

# Continue with existing overall summary section
st.header("Summary Statistics Overall")

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

# Display POTD stats
st.header("POTD Summary Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Wins (POTD)", w_count_potd)
    
with col2:
    st.metric("Total Losses (POTD)", l_count_potd)

with col3:
    st.metric("Total Pushes (POTD)", p_count_potd)

with col4:
    st.metric("Win Percentage (POTD)", f"{win_percentage_potd:.2f}%")

with col5:
    st.metric("Total Units (POTD)", f"{total_units_potd:.2f}")

# Calculate cumulative units chart
df_cumulative = df.groupby('Date').agg({'Units_W_L': 'sum'}).cumsum().reset_index()
df_cumulative.rename(columns={'Units_W_L': 'Units'}, inplace=True)

# Min and Max of the cumulative units for dynamic y-axis scaling
y_min = df_cumulative['Units'].min() - 10
y_max = df_cumulative['Units'].max() + 10

# Sum the Units_W_L for each day
df_daily_sum = df.groupby('Date')['Units_W_L'].sum().reset_index()

# Create Daily Bar Chart using Plotly
fig_daily = go.Figure()

# Add bars for positive and negative Units_W_L (Green for wins, Red for losses)
fig_daily.add_trace(go.Bar(
    x=df_daily_sum['Date'],
    y=df_daily_sum['Units_W_L'],
    marker=dict(color=df_daily_sum['Units_W_L'].apply(lambda x: 'green' if x > 0 else 'red')),
    text=df_daily_sum['Units_W_L'].round(2),
    textposition='auto',
    hoverinfo='x+y+text',
))

# Customize daily bar chart
fig_daily.update_layout(
    title='Daily Units Won / Lost',
    xaxis_title='Date',
    yaxis_title='Units Won / Lost',
    showlegend=False,
    template='plotly_white',
    xaxis_tickangle=-45,
)

# Create Weekly Bar Chart using Plotly
df['Week'] = df['Date'].dt.to_period('W').dt.start_time  # Convert dates to start of the week
df_weekly_sum = df.groupby('Week')['Units_W_L'].sum().reset_index()

fig_weekly = go.Figure()

# Add bars for weekly data (Green for wins, Red for losses)
fig_weekly.add_trace(go.Bar(
    x=df_weekly_sum['Week'],
    y=df_weekly_sum['Units_W_L'],
    marker=dict(color=df_weekly_sum['Units_W_L'].apply(lambda x: 'green' if x > 0 else 'red')),
    text=df_weekly_sum['Units_W_L'].round(2),
    textposition='auto',
    hoverinfo='x+y+text',
))

# Customize weekly bar chart
fig_weekly.update_layout(
    title='Weekly Units Won / Lost',
    xaxis_title='Week',
    yaxis_title='Units Won / Lost',
    showlegend=False,
    template='plotly_white',
    xaxis_tickangle=-45,
)

# Display the weekly and daily 
st.plotly_chart(fig_daily)  # Display daily chart below the weekly chart
st.plotly_chart(fig_weekly)  # Display weekly chart above the daily chart

# Summary table
summary_table = df.groupby('Sport')['Units_W_L'].sum().reset_index()
summary_table.rename(columns={'Units_W_L': 'Units'}, inplace=True)
summary_table['Units'] = summary_table['Units'].round(2)

summary_table = summary_table.sort_values(by='Units', ascending=False)

st.subheader("Units Summary by Sport")
st.table(summary_table)

# Calendar for daily units
calendar_data = df.groupby(df['Date'].dt.date)['Units_W_L'].sum().reset_index()
calendar_data['Date'] = pd.to_datetime(calendar_data['Date'])

df = df.sort_values(by='Date', ascending=False)

st.header('Full Data')

df['Date'] = df['Date'].dt.strftime('%m/%d/%Y')
st.dataframe(df)

# Create the plot with dynamic y-axis range
fig = px.line(df_cumulative, x='Date', y='Units', title='Cumulative Units Over Time')

# Set dynamic y-axis range
fig.update_layout(
    yaxis=dict(
        range=[y_min, y_max]
    )
)

# Display the plot
st.plotly_chart(fig)
