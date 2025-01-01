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

# Load the data from the Google sheet
url = "https://docs.google.com/spreadsheets/d/1XoVHcy6qqwKKT7HiIb5CKwv32_1Ce1fhl5XoPW-lREI/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)
df = conn.read(spreadsheet=url, usecols=[0, 1, 2, 3, 4, 5, 6, 7,8])
df['Date'] = pd.to_datetime(df['Date'])

# Sidebar to filter by sport and date range
sports = df['Sport'].unique()
selected_sport = st.sidebar.selectbox('Select Sport', options=['All'] + list(sports))

if selected_sport != 'All':
    df = df[df['Sport'] == selected_sport]

date_range = st.sidebar.date_input("Select Date Range", [df['Date'].min(), df['Date'].max()])

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
else:
    st.sidebar.warning("Please select both start and end dates.")

# Filter for 2024 data
df_2024 = df[df['Date'].dt.year == 2024]

# Create dataframe for the Pick of the Day (POTD)
df_potd_2024 = df_2024[df_2024['POTD'] == 1]

# Summary stats for POTD in 2024
w_count_potd = (df_potd_2024['Win_Loss_Push'] == 'w').sum()
l_count_potd = (df_potd_2024['Win_Loss_Push'] == 'l').sum()
p_count_potd = (df_potd_2024['Win_Loss_Push'] == 'p').sum()
total_records_potd = w_count_potd + l_count_potd + p_count_potd
win_percentage_potd = (w_count_potd / total_records_potd) * 100 if total_records_potd > 0 else 0
total_units_potd = df_potd_2024['Units_W_L'].sum()

# Summary stats for overall record in 2024
w_count = (df_2024['Win_Loss_Push'] == 'w').sum()
l_count = (df_2024['Win_Loss_Push'] == 'l').sum()
p_count = (df_2024['Win_Loss_Push'] == 'p').sum()
total_records = w_count + l_count + p_count
win_percentage = (w_count / total_records) * 100 if total_records > 0 else 0
total_units = df_2024['Units_W_L'].sum()

# Calculate the average odds for the overall dataset
avg_odds_overall = df['Odds'].mean() if not df['Odds'].isnull().all() else 0

# Calculate the average odds for the POTD dataset
avg_odds_potd = df_potd_2024['Odds'].mean() if not df_potd_2024['Odds'].isnull().all() else 0

# Display 2024 Summary Statistics for Overall
st.header("Summary Statistics (September 2024- Dec 2024)")

col1, col2, col3, col4, col5, col6 = st.columns(6)

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

with col6:
    st.metric("Average Odds (Overall)", f"{avg_odds_overall:.2f}")


st.header("POTD Summary Statistics (2024)")

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


st.header("Picks Made")
# Radio button to filter full data by POTD
data_filter = st.radio("Filter Data by POTD", ("All Data", "POTD Only"))

if data_filter == "POTD Only":
    df_filtered = df[df['POTD'] == 1]
else:
    df_filtered = df

# Filter for 2025 data for visuals
df_2025 = df[df['Date'].dt.year == 2025]


df_filtered['Date'] = df_filtered['Date'].dt.strftime('%m/%d/%Y')
df_filtered = df_filtered.sort_values(by='Date', ascending=False)
st.dataframe(df_filtered)

# # Cumulative units for 2025 data
# df_cumulative_2025 = df_2025.groupby('Date').agg({'Units_W_L': 'sum'}).cumsum().reset_index()
# df_cumulative_2025.rename(columns={'Units_W_L': 'Units'}, inplace=True)
# y_min_2025 = df_cumulative_2025['Units'].min() - 10
# y_max_2025 = df_cumulative_2025['Units'].max() + 10

# # Sum the Units win/loss for each day in 2025
# df_daily_sum_2025 = df_2025.groupby('Date')['Units_W_L'].sum().reset_index()

# # Daily units chart for 2025
# fig_daily_2025 = go.Figure()

# fig_daily_2025.add_trace(go.Bar(
#     x=df_daily_sum_2025['Date'],
#     y=df_daily_sum_2025['Units_W_L'],
#     marker=dict(color=df_daily_sum_2025['Units_W_L'].apply(lambda x: 'green' if x > 0 else 'red')),
#     text=df_daily_sum_2025['Units_W_L'].round(2),
#     textposition='auto',
#     hoverinfo='x+y+text',
# ))

# fig_daily_2025.update_layout(
#     title='Daily Units Won / Lost (2025)',
#     xaxis_title='Date',
#     yaxis_title='Units Won / Lost',
#     showlegend=False,
#     template='plotly_white',
#     xaxis_tickangle=-45,
# )

# # Weekly units chart for 2025
# df_2025['Week'] = df_2025['Date'].dt.to_period('W').dt.start_time
# df_weekly_sum_2025 = df_2025.groupby('Week')['Units_W_L'].sum().reset_index()

# fig_weekly_2025 = go.Figure()

# fig_weekly_2025.add_trace(go.Bar(
#     x=df_weekly_sum_2025['Week'],
#     y=df_weekly_sum_2025['Units_W_L'],
#     marker=dict(color=df_weekly_sum_2025['Units_W_L'].apply(lambda x: 'green' if x > 0 else 'red')),
#     text=df_weekly_sum_2025['Units_W_L'].round(2),
#     textposition='auto',
#     hoverinfo='x+y+text',
# ))

# fig_weekly_2025.update_layout(
#     title='Weekly Units Won / Lost (2025)',
#     xaxis_title='Week',
#     yaxis_title='Units Won / Lost',
#     showlegend=False,
#     template='plotly_white',
#     xaxis_tickangle=-45,
# )

# # Display the daily and weekly charts for 2025
# st.plotly_chart(fig_daily_2025)  # Display daily chart
# st.plotly_chart(fig_weekly_2025)  # Display weekly chart

# # Filter for 2025 data
# df_2025 = df[df['Date'].dt.year == 2025]

# # Units Summary by Sport for 2025
# summary_table_2025 = df_2025.groupby('Sport')['Units_W_L'].sum().reset_index()
# summary_table_2025.rename(columns={'Units_W_L': 'Units'}, inplace=True)
# summary_table_2025['Units'] = summary_table_2025['Units'].round(2)
# summary_table_2025 = summary_table_2025.sort_values(by='Units', ascending=False)

# st.subheader("Units Summary by Sport (2025)")
# st.table(summary_table_2025)

# fig_2025 = px.line(df_cumulative_2025, x='Date', y='Units', title='Cumulative Units Over Time (2025)')
# fig_2025.update_layout(
#     yaxis=dict(
#         range=[y_min_2025, y_max_2025]
#     )
# )
# st.plotly_chart(fig_2025)
