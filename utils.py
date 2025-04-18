import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import altair as alt

# Cache data loading for performance
@st.cache_data
def load_data(uploaded_file):
    """
    Load and process the uploaded data

    Parameters:
        uploaded_file: The uploaded CSV file containing business metrics
    
    Returns:
        data: Processed DataFrame with additional columns for year, month, quarter, and day
        error: Error message if data loading fails
    """
    try:
        data = pd.read_csv(uploaded_file)
        
        # Ensure column names are standardized
        required_columns = ['grass_date', 'date_type', 'metrics']
        data.columns = [col.lower().strip() for col in data.columns]
        
        # Check if all required columns exist
        for col in required_columns:
            if col not in data.columns:
                missing_cols = [c for c in required_columns if c not in data.columns]
                return None, f"Missing required columns: {', '.join(missing_cols)}. Please check your CSV file."
        
        # Convert date column to datetime
        data['grass_date'] = pd.to_datetime(data['grass_date'])
        
        # Extract year and month for easier filtering
        data['year'] = data['grass_date'].dt.year
        data['month'] = data['grass_date'].dt.month
        data['quarter'] = data['grass_date'].dt.quarter
        data['day'] = data['grass_date'].dt.day
        
        # Sort by date
        data = data.sort_values('grass_date')
        
        return data, None
    except Exception as e:
        return None, f"Error loading data: {e}"

def get_yearly_data(data, target_year):
    """
    Get data for the target year and the previous year starting from January

    Parameters:
        data: DataFrame containing the uploaded data
        target_year: Year for which data is needed
    
    Returns:
        filtered_data: DataFrame containing data for the target year and the previous year
    """
    start_date = datetime(target_year - 1, 1, 1)
    end_date = datetime(target_year, 12, 31)
    
    filtered_data = data[(data['grass_date'] >= start_date) & 
                         (data['grass_date'] <= end_date)].copy()
    
    return filtered_data

def get_lunar_new_year_periods(year):
    """
    Return the Lunar New Year periods (Little New Year's Eve to New Year's 5th) for a given year
    These are approximate dates and should be adjusted for actual lunar calendar

    Parameters:
        year: Year for which the Lunar New Year periods are needed
    
    Returns:
        lny_start, lny_end: Start and end dates of the Lunar New Year period
    """
    # Approximate Lunar New Year dates (would need to be updated with actual lunar calendar)
    lunar_new_year_dates = {
        2021: (datetime(2021, 2, 10), datetime(2021, 2, 16)),   # Feb 10 - Feb 16, 2021
        2022: (datetime(2022, 1, 30), datetime(2022, 2, 5)),  # Jan 30 - Feb 5, 2022
        2023: (datetime(2023, 1, 20), datetime(2023, 1, 26)),  # Jan 20 - Jan 26, 2023
        2024: (datetime(2024, 2, 8), datetime(2024, 2, 14)),   # Feb 8 - Feb 14, 2024
        2025: (datetime(2025, 1, 27), datetime(2025, 2, 2)),   # Jan 27 - Feb 2, 2025
        2026: (datetime(2026, 2, 15), datetime(2026, 2, 21)),  # Feb 15 - Feb 21, 2026
    }
    
    if year in lunar_new_year_dates:
        return lunar_new_year_dates[year]
    else:
        # Fallback to approximate dates if not in our dictionary
        return (datetime(year, 2, 1), datetime(year, 2, 7))

def calculate_monthly_bau_mom(data, year):
    """
    Calculate Month-on-Month differences for BAU days, excluding Lunar New Year period
    
    Parameters:
        data: DataFrame containing the uploaded data
        year: Year for which the calculation is needed
    
    Returns:
        monthly_avg: DataFrame containing monthly averages and MoM differences
    """
    # Get Lunar New Year periods to exclude
    current_lny_start, current_lny_end = get_lunar_new_year_periods(year)
    prev_lny_start, prev_lny_end = get_lunar_new_year_periods(year - 1)
    
    # Filter for BAU days only, excluding Lunar New Year periods
    bau_data = data[
        (data['date_type'] == 'bau') & 
        ~(
            ((data['grass_date'] >= current_lny_start) & (data['grass_date'] <= current_lny_end)) | 
            ((data['grass_date'] >= prev_lny_start) & (data['grass_date'] <= prev_lny_end))
        )
    ].copy()
    
    # Group by year and month to get monthly averages
    monthly_avg = bau_data.groupby(['year', 'month'])['metrics'].mean().reset_index()
    
    # Calculate MoM differences
    monthly_avg['prev_month_value'] = monthly_avg['metrics'].shift(1)
    monthly_avg['mom_diff'] = monthly_avg['metrics'] - monthly_avg['prev_month_value']
    monthly_avg['mom_pct'] = (monthly_avg['metrics'] / monthly_avg['prev_month_value'] - 1) * 100
    
    # Add month-year column for display
    monthly_avg['month_year'] = monthly_avg.apply(
        lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1
    )
    
    return monthly_avg

def calculate_monthly_uplift_datetype_vs_bau(data, year):
    """
    Calculate MONTHLY uplift percentages for date_type compared to BAU
    
    Parameters:
        data: DataFrame containing the uploaded data
        year: Year for which the calculation is needed
    
    Returns:
        monthly_avg: DataFrame containing monthly averages and uplift percentages
    """
    # Get Lunar New Year periods to exclude
    current_lny_start, current_lny_end = get_lunar_new_year_periods(year)
    prev_lny_start, prev_lny_end = get_lunar_new_year_periods(year - 1)
    
    # Filter data excluding Lunar New Year periods
    filtered_data = data[
        ~(
            ((data['grass_date'] >= current_lny_start) & (data['grass_date'] <= current_lny_end)) | 
            ((data['grass_date'] >= prev_lny_start) & (data['grass_date'] <= prev_lny_end))
        )
    ].copy()
    
    # Group by year, month, and date_type to get monthly averages
    monthly_avg_by_type = filtered_data.groupby(['year', 'month', 'date_type'])['metrics'].mean().reset_index()
    
    # Pivot to get a wide format with date_type as columns
    pivoted = monthly_avg_by_type.pivot_table(
        index=['year', 'month'], 
        columns='date_type', 
        values='metrics'
    ).reset_index()
    
    # Calculate uplift for each special date_type compared to BAU
    date_types = filtered_data['date_type'].unique()
    for date_type in date_types:
        if date_type != 'bau' and date_type in pivoted.columns:
            pivoted[f'{date_type}_uplift_pct'] = (pivoted[date_type] / pivoted['bau'] - 1) * 100
    
    # Add month-year column for display
    pivoted['month_year'] = pivoted.apply(
        lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1
    )
    
    return pivoted

def calculate_quarterly_uplift_datetype_vs_bau(data, year):
    """
    Calculate QUARTERLY uplift percentages for date_type compared to BAU
    
    Parameters:
        data: DataFrame containing the uploaded data
        year: Year for which the calculation is needed
    
    Returns:
        quarterly_avg: DataFrame containing quarterly averages and uplift percentages
    """
    # Get Lunar New Year periods to exclude
    current_lny_start, current_lny_end = get_lunar_new_year_periods(year)
    prev_lny_start, prev_lny_end = get_lunar_new_year_periods(year - 1)
    
    # Filter data excluding Lunar New Year periods
    filtered_data = data[
        ~(
            ((data['grass_date'] >= current_lny_start) & (data['grass_date'] <= current_lny_end)) | 
            ((data['grass_date'] >= prev_lny_start) & (data['grass_date'] <= prev_lny_end))
        )
    ].copy()
    
    # Group by year, quarter, and date_type to get quarterly averages
    quarterly_avg_by_type = filtered_data.groupby(['year', 'quarter', 'date_type'])['metrics'].mean().reset_index()
    
    # Pivot to get a wide format with date_type as columns
    pivoted = quarterly_avg_by_type.pivot_table(
        index=['year', 'quarter'], 
        columns='date_type', 
        values='metrics'
    ).reset_index()
    
    # Calculate uplift for each special date_type compared to BAU
    date_types = filtered_data['date_type'].unique()
    for date_type in date_types:
        if date_type != 'bau' and date_type in pivoted.columns:
            pivoted[f'{date_type}_uplift_pct'] = (pivoted[date_type] / pivoted['bau'] - 1) * 100
    
    # Add quarter-year column for display
    pivoted['quarter_year'] = pivoted.apply(
        lambda x: f"Q{int(x['quarter'])}/{int(x['year'])}", axis=1
    )
    
    return pivoted

def create_line_chart(data, x_col, y_col, color_col=None, title=None):
    """
    Create a line chart using Altair
    
    Parameters:
        data: DataFrame containing the data to be plotted
        x_col: Column name for the x-axis
        y_col: Column name for the y-axis
        color_col: Column name for the color encoding
        title: Title of the chart
    
    Returns:
        chart: Altair chart object
    """
    # Create basic line chart
    if color_col:
        # Add tooltip for better interactivity
        chart = alt.Chart(data).mark_line(point=True).encode(
            x=alt.X(f'{x_col}:N', title=x_col.replace('_', ' ').title()),
            y=alt.Y(f'{y_col}:Q', title=y_col.replace('_', ' ').title()),
            color=alt.Color(f'{color_col}:N', title=color_col.replace('_', ' ').title()),
            tooltip=[x_col, alt.Tooltip(y_col, format='.2f'), color_col]
        )
    else:
        
        chart = alt.Chart(data).mark_line(point=True).encode(
            x=alt.X(f'{x_col}:N', title=x_col.replace('_', ' ').title()),
            y=alt.Y(f'{y_col}:Q', title=y_col.replace('_', ' ').title()),
            tooltip=[x_col, alt.Tooltip(y_col, format='.2f')]
        )
    
    if title:
        chart = chart.properties(title=title)
    
    return chart

def calculate_lny_vs_bau(data, lny_start, lny_end):
    """
    For a given Lunar New Year period, calculate each day's metrics vs the BAU average of the first month of the period.
    
    Parameters:
        data: DataFrame containing the uploaded data
        lny_start: Start date of the Lunar New Year period
        lny_end: End date of the Lunar New Year period
    
    Returns:
        lny_data: DataFrame containing the LNY period data and BAU comparison
    """
    # Filter LNY period data
    lny_mask = (data['grass_date'] >= lny_start) & (data['grass_date'] <= lny_end)
    lny_data = data[lny_mask].copy()
    if lny_data.empty:
        return pd.DataFrame()

    # Determine the first month of LNY period
    first_month = lny_start.month
    first_year = lny_start.year

    # BAU: same month, same year, date_type == 'bau'
    bau_mask = (
        (data['year'] == first_year) &
        (data['month'] == first_month) &
        (data['date_type'].str.lower() == 'bau')
    )
    bau_data = data[bau_mask]
    bau_avg = bau_data['metrics'].mean() if not bau_data.empty else np.nan

    # Add BAU average and calculate differences
    lny_data['bau_avg'] = bau_avg
    lny_data['diff'] = lny_data['metrics'] - bau_avg
    lny_data['pct_change'] = (lny_data['metrics'] - bau_avg) / bau_avg if bau_avg != 0 else np.nan
    return lny_data[['grass_date', 'metrics', 'bau_avg', 'diff', 'pct_change']]
