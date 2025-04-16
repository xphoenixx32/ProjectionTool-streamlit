import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import base64
import io
import streamlit as st
import altair as alt

# Cache data loading for performance
@st.cache_data
def load_data(uploaded_file):
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
    """
    start_date = datetime(target_year - 1, 1, 1)
    end_date = datetime(target_year, 12, 31)
    
    filtered_data = data[(data['grass_date'] >= start_date) & 
                         (data['grass_date'] <= end_date)].copy()
    
    return filtered_data

def get_lunar_new_year_periods(year):
    """
    Return the Lunar New Year periods (Little New Year's Eve to 5th day) for a given year
    These are approximate dates and should be adjusted for actual lunar calendar
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

def get_lunar_new_year_data(data, year):
    """
    Extract Lunar New Year period data for analysis
    """
    # Get Lunar New Year periods for current and previous year
    current_lny_start, current_lny_end = get_lunar_new_year_periods(year)
    prev_lny_start, prev_lny_end = get_lunar_new_year_periods(year - 1)
    
    # Filter data for current year Lunar New Year
    current_lny_data = data[(data['grass_date'] >= current_lny_start) & 
                            (data['grass_date'] <= current_lny_end)].copy()
    
    # Filter data for previous year Lunar New Year
    prev_lny_data = data[(data['grass_date'] >= prev_lny_start) & 
                         (data['grass_date'] <= prev_lny_end)].copy()
    
    # Add day sequence for comparison (1st day, 2nd day, etc.)
    if not current_lny_data.empty:
        current_lny_data['lny_day_seq'] = range(1, len(current_lny_data) + 1)
    
    if not prev_lny_data.empty:
        prev_lny_data['lny_day_seq'] = range(1, len(prev_lny_data) + 1)
    
    return current_lny_data, prev_lny_data

def calculate_lunar_new_year_yoy(current_lny_data, prev_lny_data):
    """
    Calculate Year-over-Year differences for Lunar New Year period
    """
    if current_lny_data.empty or prev_lny_data.empty:
        return None, None
    
    # Merge on day sequence for direct comparison
    merged_data = pd.merge(
        current_lny_data[['lny_day_seq', 'grass_date', 'date_type', 'metrics']],
        prev_lny_data[['lny_day_seq', 'grass_date', 'date_type', 'metrics']],
        on='lny_day_seq',
        suffixes=('_current', '_prev')
    )
    
    # Calculate YoY difference and percentage
    merged_data['yoy_diff'] = merged_data['metrics_current'] - merged_data['metrics_prev']
    merged_data['yoy_pct'] = (merged_data['metrics_current'] / merged_data['metrics_prev'] - 1) * 100
    
    # Calculate overall YoY difference
    total_current = current_lny_data['metrics'].sum()
    total_prev = prev_lny_data['metrics'].sum()
    overall_yoy_diff = total_current - total_prev
    overall_yoy_pct = (total_current / total_prev - 1) * 100
    
    return merged_data, {
        'total_current': total_current,
        'total_prev': total_prev,
        'overall_diff': overall_yoy_diff,
        'overall_pct': overall_yoy_pct
    }

def calculate_monthly_bau_mom(data, year):
    """
    Calculate Month-on-Month differences for BAU days, excluding Lunar New Year period
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

def calculate_monthly_special_day_uplift(data, year):
    """
    Calculate monthly uplift percentages for special day types compared to BAU
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

def calculate_quarterly_special_day_uplift(data, year):
    """
    Calculate quarterly uplift percentages for special day types compared to BAU
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
    """
    if color_col:
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

def create_bar_chart(data, x_col, y_col, color_col=None, title=None):
    """
    Create a bar chart using Altair
    """
    if color_col:
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X(f'{x_col}:N', title=x_col.replace('_', ' ').title()),
            y=alt.Y(f'{y_col}:Q', title=y_col.replace('_', ' ').title()),
            color=alt.Color(f'{color_col}:N', title=color_col.replace('_', ' ').title()),
            tooltip=[x_col, alt.Tooltip(y_col, format='.2f'), color_col]
        )
    else:
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X(f'{x_col}:N', title=x_col.replace('_', ' ').title()),
            y=alt.Y(f'{y_col}:Q', title=y_col.replace('_', ' ').title()),
            tooltip=[x_col, alt.Tooltip(y_col, format='.2f')]
        )
    
    if title:
        chart = chart.properties(title=title)
    
    return chart

def calculate_mom_growth(data, metric_name='metrics'):
    """Calculate Month-on-Month growth rates for business-as-usual days"""
    # Filter for BAU days only
    bau_data = data[data['date_type'] == 'bau'].copy()
    
    # Group by year and month to get monthly aggregates
    monthly_agg = bau_data.groupby(['year', 'month'])[metric_name].sum().reset_index()
    
    # Calculate MoM growth rates
    monthly_agg['prev_month_value'] = monthly_agg[metric_name].shift(1)
    monthly_agg['mom_growth'] = monthly_agg[metric_name] / monthly_agg['prev_month_value'] - 1
    
    return monthly_agg

def calculate_uplift_effects(data, metric_name='metrics'):
    """Calculate uplift effects for special days compared to BAU baseline"""
    result = {}
    
    # Get all unique date_types
    date_types = data['date_type'].unique()
    
    # Group data by year, month, and date_type
    grouped = data.groupby(['year', 'month', 'date_type'])[metric_name].mean().reset_index()
    
    # Pivot to get a wide format for easier comparison
    pivoted = grouped.pivot_table(index=['year', 'month'], columns='date_type', values=metric_name).reset_index()
    
    # Calculate uplift for each special date_type compared to BAU
    for date_type in date_types:
        if date_type != 'bau':
            pivoted[f'{date_type}_uplift'] = pivoted[date_type] / pivoted['bau'] - 1
            # Average uplift across all months
            result[date_type] = pivoted[f'{date_type}_uplift'].mean()
    
    return result, pivoted

def project_future_values(data, target_year, target_month, metric_name='metrics'):
    """Project future values based on MoM growth trends and event adjustments"""
    # Get historical data
    historical_end_date = datetime(target_year, target_month, 1) - timedelta(days=1)
    historical_start_date = datetime(target_year-1, 1, 1)
    
    historical_data = data[(data['grass_date'] >= historical_start_date) & 
                            (data['grass_date'] <= historical_end_date)].copy()
    
    if len(historical_data) == 0:
        return None, "No historical data found for period {historical_start_date.date()} to {historical_end_date.date()}"
    
    # Calculate MoM growth rates for BAU days
    mom_growth_data = calculate_mom_growth(historical_data, metric_name)
    
    # Get average MoM growth rate from the last 3 months
    recent_growth = mom_growth_data.tail(3)['mom_growth'].mean()
    if pd.isna(recent_growth):
        recent_growth = mom_growth_data['mom_growth'].mean()  # Use all available if recent is not available
    
    # Calculate uplift effects for special days
    uplift_effects, uplift_data = calculate_uplift_effects(historical_data, metric_name)
    
    # Get the most recent month's data
    last_month_data = historical_data[
        (historical_data['year'] == historical_end_date.year) & 
        (historical_data['month'] == historical_end_date.month)
    ]
    
    # Get the BAU baseline for the last month
    last_month_bau = last_month_data[last_month_data['date_type'] == 'bau'][metric_name].mean()
    
    # Project next month
    next_month_bau = last_month_bau * (1 + recent_growth)
    
    # Create projection dataframe
    days_in_month = (datetime(target_year, target_month + 1, 1) if target_month < 12 else datetime(target_year + 1, 1, 1)) - datetime(target_year, target_month, 1)
    days_in_month = days_in_month.days
    
    # Create dates for the target month
    projection_dates = [datetime(target_year, target_month, day) for day in range(1, days_in_month + 1)]
    
    # Get the day types for this month from previous year if available
    prev_year_same_month = data[
        (data['year'] == target_year - 1) & 
        (data['month'] == target_month)
    ].copy()
    
    if len(prev_year_same_month) == 0:
        # Use default distribution if no previous year data
        date_types = ['bau'] * days_in_month
        # Add some special days for demonstration
        if days_in_month > 15:
            date_types[7] = '1st_spike'
            date_types[14] = '2nd_spike'
            date_types[21] = '3rd_spike'
            date_types[28] = 'FSS' if days_in_month > 28 else 'bau'
    else:
        # Map the previous year's date types to this year's dates
        # Handling cases where month lengths differ between years
        prev_year_days = prev_year_same_month.sort_values('grass_date')
        date_types = []
        for i in range(days_in_month):
            if i < len(prev_year_days):
                date_types.append(prev_year_days.iloc[i]['date_type'])
            else:
                date_types.append('bau')  # Default to BAU for extra days
    
    # Create projection dataframe
    projection = pd.DataFrame({
        'grass_date': projection_dates,
        'date_type': date_types,
        'year': target_year,
        'month': target_month
    })
    
    # Apply projections based on date type
    projection[metric_name] = projection['date_type'].apply(
        lambda dt: next_month_bau * (1 + uplift_effects.get(dt, 0)) if dt != 'bau' else next_month_bau
    )
    
    return (projection, mom_growth_data, uplift_effects, uplift_data), None

def get_download_link(df, filename, text):
    """Generate a download link for a dataframe"""
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Convert any problematic columns to string to avoid formatting issues
    for col in df_copy.columns:
        if df_copy[col].dtype == 'datetime64[ns]':
            df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d')
    
    # Ensure all column names are strings
    df_copy.columns = [str(col) for col in df_copy.columns]
    
    # Convert to CSV
    csv = df_copy.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href
