import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

from utils import (
    load_data,
    get_yearly_data,
    get_lunar_new_year_periods,
    get_lunar_new_year_data,
    calculate_lunar_new_year_yoy,
    calculate_monthly_bau_mom,
    calculate_monthly_special_day_uplift,
    calculate_quarterly_special_day_uplift,
    create_line_chart,
)

# Set page configuration
st.set_page_config(
    page_title="Projection Automation Tool",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🤖 Projection Automation Tool")
    st.write('---')
    
    # Set up sidebar navigation
    with st.sidebar:
        st.title("🧭 Navigation")
        selected_tab = option_menu(
            menu_title=None,
            options=["Upload Data", "Projection Analysis", "Manual Projection"],
            icons=["cloud-upload", "graph-up", "sliders"],
            menu_icon="cast",
            default_index=0,
        )
    
    # Global state for uploaded data
    if 'data' not in st.session_state:
        st.session_state.data = None
        st.session_state.target_year = None
        st.session_state.projection_data = None
    
    # Tab 1: Upload Data
    if selected_tab == "Upload Data":
        st.subheader("☁️ Upload Your Data")
        st.warning("Data must contain columns: [grass_date], [date_type], [metrics]", icon="⚠️")
        uploaded_file = st.file_uploader("Upload CSV file containing business metrics for projection", type=["csv"])
        
        if uploaded_file:
            data, error = load_data(uploaded_file)
            
            if error:
                st.error(error)
            else:
                st.success("✅ CSV Loaded")
                
                # Store data in session state
                st.session_state.data = data
                
                # Data preview
                with st.expander("👀 Preview Uploaded Data", expanded=True):
                    st.dataframe(data.sort_values('grass_date').reset_index(drop=True), use_container_width=True)
                    
                    # Show summary of available data
                    min_date = data['grass_date'].min().date()
                    max_date = data['grass_date'].max().date()
                    st.success(f"Data ranges from {min_date} ~ {max_date}")
                    
                    # Show unique date_types
                    date_types = data['date_type'].unique()
                    st.caption(f"Day types in data: {', '.join(date_types)}")
                
                st.info("⬅️ Select 'Projection Analysis' from the sidebar to continue")
    
    # Tab 2: Projection Analysis
    elif selected_tab == "Projection Analysis":
        if st.session_state.data is None:
            st.error("Please upload data first in the 'Upload Data' tab")
        else:
            data = st.session_state.data
            
            # Year selection
            st.subheader("🔎 Select Year for Analysis")
            
            # Year input for projection
            available_years = sorted(data['year'].unique())
            max_year = min(2025, max(available_years))
            target_year = st.selectbox(
                "",
                options=list(range(min(available_years), max_year + 1)),
                index=len(list(range(min(available_years), max_year + 1))) - 1  # Default to latest year
            )
            
            # Store selected year in session state
            st.session_state.target_year = target_year
            
            # Get data for projection (target year and previous year)
            projection_data = get_yearly_data(data, target_year)
            
            # Store projection data in session state
            st.session_state.projection_data = projection_data
            st.write('---')
            
            if projection_data.empty:
                st.error(f"❌ No data available for {target_year} and the previous year")
            else:
                st.subheader(f"🪄 Projection Analysis for {target_year}")
                
                # Create option menu for different projection steps
                selected_option = option_menu(
                    menu_title="",
                    options=[
                        "❶ Yearly Trend", 
                        "❷ Lunar New Year Effect", 
                        "❸ BAU MoM", 
                        "❹ Monthly Uplift", 
                        "❺ Quarterly Uplift"
                    ],
                    icons=["graph-up", "moon-fill", "calendar3", "arrows-expand", "arrows-expand"],
                    menu_icon="list-task",
                    default_index=0,
                    orientation="horizontal",
                )
                
                # Tab 1: Yearly Trends
                if selected_option == "❶ Yearly Trend":
                    st.subheader("Yearly Trend")
                    st.caption(f"Showing data from January {target_year-1} to December {target_year}")
                    
                    # Prepare data for yearly trend visualization
                    yearly_trend_data = projection_data.copy()
                    yearly_trend_data['year_month'] = yearly_trend_data['grass_date'].dt.strftime('%Y-%m')
                    yearly_trend_data['month'] = yearly_trend_data['grass_date'].dt.month
                    
                    # Group by year, month, and date_type for trend analysis
                    monthly_metrics = yearly_trend_data.groupby(['year', 'month', 'date_type'])['metrics'].mean().reset_index()
                    monthly_metrics['month_name'] = monthly_metrics['month'].apply(lambda x: datetime(2000, x, 1).strftime('%b'))
                    
                    # Get unique date types for selection
                    unique_date_types = sorted(monthly_metrics['date_type'].unique())
                    
                    # Date type selector
                    selected_date_type = st.selectbox(
                        "Select Date Type to View",
                        options=unique_date_types,
                        index=0
                    )
                    
                    # Filter data for selected date type
                    filtered_metrics = monthly_metrics[monthly_metrics['date_type'] == selected_date_type]
                    
                    # Calculate min value for y-axis (min value - 20%)
                    min_metrics = filtered_metrics['metrics'].min()
                    max_metrics = filtered_metrics['metrics'].max()
                    y_min = min_metrics - (min_metrics * 0.05)
                    y_max = max_metrics + (max_metrics * 0.05)
                    
                    # Create chart with adjusted y-axis (fix y-axis orientation)
                    year_comparison_chart = alt.Chart(filtered_metrics).mark_line(point=True).encode(
                        x=alt.X('month_name:N', title='Month', sort=list(filtered_metrics['month_name'].unique())),
                        y=alt.Y('metrics:Q', title='Metrics Value', scale=alt.Scale(domain=[y_min, y_max]), sort='ascending'),
                        color=alt.Color('year:N', title='Year'),
                        tooltip=['month_name', 'year', alt.Tooltip('metrics', format='.2f')]
                    ).properties(
                        title=f'Monthly Metrics for {selected_date_type} ({target_year-1} vs {target_year})'
                    )
                    
                    st.altair_chart(year_comparison_chart, use_container_width=True)
                    
                    # Create YoY waterfall chart
                    st.subheader(f"Year-over-Year Comparison ({target_year} vs {target_year-1})")
                    
                    # Prepare data for YoY waterfall chart
                    yoy_data = filtered_metrics.pivot(index='month_name', columns='year', values='metrics').reset_index()
                    yoy_data.columns = ['month_name', f'{target_year-1}', f'{target_year}']
                    yoy_data['yoy_diff'] = yoy_data[f'{target_year}'] - yoy_data[f'{target_year-1}']
                    
                    # Calculate YoY percentage
                    yoy_data['yoy_pct'] = (yoy_data['yoy_diff'] / yoy_data[f'{target_year-1}']) * 100
                    
                    # Sort by month order
                    month_order = {datetime(2000, i, 1).strftime('%b'): i for i in range(1, 13)}
                    yoy_data = yoy_data.sort_values('month_name', key=lambda x: x.map(lambda m: month_order.get(m, 0)))
                    
                    # Create waterfall chart with YoY percentage
                    waterfall_chart = alt.Chart(yoy_data).mark_bar().encode(
                        x=alt.X('month_name:N', title='Month', sort=list(yoy_data['month_name'])),
                        y=alt.Y('yoy_pct:Q', title='YoY Difference (%)'),
                        color=alt.condition(
                            alt.datum.yoy_pct >= 0,
                            alt.value('#4CAF50'),  # green for positive
                            alt.value('#F44336')   # red for negative
                        ),
                        tooltip=[
                            'month_name', 
                            alt.Tooltip(f'{target_year-1}', format='.2f'),
                            alt.Tooltip(f'{target_year}', format='.2f'),
                            alt.Tooltip('yoy_diff', title='YoY Difference', format='.2f'),
                            alt.Tooltip('yoy_pct', title='YoY %', format='.2f')
                        ]
                    ).properties(
                        title=f'YoY Percentage Difference for {selected_date_type} ({target_year} vs {target_year-1})'
                    )
                    
                    st.altair_chart(waterfall_chart, use_container_width=True)
                    
                    # Display data table
                    st.subheader("Monthly Average Metrics by Date Type")
                    
                    # Pivot table for display
                    pivot_table = filtered_metrics.pivot_table(
                        index=['month_name', 'month'], 
                        columns=['year'], 
                        values='metrics'
                    ).reset_index()
                    
                    # Calculate YoY percentage
                    pivot_table[f'YoY %'] = ((pivot_table[target_year] - pivot_table[target_year-1]) / pivot_table[target_year-1] * 100).round(2)
                    
                    # Sort by month
                    month_order = {datetime(2000, i, 1).strftime('%b'): i for i in range(1, 13)}
                    pivot_table = pivot_table.sort_values('month', key=lambda x: x.map(lambda m: month_order.get(m, 0)))
                    
                    # Display the table
                    st.dataframe(pivot_table.sort_values('month').reset_index(drop=True), use_container_width=True)
                
                # Tab 2: Lunar New Year Effect
                elif selected_option == "❷ Lunar New Year Effect":
                    st.subheader("Lunar New Year Effect Analysis")
                    
                    # Get Lunar New Year periods
                    current_lny_start, current_lny_end = get_lunar_new_year_periods(target_year)
                    prev_lny_start, prev_lny_end = get_lunar_new_year_periods(target_year - 1)
                    
                    st.caption(f'''
                    *Current Year Lunar New Year Period* ⇨ {current_lny_start.date()} ~ {current_lny_end.date()}  
                    
                    *Previous Year Lunar New Year Period* ⇨ {prev_lny_start.date()} ~ {prev_lny_end.date()}
                    ''')
                    
                    # Get Lunar New Year data
                    current_lny_data, prev_lny_data = get_lunar_new_year_data(projection_data, target_year)
                    
                    if current_lny_data.empty or prev_lny_data.empty:
                        st.warning("Insufficient data for Lunar New Year analysis.")
                    else:
                        # Calculate YoY differences
                        lny_yoy_data, lny_summary = calculate_lunar_new_year_yoy(current_lny_data, prev_lny_data)
                        
                        # Display Lunar New Year data
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader(f"{target_year} Data")
                            st.dataframe(
                                current_lny_data[['grass_date', 'date_type', 'metrics', 'lny_day_seq']],
                                use_container_width=True
                            )
                        
                        with col2:
                            st.subheader(f"{target_year-1} Data")
                            st.dataframe(
                                prev_lny_data[['grass_date', 'date_type', 'metrics', 'lny_day_seq']],
                                use_container_width=True
                            )
                        
                        # Display YoY comparison
                        st.subheader("Lunar New Year YoY Comparison")
                        st.dataframe(
                            lny_yoy_data[[
                                'lny_day_seq', 
                                'grass_date_current', 
                                'date_type_current', 
                                'metrics_current',
                                'grass_date_prev', 
                                'date_type_prev', 
                                'metrics_prev',
                                'yoy_diff', 
                                'yoy_pct'
                            ]],
                            use_container_width=True
                        )
                        
                        # Create chart for YoY comparison
                        lny_chart_data = lny_yoy_data.copy()
                        lny_chart_data['day_sequence'] = lny_chart_data['lny_day_seq'].astype(str)
                        
                        lny_chart = alt.Chart(lny_chart_data).mark_bar().encode(
                            x=alt.X('day_sequence:N', title='Day Sequence'),
                            y=alt.Y('yoy_pct:Q', title='YoY Change (%)'),
                            color=alt.condition(
                                alt.datum.yoy_pct > 0,
                                alt.value("green"),
                                alt.value("red")
                            ),
                            tooltip=[
                                'day_sequence', 
                                'grass_date_current', 
                                'grass_date_prev',
                                alt.Tooltip('metrics_current', title='Current Year', format='.2f'),
                                alt.Tooltip('metrics_prev', title='Previous Year', format='.2f'),
                                alt.Tooltip('yoy_pct', title='YoY Change (%)', format='.2f')
                            ]
                        ).properties(
                            title=f'Lunar New Year YoY Change ({target_year} vs {target_year-1})'
                        )
                        
                        st.altair_chart(lny_chart, use_container_width=True)
                        
                        # Display summary
                        st.subheader("Lunar New Year YoY Summary")
                        summary_data = pd.DataFrame({
                            'Metric': [
                                f'Total {target_year}', 
                                f'Total {target_year-1}', 
                                'Overall Difference', 
                                'Overall Change (%)'
                            ],
                            'Value': [
                                f"{lny_summary['total_current']:,.2f}",
                                f"{lny_summary['total_prev']:,.2f}",
                                f"{lny_summary['overall_diff']:,.2f}",
                                f"{lny_summary['overall_pct']:,.2f}%"
                            ]
                        })
                        
                        st.dataframe(summary_data, use_container_width=True)
                
                # Tab 3: BAU MoM Analysis
                elif selected_option == "❸ BAU MoM":
                    st.subheader("Business-As-Usual Month-on-Month Analysis")
                    st.caption("Calculating MoM differences for BAU days, excluding Lunar New Year periods")
                    
                    # Calculate BAU MoM differences
                    bau_mom_data = calculate_monthly_bau_mom(projection_data, target_year)
                    
                    if bau_mom_data.empty:
                        st.warning("Insufficient BAU data for MoM analysis.")
                    else:
                        # Create chart for BAU MoM trends
                        bau_mom_chart = create_line_chart(
                            bau_mom_data, 
                            'month_year', 
                            'mom_pct', 
                            title='BAU Month-on-Month Growth Rate (%)'
                        )
                        
                        st.altair_chart(bau_mom_chart, use_container_width=True)
                        
                        # Display data table
                        st.subheader("BAU Month-on-Month Analysis Data")
                        display_columns = [
                            'month_year', 
                            'metrics', 
                            'prev_month_value', 
                            'mom_diff', 
                            'mom_pct'
                        ]
                        
                        # Format the percentage column
                        formatted_bau_mom = bau_mom_data[display_columns].copy()
                        formatted_bau_mom['mom_pct'] = formatted_bau_mom['mom_pct'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
                        
                        st.dataframe(
                            formatted_bau_mom.sort_values('month_year'),
                            use_container_width=True
                        )
                
                # Tab 4: Monthly Date_Type Uplift
                elif selected_option == "❹ Monthly Uplift":
                    st.subheader("Monthly Date-Type Uplift Analysis")
                    st.caption("Calculating monthly uplift percentages for special day types compared to BAU")
                    
                    # Calculate monthly special day uplift
                    monthly_uplift_data = calculate_monthly_special_day_uplift(projection_data, target_year)
                    
                    if monthly_uplift_data.empty:
                        st.warning("Insufficient data for monthly special day type uplift analysis.")
                    else:
                        # Get all special day types
                        special_day_types = [col for col in monthly_uplift_data.columns if col not in 
                                            ['year', 'month', 'month_year', 'bau'] and not col.endswith('_uplift_pct')]
                        
                        # Create selector for special day types
                        selected_day_type = st.selectbox(
                            "Select Special Day Type",
                            options=special_day_types,
                            index=0,
                            key="monthly_day_type"
                        )
                        
                        if f'{selected_day_type}_uplift_pct' in monthly_uplift_data.columns:
                            # Prepare data for waterfall chart
                            chart_data = monthly_uplift_data[['month_year', f'{selected_day_type}_uplift_pct']].copy()
                            chart_data = chart_data.rename(columns={f'{selected_day_type}_uplift_pct': 'uplift_pct'})
                            
                            # Create waterfall chart (bar chart with color based on value)
                            waterfall_chart = alt.Chart(chart_data).mark_bar().encode(
                                x=alt.X('month_year:N', title='Month', sort=None),
                                y=alt.Y('uplift_pct:Q', title='Uplift vs BAU (%)'),
                                color=alt.condition(
                                    alt.datum.uplift_pct > 0,
                                    alt.value("green"),
                                    alt.value("red")
                                ),
                                tooltip=[
                                    'month_year',
                                    alt.Tooltip('uplift_pct', title='Uplift (%)', format='.2f')
                                ]
                            ).properties(
                                title=f'{selected_day_type} Monthly Uplift vs BAU (%)'
                            )
                            
                            st.altair_chart(waterfall_chart, use_container_width=True)
                        
                        # Display data table
                        st.subheader("Monthly Special Day Uplift Data")
                        
                        # Prepare display columns
                        display_cols = ['month_year', 'bau']
                        for day_type in special_day_types:
                            if day_type in monthly_uplift_data.columns:
                                display_cols.append(day_type)
                                if f'{day_type}_uplift_pct' in monthly_uplift_data.columns:
                                    display_cols.append(f'{day_type}_uplift_pct')
                        
                        # Format the percentage columns
                        formatted_monthly_uplift = monthly_uplift_data[display_cols].copy()
                        for col in formatted_monthly_uplift.columns:
                            if col.endswith('_uplift_pct'):
                                formatted_monthly_uplift[col] = formatted_monthly_uplift[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
                        
                        st.dataframe(
                            formatted_monthly_uplift.sort_values('month_year'),
                            use_container_width=True
                        )
                
                # Tab 5: Quarterly Special Day Uplift
                elif selected_option == "❺ Quarterly Uplift":
                    st.subheader("Quarterly Date-Type Uplift Analysis")
                    st.caption("Calculating quarterly uplift percentages for special day types compared to BAU")
                    
                    # Calculate quarterly special day uplift
                    quarterly_uplift_data = calculate_quarterly_special_day_uplift(projection_data, target_year)
                    
                    if quarterly_uplift_data.empty:
                        st.warning("Insufficient data for quarterly special day type uplift analysis.")
                    else:
                        # Get all special day types
                        special_day_types = [col for col in quarterly_uplift_data.columns if col not in 
                                            ['year', 'quarter', 'quarter_year', 'bau'] and not col.endswith('_uplift_pct')]
                        
                        # Create selector for special day types
                        selected_day_type = st.selectbox(
                            "Select Special Day Type",
                            options=special_day_types,
                            index=0,
                            key="quarterly_day_type"
                        )
                        
                        if f'{selected_day_type}_uplift_pct' in quarterly_uplift_data.columns:
                            # Prepare data for waterfall chart
                            chart_data = quarterly_uplift_data[['quarter_year', f'{selected_day_type}_uplift_pct']].copy()
                            chart_data = chart_data.rename(columns={f'{selected_day_type}_uplift_pct': 'uplift_pct'})
                            
                            # Create waterfall chart (bar chart with color based on value)
                            waterfall_chart = alt.Chart(chart_data).mark_bar().encode(
                                x=alt.X('quarter_year:N', title='Quarter', sort=None),
                                y=alt.Y('uplift_pct:Q', title='Uplift vs BAU (%)'),
                                color=alt.condition(
                                    alt.datum.uplift_pct > 0,
                                    alt.value("green"),
                                    alt.value("red")
                                ),
                                tooltip=[
                                    'quarter_year',
                                    alt.Tooltip('uplift_pct', title='Uplift (%)', format='.2f')
                                ]
                            ).properties(
                                title=f'{selected_day_type} Quarterly Uplift vs BAU (%)'
                            )
                            
                            st.altair_chart(waterfall_chart, use_container_width=True)
                        
                        # Display data table
                        st.subheader("Quarterly Special Day Uplift Data")
                        
                        # Prepare display columns
                        display_cols = ['quarter_year', 'bau']
                        for day_type in special_day_types:
                            if day_type in quarterly_uplift_data.columns:
                                display_cols.append(day_type)
                                if f'{day_type}_uplift_pct' in quarterly_uplift_data.columns:
                                    display_cols.append(f'{day_type}_uplift_pct')
                        
                        # Format the percentage columns
                        formatted_quarterly_uplift = quarterly_uplift_data[display_cols].copy()
                        for col in formatted_quarterly_uplift.columns:
                            if col.endswith('_uplift_pct'):
                                formatted_quarterly_uplift[col] = formatted_quarterly_uplift[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
                        
                        st.dataframe(
                            formatted_quarterly_uplift.sort_values('quarter_year'),
                            use_container_width=True
                        )
    
    # Tab 3: Manual Projection
    elif selected_tab == "Manual Projection":
        st.header("🛠️ Manual Projection")
        st.caption("Adjust parameters to create your own projection")
        
        # Baseline input
        baseline = st.number_input("Enter Baseline Value", min_value=0, value=5_000_000, step=10_000)
        
        st.write("---")
        st.subheader("Adjustment Factors")
        
        # Create sliders for different effects
        lny_effect = st.select_slider(
            "Lunar New Year Effect",
            options=[float(f"{-30.0 + i * 0.1:.1f}") for i in range(601)],  # -30% to +30% in 0.1% increments
            value=0.0,
            format_func=lambda x: f"{x:+.1f}%"
        )
        
        bau_mom_effect = st.select_slider(
            "BAU MoM Growth",
            options=[float(f"{-30.0 + i * 0.1:.1f}") for i in range(601)],  # -30% to +30% in 0.1% increments
            value=0.0,
            format_func=lambda x: f"{x:+.1f}%"
        )
        
        monthly_uplift = st.select_slider(
            "Monthly Uplift - Date Type",
            options=[float(f"{-30.0 + i * 0.1:.1f}") for i in range(601)],  # -30% to +30% in 0.1% increments
            value=0.0,
            format_func=lambda x: f"{x:+.1f}%"
        )
        
        quarterly_uplift = st.select_slider(
            "Quarterly Uplift - Date Type",
            options=[float(f"{-30.0 + i * 0.1:.1f}") for i in range(601)],  # -30% to +30% in 0.1% increments
            value=0.0,
            format_func=lambda x: f"{x:+.1f}%"
        )
        
        # Calculate total effect
        total_effect_pct = lny_effect + bau_mom_effect + monthly_uplift + quarterly_uplift
        projected_value = baseline * (1 + total_effect_pct / 100)
        
        # Display results
        st.write("---")
        st.subheader("Projection Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Baseline Value", f"{baseline:,.2f}")
            st.metric("Total Effect", f"{total_effect_pct:+.2f}%")
        
        with col2:
            st.metric("Projected Value", f"{projected_value:,.2f}")
            st.metric("Absolute Change", f"{projected_value - baseline:+,.2f}")
        
        # Display breakdown
        st.write("---")
        st.subheader("Effect Breakdown")
        
        effect_data = pd.DataFrame({
            'Factor': ['Lunar New Year Effect', 'BAU MoM Growth', 'Monthly Uplift', 'Quarterly Uplift', 'Total Effect'],
            'Percentage': [lny_effect, bau_mom_effect, monthly_uplift, quarterly_uplift, total_effect_pct],
            'Value Impact': [
                baseline * (lny_effect / 100),
                baseline * (bau_mom_effect / 100),
                baseline * (monthly_uplift / 100),
                baseline * (quarterly_uplift / 100),
                baseline * (total_effect_pct / 100)
            ]
        })
        
        # Format the percentage and value columns
        effect_data['Percentage'] = effect_data['Percentage'].apply(lambda x: f"{x:+.2f}%")
        effect_data['Value Impact'] = effect_data['Value Impact'].apply(lambda x: f"{x:+,.2f}")
        
        st.dataframe(effect_data, use_container_width=True)
    
    else:
        # Display welcome message when no tab is selected (should not happen)
        st.title("🤖 Business Metrics Analysis Tool")
        st.markdown("""
        ### Welcome to the Business Metrics Analysis Tool!
        
        Please select an option from the sidebar to get started.
        """)

if __name__ == "__main__":
    main()