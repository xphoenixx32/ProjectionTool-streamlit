import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from streamlit_option_menu import option_menu
from streamlit.errors import StreamlitAPIException

from utils import (
    load_data,
    get_yearly_data,
    get_lunar_new_year_periods,
    calculate_monthly_bau_mom,
    calculate_monthly_uplift_datetype_vs_bau,
    calculate_quarterly_uplift_datetype_vs_bau,
    create_line_chart,
    calculate_lny_vs_bau
)

# Set page configuration
st.set_page_config(
    page_title="Projection Automation Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🤖 Projection Automation Tool")
    st.write('---')
    
    # Set up sidebar navigation
    with st.sidebar:
        st.title("🧭 Navigation")
        st.caption('''
        *Select tab to Proceed*
        ''')
        selected_tab = option_menu(
            menu_title=None,
            options=["Upload Data", "Projection Analysis", "Manual Projection"],
            icons=["upload", "bar-chart", "sliders"],
            menu_icon="cast",
            default_index=0,
        )
        st.write('---')
        st.title("✏️ Notes")
        st.caption('''
        *Enter the needed values for each factor*
        ''')
        with st.expander("Expand Notes", expanded=False):
            base_df = pd.DataFrame(
                [
                    {"dims": "Baseline Value", "value": 5_000_000},
                ],
                columns=["dims", "value"],
            )
            metrics_df = pd.DataFrame(
                [
                    {"factors": "Lunar New Year Effect", "value": 0.000},
                    {"factors": "BAU MoM", "value": 0.000},
                    {"factors": "Monthly Uplift", "value": 0.000},
                    {"factors": "Quarterly Uplift", "value": 0.000},
                    {"factors": "Additional Factors", "value": 0.000},
                ],
                columns=["factors", "value"],
            )
            st.data_editor(base_df, use_container_width=True, key="base_df", hide_index=True)
            st.data_editor(metrics_df, use_container_width=True, key="note_df", hide_index=True)
    
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
                st.success("✅ CSV Loaded Successfully")
                
                # Store data in session state
                st.session_state.data = data
                
                # Data preview
                with st.expander("👀 Preview Uploaded Data", expanded=False):
                    st.dataframe(data.sort_values('grass_date').reset_index(drop=True), use_container_width=True, hide_index=True)
                    
                    # Show summary of available data
                    min_date = data['grass_date'].min().date()
                    max_date = data['grass_date'].max().date()
                    st.info(f"Data ranges from {min_date} ~ {max_date}")
                    
                    # Show unique date_types
                    date_types = data['date_type'].unique()
                    st.caption(f"Day types in data: {', '.join(date_types)}")
                
                st.error("⬅️ Select 'Projection Analysis' from the sidebar to continue")
    
    # Tab 2: Projection Analysis
    elif selected_tab == "Projection Analysis":
        if st.session_state.data is None:
            st.error("Please upload data first in the 'Upload Data' tab")
        else:
            data = st.session_state.data
            
            # Year selection
            st.subheader("🔎 Select Year for Metrics Analysis")
            
            # Year input for projection
            available_years = sorted(data['year'].unique())
            max_year = min(2025, max(available_years))
            target_year = st.selectbox(
                "---",
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
                st.header(f"📈 Metrics Analysis of {target_year}")
                
                # Create option menu for different projection steps
                selected_option = option_menu(
                    menu_title=" ",
                    options=[
                        "Yearly Trend", 
                        "Lunar New Year Effect", 
                        "BAU MoM", 
                        "Monthly Uplift", 
                        "Quarterly Uplift"
                    ],
                    icons=["graph-up", "moon-fill", "calendar3", "arrows-expand", "arrows-expand"],
                    menu_icon="list-task",
                    default_index=0,
                    orientation="horizontal",
                )
                
                # Tab 1: Yearly Trends
                if selected_option == "Yearly Trend":
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
                    st.write('---')
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
                    
                    # Add text labels to the line chart
                    text_labels = alt.Chart(filtered_metrics).mark_text(
                        align='center',
                        baseline='bottom',
                        dy=0,
                        fontSize=10
                    ).encode(
                        x=alt.X('month_name:N', sort=list(filtered_metrics['month_name'].unique())),
                        y='metrics:Q',
                        text=alt.Text('metrics:Q', format='.0f'),
                        color=alt.Color('year:N')
                    )
                    
                    # Combine the line chart with labels
                    final_year_chart = year_comparison_chart + text_labels
                    
                    st.subheader("Comparison View for Yearly Trend")
                    st.altair_chart(final_year_chart, use_container_width=True)
                    
                    # Create YoY waterfall chart
                    st.write('---')
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
                    
                    # Add text labels to the waterfall chart
                    waterfall_labels = alt.Chart(yoy_data).mark_text(
                        align='center',
                        baseline='bottom',
                        dy=0,
                        fontSize=10
                    ).encode(
                        x=alt.X('month_name:N', sort=list(yoy_data['month_name'])),
                        y='yoy_pct:Q',
                        text=alt.Text('yoy_pct:Q', format='+.1f%'),
                        color=alt.condition(
                            alt.datum.yoy_pct >= 0,
                            alt.value('#4CAF50'),  # green for positive
                            alt.value('#F44336')   # red for negative
                        )
                    )
                    
                    # Combine the waterfall chart with labels
                    final_waterfall_chart = waterfall_chart + waterfall_labels
                    
                    st.altair_chart(final_waterfall_chart, use_container_width=True)
                    
                    # Display data table
                    st.write(f'''
                    **Monthly Average Metrics for {selected_date_type}**
                    ''')
                    
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
                    st.dataframe(pivot_table.sort_values('month').reset_index(drop=True), use_container_width=True, hide_index=True)
                
                # Tab 2: Lunar New Year Effect
                elif selected_option == "Lunar New Year Effect":
                    st.subheader("Lunar New Year Effect Analysis")
                    
                    # Lunar New Year period selection UI
                    st.caption('Lunar New Year Period Selection')
                    lny_period_mode = st.segmented_control(
                        'Choose LNY period selection mode:',
                        ["Auto (Hard Code)", "Manual Selection"],
                        key='lny_period_mode',
                        default='Auto (Hard Code)'
                    )

                    # Get default Lunar New Year periods (for manual selection)
                    current_lny_start, current_lny_end = get_lunar_new_year_periods(target_year)
                    prev_lny_start, prev_lny_end = get_lunar_new_year_periods(target_year - 1)

                    if lny_period_mode == 'Auto (Hard Code)':
                        lny_periods = {
                            'current': (current_lny_start, current_lny_end),
                            'prev': (prev_lny_start, prev_lny_end)
                        }
                    else:
                        try:
                            manual_current = st.date_input(
                                f'**{target_year} LNY Period**',
                                value=(current_lny_start.date(), current_lny_end.date()),
                                key='manual_current_lny',
                                min_value=projection_data['grass_date'].min().date(),
                                max_value=projection_data['grass_date'].max().date()
                            )
                        except StreamlitAPIException as e:
                            st.error(f"‼️ LNY period default value is out of data range. Please check your data or adjust the default period.\n\nError: {e}")
                            return
                        try:
                            manual_prev = st.date_input(
                                f'**{target_year - 1} LNY Period**',
                                value=(prev_lny_start.date(), prev_lny_end.date()),
                                key='manual_prev_lny',
                                min_value=projection_data['grass_date'].min().date(),
                                max_value=projection_data['grass_date'].max().date()
                            )
                        except StreamlitAPIException as e:
                            st.error(f"‼️ LNY period default value is out of data range. Please check your data or adjust the default period.\n\nError: {e}")
                            return
                        lny_periods = {
                            'current': (pd.Timestamp(manual_current[0]), pd.Timestamp(manual_current[1])),
                            'prev': (pd.Timestamp(manual_prev[0]), pd.Timestamp(manual_prev[1]))
                        }

                    # Calculate LNY vs BAU for both years
                    lny_vs_bau_current = calculate_lny_vs_bau(projection_data, lny_periods['current'][0], lny_periods['current'][1])
                    lny_vs_bau_prev = calculate_lny_vs_bau(projection_data, lny_periods['prev'][0], lny_periods['prev'][1])

                    st.write('---')
                    st.subheader('Lunar New Year Period vs Mean of Business-As-Usual')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'#### *{target_year} LNY vs {lny_periods["current"][0].strftime("%Y-%m")} BAU*')
                        if not lny_vs_bau_current.empty:
                            # Add metric
                            total_diff = lny_vs_bau_current['diff'].sum()
                            total_bau = lny_vs_bau_current['bau_avg'].sum() if not lny_vs_bau_current['bau_avg'].isnull().all() else 0
                            total_pct = (total_diff / total_bau) if total_bau else 0
                            st.metric(label="Total Change", value=f"{total_diff:,.0f}", delta=f"{total_pct:,.2%}")
                            # Show chart
                            chart_data = lny_vs_bau_current.copy()
                            chart_data['date_str'] = chart_data['grass_date'].dt.strftime('%Y-%m-%d')
                            lny_bau_chart = alt.Chart(chart_data).mark_bar().encode(
                                x=alt.X('date_str:N', title='Date'),
                                y=alt.Y('pct_change:Q', title='Change vs BAU (%)', axis=alt.Axis(format='%')), 
                                color=alt.condition(alt.datum.pct_change > 0, alt.value('green'), alt.value('red')),
                                tooltip=['date_str', 'metrics', 'bau_avg', alt.Tooltip('pct_change', format='.2%')]
                            ).properties(title=f'{target_year} LNY vs {lny_periods["current"][0].strftime("%Y-%m")} BAU day-by-day comparison')
                            st.altair_chart(lny_bau_chart, use_container_width=True)
                            # Show dataframe
                            st.dataframe(lny_vs_bau_current[['grass_date', 'metrics', 'bau_avg', 'diff', 'pct_change']].reset_index(drop=True), use_container_width=True, hide_index=True)
                        else:
                            st.error('‼️ No data for selected period.')
                    with col2:
                        st.markdown(f'#### *{target_year-1} LNY vs {lny_periods["prev"][0].strftime("%Y-%m")} BAU*')
                        if not lny_vs_bau_prev.empty:
                            # Add metric
                            total_diff = lny_vs_bau_prev['diff'].sum()
                            total_bau = lny_vs_bau_prev['bau_avg'].sum() if not lny_vs_bau_prev['bau_avg'].isnull().all() else 0
                            total_pct = (total_diff / total_bau) if total_bau else 0
                            st.metric(label="Total Change", value=f"{total_diff:,.0f}", delta=f"{total_pct:,.2%}")
                            # Show chart
                            chart_data = lny_vs_bau_prev.copy()
                            chart_data['date_str'] = chart_data['grass_date'].dt.strftime('%Y-%m-%d')
                            lny_bau_chart = alt.Chart(chart_data).mark_bar().encode(
                                x=alt.X('date_str:N', title='Date'),
                                y=alt.Y('pct_change:Q', title='Change vs BAU (%)', axis=alt.Axis(format='%')),
                                color=alt.condition(alt.datum.pct_change > 0, alt.value('green'), alt.value('red')),
                                tooltip=['date_str', 'metrics', 'bau_avg', alt.Tooltip('pct_change', format='.2%')]
                            ).properties(title=f'{target_year-1} LNY vs {lny_periods["prev"][0].strftime("%Y-%m")} BAU day-by-day comparison')
                            st.altair_chart(lny_bau_chart, use_container_width=True)
                            # Show dataframe
                            st.dataframe(lny_vs_bau_prev[['grass_date', 'metrics', 'bau_avg', 'diff', 'pct_change']].reset_index(drop=True), use_container_width=True, hide_index=True)
                        else:
                            st.error('‼️ No data for selected period.')
                    st.write('---')

                    # Add YoY of Lunar New Year Period section
                    st.subheader('YoY of Lunar New Year Period')
                    # Ensure both years' data are not empty and lny_day_seq exists
                    if not lny_vs_bau_current.empty and not lny_vs_bau_prev.empty:
                        # Add lny_day_seq to both years' data
                        lny_vs_bau_current = lny_vs_bau_current.copy()
                        lny_vs_bau_prev = lny_vs_bau_prev.copy()
                        lny_vs_bau_current['lny_day_seq'] = (lny_vs_bau_current['grass_date'] - lny_vs_bau_current['grass_date'].min()).dt.days + 1
                        lny_vs_bau_prev['lny_day_seq'] = (lny_vs_bau_prev['grass_date'] - lny_vs_bau_prev['grass_date'].min()).dt.days + 1
                        # Align using lny_day_seq
                        yoy_df = pd.merge(
                            lny_vs_bau_current[['lny_day_seq', 'metrics']],
                            lny_vs_bau_prev[['lny_day_seq', 'metrics']],
                            left_on='lny_day_seq', right_on='lny_day_seq',
                            suffixes=('_current', '_prev')
                        )
                        yoy_df['yoy_diff'] = yoy_df['metrics_current'] - yoy_df['metrics_prev']
                        yoy_df['yoy_pct'] = (yoy_df['metrics_current'] - yoy_df['metrics_prev']) / yoy_df['metrics_prev'].replace(0, np.nan)
                        yoy_df['day_seq'] = yoy_df['lny_day_seq'].astype(str)
                        st.dataframe(yoy_df[['lny_day_seq', 'metrics_current', 'metrics_prev', 'yoy_diff', 'yoy_pct']].reset_index(drop=True), use_container_width=True, hide_index=True)
                        # Bar chart
                        yoy_chart = alt.Chart(yoy_df).mark_bar().encode(
                            x=alt.X('day_seq:N', title='LNY Day Sequence'),
                            y=alt.Y('yoy_pct:Q', title='YoY Change (%)', axis=alt.Axis(format='%')),
                            color=alt.condition(alt.datum.yoy_pct > 0, alt.value('green'), alt.value('red')),
                            tooltip=['day_seq', 'metrics_current', 'metrics_prev', alt.Tooltip('yoy_diff', format='.2f'), alt.Tooltip('yoy_pct', format='.2%')]
                        ).properties(title=f'LNY YoY Change ({target_year} vs {target_year-1})')
                        st.altair_chart(yoy_chart, use_container_width=True)
                        # Total YoY change metric
                        total_current = yoy_df['metrics_current'].sum()
                        total_prev = yoy_df['metrics_prev'].sum()
                        yoy_total_diff = total_current - total_prev
                        yoy_total_pct = (yoy_total_diff / total_prev) if total_prev else 0
                        st.metric(label="Total LNY YoY Change", value=f"{yoy_total_diff:,.2f}", delta=f"{yoy_total_pct:.2%}")
                    else:
                        st.error('‼️ No sufficient data for YoY analysis of LNY period.')
                
                # Tab 3: BAU MoM Analysis
                elif selected_option == "BAU MoM":
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
                        st.write('---')
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
                            use_container_width=True,
                            hide_index=True
                        )
                
                # Tab 4: Monthly Date_Type Uplift
                elif selected_option == "Monthly Uplift":
                    st.subheader("Monthly Date-Type Uplift Analysis")
                    st.caption("Calculating monthly uplift percentages for special day types compared to BAU")
                    
                    # Calculate monthly special day uplift
                    monthly_uplift_data = calculate_monthly_uplift_datetype_vs_bau(projection_data, target_year)
                    
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
                            
                            # Add text labels to the waterfall chart
                            waterfall_labels = alt.Chart(chart_data).mark_text(
                                align='center',
                                baseline='bottom',
                                dy=0,
                                fontSize=10
                            ).encode(
                                x=alt.X('month_year:N', sort=None),
                                y='uplift_pct:Q',
                                text=alt.Text('uplift_pct:Q', format='+.1f%'),
                                color=alt.condition(
                                    alt.datum.uplift_pct > 0,
                                    alt.value("green"),
                                    alt.value("red")
                                )
                            )
                            
                            # Combine the waterfall chart with labels
                            final_waterfall_chart = waterfall_chart + waterfall_labels
                    
                            st.altair_chart(final_waterfall_chart, use_container_width=True)
                        
                        # Display data table
                        st.write('---')
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
                            use_container_width=True,
                            hide_index=True
                        )
                
                # Tab 5: Quarterly Special Day Uplift
                elif selected_option == "Quarterly Uplift":
                    st.subheader("Quarterly Date-Type Uplift Analysis")
                    st.caption("Calculating quarterly uplift percentages for special day types compared to BAU")
                    
                    # Calculate quarterly special day uplift
                    quarterly_uplift_data = calculate_quarterly_uplift_datetype_vs_bau(projection_data, target_year)
                    
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
                                title=f'{selected_day_type} Quarterly Uplift vs BAU (%)',
                            )
                            
                            # Add text labels to the waterfall chart
                            waterfall_labels = alt.Chart(chart_data).mark_text(
                                align='center',
                                baseline='bottom',
                                dy=0,
                                fontSize=10
                            ).encode(
                                x=alt.X('quarter_year:N', sort=None),
                                y='uplift_pct:Q',
                                text=alt.Text('uplift_pct:Q', format='+.1f%'),
                                color=alt.condition(
                                    alt.datum.uplift_pct > 0,
                                    alt.value("green"),
                                    alt.value("red")
                                )
                            )
                            
                            # Combine the waterfall chart with labels
                            final_waterfall_chart = waterfall_chart + waterfall_labels
                            
                            st.altair_chart(final_waterfall_chart, use_container_width=True)
                        
                        # Display data table
                        st.write('---')
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
                            use_container_width=True,
                            hide_index=True
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
        with st.expander("Expand Settings", expanded=False):
            # ❶ Lunar New Year Effect
            st.write('''
            ##### ❶ Lunar New Year Effect        
            ''')
            lny_input_type = st.segmented_control(
                "Input Type",
                ["Slider", "Direct Input"],
                key="lny_input_type",
                default="Direct Input"
            )
            # Use unique key for each widget to avoid StreamlitDuplicateElementId error
            if lny_input_type == "Slider":
                lny_effect = st.select_slider(
                    "---",
                    options=[float(f"{-30.00 + i * 0.1:.1f}") for i in range(601)],  # -30% to +30% in 0.1% increments
                    value=0.00,
                    format_func=lambda x: f"{x:+.2f}%",
                    key='lny_effect_slider'  # unique key
                )
            else:
                lny_effect = st.number_input(
                    "(%)",
                    min_value=-30.00,
                    max_value=30.00,
                    value=0.00,
                    step=0.01,
                    format="%.2f",
                    key='lny_effect_input'  # unique key
                )
            # ❷ BAU MoM Growth Effect
            st.write('''
            ##### ❷ BAU MoM Growth Effect        
            ''')
            bau_mom_input_type = st.segmented_control(
                "Input Type",
                ["Slider", "Direct Input"],
                key="bau_mom_input_type",
                default="Direct Input"
            )
            if bau_mom_input_type == "Slider":
                bau_mom_effect = st.select_slider(
                    "---",
                    options=[float(f"{-30.00 + i * 0.1:.1f}") for i in range(601)],  # -30% to +30% in 0.1% increments
                    value=0.00,
                    format_func=lambda x: f"{x:+.2f}%",
                    key='bau_mom_effect_slider'  # unique key
                )
            else:
                bau_mom_effect = st.number_input(
                    "(%)",
                    min_value=-30.00,
                    max_value=30.00,
                    value=0.00,
                    step=0.01,
                    format="%.2f",
                    key='bau_mom_effect_input'  # unique key
                )
            # ❸ Monthly Uplift Effect (of Date Type)
            st.write('''
            ##### ❸ Monthly Uplift Effect (of Date Type)       
            ''')
            monthly_uplift_input_type = st.segmented_control(
                "Input Type",
                ["Slider", "Direct Input"],
                key="monthly_uplift_input_type",
                default="Direct Input"
            )
            if monthly_uplift_input_type == "Slider":
                monthly_uplift = st.select_slider(
                    "---",
                    options=[float(f"{-30.00 + i * 0.1:.1f}") for i in range(601)],  # -30% to +30% in 0.1% increments
                    value=0.00,
                    format_func=lambda x: f"{x:+.2f}%",
                    key='monthly_uplift_slider'  # unique key
                )
            else:
                monthly_uplift = st.number_input(
                    "(%)",
                    min_value=-30.00,
                    max_value=30.00,
                    value=0.00,
                    step=0.01,
                    format="%.2f",
                    key='monthly_uplift_input'  # unique key
                )
            # ❹ Quarterly Uplift Effect (of Date Type)
            st.write('''
            ##### ❹ Quarterly Uplift Effect (of Date Type)       
            ''')
            quarterly_uplift_input_type = st.segmented_control(
                "Input Type",
                ["Slider", "Direct Input"],
                key="quarterly_uplift_input_type",
                default="Direct Input"
            )
            if quarterly_uplift_input_type == "Slider":
                quarterly_uplift = st.select_slider(
                    "---",
                    options=[float(f"{-30.00 + i * 0.1:.1f}") for i in range(601)],  # -30% to +30% in 0.1% increments
                    value=0.00,
                    format_func=lambda x: f"{x:+.2f}%",
                    key='quarterly_uplift_slider'  # unique key
                )
            else:
                quarterly_uplift = st.number_input(
                    "(%)",
                    min_value=-30.00,
                    max_value=30.00,
                    value=0.00,
                    step=0.01,
                    format="%.2f",
                    key='quarterly_uplift_input'  # unique key
                )
            # ❺ Additional Effect
            st.write('''
            ##### ❺ Additional Effect    
            ''')
            additional_input_type = st.segmented_control(
                "Input Type",
                ["Slider", "Direct Input"],
                key="additional_input_type",
                default="Direct Input"
            )
            if additional_input_type == "Slider":
                additional_input = st.select_slider(
                    "---",
                    options=[float(f"{-30.00 + i * 0.1:.1f}") for i in range(601)],  # -30% to +30% in 0.1% increments
                    value=0.00,
                    format_func=lambda x: f"{x:+.2f}%",
                    key='additional_input_slider'  # unique key
                )
            else:
                additional_input = st.number_input(
                    "(%)",
                    min_value=-30.00,
                    max_value=30.00,
                    value=0.00,
                    step=0.01,
                    format="%.2f",
                    key='additional_input_input'  # unique key
                )

        # Calculate total effect
        total_effect_pct = lny_effect + bau_mom_effect + monthly_uplift + quarterly_uplift + additional_input
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
            'Factor': ['Lunar New Year Effect', 'BAU MoM Growth', 'Monthly Uplift', 'Quarterly Uplift', 'Additional Effect', 'Total Effect'],
            'Percentage': [lny_effect, bau_mom_effect, monthly_uplift, quarterly_uplift, additional_input, total_effect_pct],
            'Value Impact': [
                baseline * (lny_effect / 100),
                baseline * (bau_mom_effect / 100),
                baseline * (monthly_uplift / 100),
                baseline * (quarterly_uplift / 100),
                baseline * (additional_input / 100),
                baseline * (total_effect_pct / 100)
            ]
        })
        
        # Format the percentage and value columns
        effect_data['Percentage'] = effect_data['Percentage'].apply(lambda x: f"{x:+.2f}%")
        effect_data['Value Impact'] = effect_data['Value Impact'].apply(lambda x: f"{x:+,.2f}")
        
        st.dataframe(effect_data, use_container_width=True, hide_index=True)
    
    else:
        # Display welcome message when no tab is selected (should not happen)
        st.title("🤖 Projection Automation Tool")
        st.markdown("""
        Please select an option from the sidebar to get started.
        """)

if __name__ == "__main__":
    main()
