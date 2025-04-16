# Projection Automation Tool

A powerful Streamlit application for analyzing business metrics and creating projections based on historical data, designed for business analysts and data scientists.

## Features

### Navigation Structure
The app is organized into three main sections accessible from the sidebar:

1. **Upload Data**: Upload and preview your business metrics data
2. **Projection Analysis**: Analyze historical data through multiple visualization methods
3. **Manual Projection**: Create custom projections by adjusting various business factors

### Analysis Capabilities
- **Yearly Trend Analysis**: Compare metrics between years with interactive line charts and YoY waterfall plots
- **Lunar New Year Effect**: Analyze the impact of Lunar New Year on business metrics
- **BAU Month-on-Month Growth**: Visualize the month-on-month growth rates for business-as-usual days
- **Special Day Type Analysis**: Visualize monthly and quarterly uplift effects from special day types compared to business-as-usual days
- **Manual Projection Tool**: Create custom projections by adjusting factors like Lunar New Year effect, BAU MoM growth, and special day uplifts

## Data Format Requirements

Your CSV file should include the following columns:
- `grass_date`: Date in YYYY-MM-DD format
- `date_type`: Category of day (e.g., 'bau', '1st_spike', '2nd_spike', '3rd_spike', 'FSS')
- `metrics`: Numerical values (can be GMV, Orders, Login User Count, etc.)

## How to Run

1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your browser to the URL displayed in the terminal (typically http://localhost:8501)

## Usage Guide

1. **Upload Data Tab**:
   - Upload your CSV file with business metrics data
   - Review the data preview and date range information

2. **Projection Analysis Tab**:
   - Select the year for analysis
   - Navigate through the five analysis steps using the horizontal menu
   - View visualizations and data tables for each analysis type

3. **Manual Projection Tab**:
   - Enter a baseline value
   - Adjust the different effect sliders to see how they impact the projection
   - View the breakdown of effects and the final projected value
