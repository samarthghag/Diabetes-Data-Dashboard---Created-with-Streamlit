import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.figure_factory as ff

# Set Streamlit page config
st.set_page_config(
    page_title="Diabetes Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Diabetes Data Dashboard - Created with Streamlit"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stat-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    return df, numeric_columns

df, numeric_columns = load_data()

# App title and description
st.markdown('<h1 class="main-header">Diabetes Data Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
This interactive dashboard provides analysis of diabetes dataset with visualizations to help understand key factors and relationships.
Use the sidebar to customize your analysis and explore the data through different tabs below.
""")

# Sidebar with filters and options
st.sidebar.markdown('<h3>Dashboard Controls</h3>', unsafe_allow_html=True)

# Add filters to sidebar
st.sidebar.markdown('## Data Filters')
# Age filter
age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider('Age Range', age_min, age_max, (age_min, age_max))

# Glucose filter
glucose_min, glucose_max = int(df['Glucose'].min()), int(df['Glucose'].max())
glucose_range = st.sidebar.slider('Glucose Range', glucose_min, glucose_max, (glucose_min, glucose_max))

# BMI filter
bmi_min, bmi_max = float(df['BMI'].min()), float(df['BMI'].max())
bmi_range = st.sidebar.slider('BMI Range', bmi_min, bmi_max, (bmi_min, bmi_max))

# Outcome filter
outcome_filter = st.sidebar.multiselect('Outcome', options=df['Outcome'].unique(), default=df['Outcome'].unique())

# Apply filters
filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) & 
                (df['Glucose'] >= glucose_range[0]) & (df['Glucose'] <= glucose_range[1]) &
                (df['BMI'] >= bmi_range[0]) & (df['BMI'] <= bmi_range[1]) &
                (df['Outcome'].isin(outcome_filter))]

# Display filtered data count
st.sidebar.markdown(f"**Filtered Data: {filtered_df.shape[0]} records**")

# Display reset button
if st.sidebar.button('Reset Filters'):
    st.rerun()

# Create tabs for organization
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Data Explorer", "📈 Visualizations", "📊 Advanced Analytics"])

with tab1:
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stat-container">', unsafe_allow_html=True)
        st.metric("Total Records", f"{filtered_df.shape[0]}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="stat-container">', unsafe_allow_html=True)
        diabetic_pct = (filtered_df['Outcome'] == 1).mean() * 100
        st.metric("Diabetic Patients", f"{diabetic_pct:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="stat-container">', unsafe_allow_html=True)
        avg_age = filtered_df['Age'].mean()
        st.metric("Average Age", f"{avg_age:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="stat-container">', unsafe_allow_html=True)
        avg_glucose = filtered_df['Glucose'].mean()
        st.metric("Avg Glucose", f"{avg_glucose:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h3>Data Summary</h3>', unsafe_allow_html=True)
    
    # Show data summary statistics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="stat-container">', unsafe_allow_html=True)
        st.write("Descriptive Statistics")
        st.dataframe(filtered_df.describe().round(2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-container">', unsafe_allow_html=True)
        st.write("Feature Distribution by Outcome")
        outcome_dist = filtered_df.groupby('Outcome').mean().round(2).T
        outcome_dist.columns = ['Non-Diabetic', 'Diabetic']
        st.dataframe(outcome_dist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<h2 class="sub-header">Data Explorer</h2>', unsafe_allow_html=True)
    
    # Show dataframe with expanded functionality
    st.markdown('<h3>Raw Data</h3>', unsafe_allow_html=True)
    
    # Add search functionality
    search_term = st.text_input("Search in data", "")
    
    if search_term:
        # Search across all columns that can be converted to string
        search_results = filtered_df[filtered_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)]
        st.dataframe(search_results, use_container_width=True)
    else:
        st.dataframe(filtered_df, use_container_width=True)
    
    # Add download button
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(filtered_df)
    st.download_button(
        "Download Filtered Data as CSV",
        csv,
        "filtered_diabetes_data.csv",
        "text/csv",
        key='download-csv'
    )

# Visualization controls moved to the tabs
with tab3:
    st.markdown('<h2 class="sub-header">Data Visualizations</h2>', unsafe_allow_html=True)
    
    # Organized visualization options
    viz_col1, viz_col2 = st.columns([1, 3])
    
    with viz_col1:
        st.markdown('<div class="stat-container">', unsafe_allow_html=True)
        st.markdown("### Chart Controls")
        
        # Chart type selector
        chart_type = st.radio(
            "Select Chart Type",
            ["Histogram", "Scatterplot", "Box Plot", "Distribution Plot"]
        )
        
        # Column selections based on chart type
        if chart_type in ["Histogram", "Distribution Plot"]:
            col1 = st.selectbox("Select Feature", numeric_columns)
            
        elif chart_type == "Scatterplot":
            col1 = st.selectbox("X-axis", numeric_columns)
            col2 = st.selectbox("Y-axis", numeric_columns, 
                              index=1 if len(numeric_columns) > 1 else 0)
            hue_col = st.selectbox("Color by", ["None", "Outcome"] + numeric_columns.tolist())
            
        elif chart_type == "Box Plot":
            col1 = st.selectbox("Feature to Plot", numeric_columns)
            
        # Color theme - Using valid Plotly colorscales
        color_theme = st.selectbox(
            "Color Theme",
            ["viridis", "plasma", "inferno", "magma", "cividis", "blues", "greens", "reds", "purples"]
        )
        
        # Additional customization
        show_kde = st.checkbox("Show KDE", value=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with viz_col2:
        st.markdown('<div class="stat-container">', unsafe_allow_html=True)
        st.markdown("### Visualization")
        
        # Dynamic chart rendering based on selection
        if chart_type == "Histogram":
            fig = px.histogram(
                filtered_df, 
                x=col1,
                color="Outcome" if "Outcome" in filtered_df.columns else None,
                marginal="box" if show_kde else None,
                color_discrete_sequence=px.colors.sequential.Plasma,
                opacity=0.8,
                title=f"Histogram of {col1}"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Scatterplot":
            fig = px.scatter(
                filtered_df,
                x=col1,
                y=col2,
                color=filtered_df[hue_col] if hue_col != "None" else None,
                color_continuous_scale=color_theme if hue_col not in ["None", "Outcome"] else None,
                title=f"Scatterplot: {col1} vs {col2}",
                opacity=0.7,
                size_max=10,
                hover_data=filtered_df.columns
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Box Plot":
            fig = px.box(
                filtered_df,
                y=col1,
                x="Outcome" if "Outcome" in filtered_df.columns else None,
                color="Outcome" if "Outcome" in filtered_df.columns else None,
                points="all",
                title=f"Box Plot of {col1} by Outcome",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Distribution Plot":
            # Create distplots with plotly
            diabetic = filtered_df[filtered_df['Outcome'] == 1][col1].dropna()
            non_diabetic = filtered_df[filtered_df['Outcome'] == 0][col1].dropna()
            
            # Create distplots if both have data
            if len(diabetic) > 0 and len(non_diabetic) > 0:
                hist_data = [diabetic, non_diabetic]
                group_labels = ['Diabetic', 'Non-Diabetic']
                
                fig = ff.create_distplot(
                    hist_data, 
                    group_labels, 
                    show_hist=True, 
                    show_rug=False,
                    colors=['red', 'blue']
                )
                fig.update_layout(
                    title=f'Distribution Plot of {col1} by Outcome',
                    xaxis_title=col1,
                    yaxis_title='Density',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Not enough data to create distribution plot after filtering")
                
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Correlation Heatmap section
    st.markdown('<h3>Correlation Analysis</h3>', unsafe_allow_html=True)
    st.markdown('<div class="stat-container">', unsafe_allow_html=True)
    
    corr_col1, corr_col2 = st.columns([1, 3])
    
    with corr_col1:
        # Correlation options
        corr_method = st.radio(
            "Correlation Method",
            ["pearson", "spearman", "kendall"]
        )
        
        mask_option = st.checkbox("Mask Upper Triangle", value=True)
        
        # Valid plotly colorscales
        valid_colorscales = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'blues', 'greens', 'reds', 'purples', 'oranges',
            'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'turbo'
        ]
        
        color_scale = st.selectbox(
            "Color Scale",
            valid_colorscales
        )
        
    with corr_col2:
        # Create correlation matrix
        corr_matrix = filtered_df.corr(method=corr_method).round(2)
        
        # Create mask for upper triangle if selected
        mask = np.zeros_like(corr_matrix)
        if mask_option:
            mask[np.triu_indices_from(mask)] = True
            # Apply the mask to the correlation matrix
            corr_matrix_masked = corr_matrix.copy()
            corr_matrix_masked.values[mask.astype(bool)] = np.nan
            # Use the masked matrix
            display_matrix = corr_matrix_masked
        else:
            display_matrix = corr_matrix
        
        # Plot correlation heatmap using plotly
        fig = px.imshow(
            display_matrix,
            text_auto=True,
            color_continuous_scale=color_scale,
            title=f"Correlation Heatmap ({corr_method})",
            height=600
        )
        fig.update_layout(
            height=500,
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# Advanced Analytics tab
with tab4:
    st.markdown('<h2 class="sub-header">Advanced Analytics</h2>', unsafe_allow_html=True)
    
    # Feature relationships by outcome
    st.markdown('<h3>Feature Relationships by Outcome</h3>', unsafe_allow_html=True)
    st.markdown('<div class="stat-container">', unsafe_allow_html=True)
    
    feat_col1, feat_col2 = st.columns([1, 3])
    
    with feat_col1:
        selected_features = st.multiselect(
            "Select Features to Compare",
            numeric_columns,
            default=["Glucose", "BMI", "Age"] if all(x in numeric_columns for x in ["Glucose", "BMI", "Age"]) else numeric_columns[:3]
        )
        
        if len(selected_features) < 2:
            st.warning("Please select at least 2 features")
    
    with feat_col2:
        if len(selected_features) >= 2:
            # Create pairplot with plotly
            fig = px.scatter_matrix(
                filtered_df,
                dimensions=selected_features,
                color="Outcome" if "Outcome" in filtered_df.columns else None,
                opacity=0.7,
                title="Feature Relationships Matrix"
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistical analysis
    st.markdown('<h3>Statistical Analysis by Outcome</h3>', unsafe_allow_html=True)
    
    stat_col1, stat_col2 = st.columns(2)
    
    with stat_col1:
        st.markdown('<div class="stat-container">', unsafe_allow_html=True)
        if "Outcome" in filtered_df.columns:
            diabetic_data = filtered_df[filtered_df["Outcome"] == 1].describe().T
            diabetic_data["count"] = diabetic_data["count"].astype(int)
            st.write("Diabetic Patients Statistics")
            st.dataframe(diabetic_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with stat_col2:
        st.markdown('<div class="stat-container">', unsafe_allow_html=True)
        if "Outcome" in filtered_df.columns:
            non_diabetic_data = filtered_df[filtered_df["Outcome"] == 0].describe().T
            non_diabetic_data["count"] = non_diabetic_data["count"].astype(int)
            st.write("Non-Diabetic Patients Statistics")
            st.dataframe(non_diabetic_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)