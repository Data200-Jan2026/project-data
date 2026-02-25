import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, zscore, ttest_ind
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="E-Commerce Statistical Analysis", layout="wide")

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

st.title("📊 E-Commerce Customer Behavior - Statistical Analysis")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Analysis Section", [
    "📈 Data Overview",
    "🔗 Correlation Analysis",
    "📉 Distribution & Diagnostics",
    "🎯 Outlier Detection",
    "📋 Group Comparisons (T-Tests)"
])

# File uploader
st.sidebar.markdown("---")

# Load and process data
@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file)
    
    # Process data
    df_processed = df.copy()
    
    # Handle missing values
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            if df_processed[col].dtype in ['int64', 'float64']:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
            else:
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # Encode categorical columns
    for col in ['Gender', 'Membership Type', 'Satisfaction Level']:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
    
    # Convert discount to int
    if 'Discount Applied' in df_processed.columns:
        df_processed['Discount Applied'] = df_processed['Discount Applied'].astype(int)
    
    return df, df_processed

df, df_processed = load_and_process_data("ecommerce.csv")

# Numeric columns for analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ==================== PAGE 1: DATA OVERVIEW ====================
if page == "📈 Data Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Customers", df.shape[0])
    col2.metric("Number of Features", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    st.subheader("First 5 Records")
    st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Types:**")
        st.write(df.dtypes)
    
    with col2:
        st.write("**Basic Statistics:**")
        st.dataframe(df.describe(), use_container_width=True)

# ==================== PAGE 2: CORRELATION ANALYSIS ====================
elif page == "🔗 Correlation Analysis":
    st.header("Correlation Analysis")
    
    # Default correlation pairs
    correlation_pairs = [
        ('Age', 'Total Spend'),
        ('Items Purchased', 'Total Spend'),
        ('Average Rating', 'Total Spend'),
        ('Days Since Last Purchase', 'Total Spend'),
        ('Average Rating', 'Items Purchased')
    ]
    
    # Filter pairs that exist in data
    correlation_pairs = [(v1, v2) for v1, v2 in correlation_pairs 
                         if v1 in df.columns and v2 in df.columns]
    
    # Tabs for Pearson and Spearman
    tab1, tab2, tab3 = st.tabs(["Pearson Correlation", "Spearman Correlation", "Correlation Matrix"])
    
    # PEARSON CORRELATION
    with tab1:
        st.subheader("Pearson Correlation (Linear Relationships)")
        
        pearson_data = []
        for var1, var2 in correlation_pairs:
            r, p_value = pearsonr(df[var1], df[var2])
            
            if abs(r) > 0.8:
                strength = "Very Strong"
            elif abs(r) > 0.6:
                strength = "Strong"
            elif abs(r) > 0.4:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            direction = "Positive" if r > 0 else "Negative"
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            pearson_data.append({
                'Variables': f'{var1} ↔ {var2}',
                'Correlation (r)': f'{r:.4f}',
                'P-value': f'{p_value:.6f}',
                'Strength': strength,
                'Direction': direction,
                'Significance': sig
            })
        
        pearson_df = pd.DataFrame(pearson_data)
        st.dataframe(pearson_df, use_container_width=True)
        
        with st.expander("📖 Interpretation Guide"):
            st.markdown("""
            - **r > 0.8**: Very Strong correlation
            - **0.6 < r ≤ 0.8**: Strong correlation
            - **0.4 < r ≤ 0.6**: Moderate correlation
            - **r ≤ 0.4**: Weak correlation
            - **p < 0.001 (\\***)**: Highly significant
            - **p < 0.01 (\\**)**: Very significant
            - **p < 0.05 (\\*)**: Significant
            """)
    
    # SPEARMAN CORRELATION
    with tab2:
        st.subheader("Spearman Correlation (Non-parametric)")
        
        spearman_data = []
        for var1, var2 in correlation_pairs:
            rho, p_value = spearmanr(df[var1], df[var2])
            
            if abs(rho) > 0.8:
                strength = "Very Strong"
            elif abs(rho) > 0.6:
                strength = "Strong"
            elif abs(rho) > 0.4:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            direction = "Positive" if rho > 0 else "Negative"
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            spearman_data.append({
                'Variables': f'{var1} ↔ {var2}',
                'Correlation (ρ)': f'{rho:.4f}',
                'P-value': f'{p_value:.6f}',
                'Strength': strength,
                'Direction': direction,
                'Significance': sig
            })
        
        spearman_df = pd.DataFrame(spearman_data)
        st.dataframe(spearman_df, use_container_width=True)
        
        with st.expander("📖 When to Use Spearman"):
            st.markdown("""
            - Non-parametric alternative to Pearson
            - Robust to outliers
            - Measures monotonic relationships
            - Useful when data is not normally distributed
            """)
    
    # CORRELATION MATRIX
    with tab3:
        st.subheader("Correlation Matrix Heatmap")
        
        numeric_df = df[numeric_cols]
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                    square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
        ax.set_title('Pearson Correlation Matrix - Numerical Features', fontweight='bold')
        st.pyplot(fig)

# ==================== PAGE 3: DISTRIBUTION & DIAGNOSTICS ====================
elif page == "📉 Distribution & Diagnostics":
    st.header("Distribution Analysis & Diagnostics")
    
    st.subheader("Skewness & Kurtosis Analysis")
    
    diagnostic_data = []
    for col in numeric_cols:
        skewness = stats.skew(df[col])
        kurtosis = stats.kurtosis(df[col])
        mean = df[col].mean()
        median = df[col].median()
        
        if abs(skewness) < 0.5:
            skew_type = "Approximately Symmetric"
        elif skewness > 0:
            skew_type = "Right-skewed"
        else:
            skew_type = "Left-skewed"
        
        diagnostic_data.append({
            'Variable': col,
            'Mean': f'{mean:.2f}',
            'Median': f'{median:.2f}',
            'Skewness': f'{skewness:.4f}',
            'Kurtosis': f'{kurtosis:.4f}',
            'Distribution': skew_type
        })
    
    diagnostic_df = pd.DataFrame(diagnostic_data)
    st.dataframe(diagnostic_df, use_container_width=True)
    
    # Tabs for visualizations
    tab1, tab2 = st.tabs(["Distributions", "Q-Q Plots"])
    
    with tab1:
        st.subheader("Distribution Plots (Histogram)")
        
        # Create distribution plots
        num_cols = len(numeric_cols)
        cols_per_row = 2
        num_rows = (num_cols + cols_per_row - 1) // cols_per_row
        
        for i in range(num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < num_cols:
                    with cols[j]:
                        col = numeric_cols[idx]
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.hist(df[col], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                        ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
                        ax.axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label='Median')
                        ax.set_title(f'{col} Distribution')
                        ax.set_ylabel('Frequency')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
    
    with tab2:
        st.subheader("Q-Q Plots (Normality Assessment)")
        
        num_cols_list = len(numeric_cols)
        cols_per_row = 2
        num_rows = (num_cols_list + cols_per_row - 1) // cols_per_row
        
        for i in range(num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < num_cols_list:
                    with cols[j]:
                        col = numeric_cols[idx]
                        fig, ax = plt.subplots(figsize=(6, 4))
                        stats.probplot(df[col], dist="norm", plot=ax)
                        ax.set_title(f'{col} Q-Q Plot')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
    
    with st.expander("📖 Interpretation Guide"):
        st.markdown("""
        **Skewness:**
        - |skewness| < 0.5: Approximately symmetric
        - skewness > 0.5: Right-skewed (tail on right)
        - skewness < -0.5: Left-skewed (tail on left)
        
        **Q-Q Plots:**
        - Points on diagonal: Data is normally distributed ✓
        - Points deviate from line: Non-normal distribution
        - S-shaped pattern: Data is non-normal
        """)

# ==================== PAGE 4: OUTLIER DETECTION ====================
elif page == "🎯 Outlier Detection":
    st.header("Outlier Detection (IQR Method)")
    
    st.subheader("Outlier Summary")
    
    outlier_summary = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outliers_mask.sum()
        outlier_percentage = (outlier_count / len(df)) * 100
        
        outlier_summary.append({
            'Variable': col,
            'Q1': f'{Q1:.2f}',
            'Q3': f'{Q3:.2f}',
            'IQR': f'{IQR:.2f}',
            'Lower Bound': f'{lower_bound:.2f}',
            'Upper Bound': f'{upper_bound:.2f}',
            'Outlier Count': outlier_count,
            'Outlier %': f'{outlier_percentage:.2f}%'
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    st.dataframe(outlier_df, use_container_width=True)
    
    st.subheader("Box Plots")
    
    num_cols_list = len(numeric_cols)
    cols_per_row = 2
    num_rows = (num_cols_list + cols_per_row - 1) // cols_per_row
    
    for i in range(num_rows):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx < num_cols_list:
                with cols[j]:
                    col = numeric_cols[idx]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.boxplot(df[col], vert=True)
                    ax.set_title(f'{col} Box Plot')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
    
    with st.expander("📖 IQR Method Explained"):
        st.markdown("""
        **How it Works:**
        - Lower Bound = Q1 - 1.5 × IQR
        - Upper Bound = Q3 + 1.5 × IQR
        - Outliers: Values outside these bounds
        
        **Outlier Percentage Guidelines:**
        - < 1%: Normal, no action needed
        - 1-5%: Acceptable level
        - 5-10%: Review data quality
        - > 10%: Potential data quality issue
        """)

# ==================== PAGE 5: GROUP COMPARISONS ====================
elif page == "📋 Group Comparisons (T-Tests)":
    st.header("Independent Samples T-Tests (Welch's)")
    
    # Default binary groups
    binary_groups = [
        ('Gender', 'Total Spend'),
        ('Gender', 'Average Rating'),
        ('Gender', 'Items Purchased'),
        ('Discount Applied', 'Total Spend'),
        ('Discount Applied', 'Average Rating')
    ]
    
    # Filter groups that exist in data
    binary_groups = [(g, v) for g, v in binary_groups 
                     if g in df_processed.columns and v in df.columns]
    
    ttest_results = []
    for group_col, value_col in binary_groups:
        unique_groups = sorted(df_processed[group_col].unique())
        if len(unique_groups) == 2:
            group1_label = unique_groups[0]
            group2_label = unique_groups[1]
            
            group1_data = df_processed[df_processed[group_col] == group1_label][value_col]
            group2_data = df_processed[df_processed[group_col] == group2_label][value_col]
            
            t_stat, p_value = ttest_ind(group1_data, group2_data, equal_var=False)
            
            mean1 = group1_data.mean()
            mean2 = group2_data.mean()
            std1 = group1_data.std()
            std2 = group2_data.std()
            mean_diff = mean1 - mean2
            
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            if abs(cohens_d) > 0.8:
                effect_size = "LARGE"
            elif abs(cohens_d) > 0.5:
                effect_size = "MEDIUM"
            elif abs(cohens_d) > 0.2:
                effect_size = "SMALL"
            else:
                effect_size = "NEGLIGIBLE"
            
            if p_value < 0.001:
                sig = "***HIGHLY SIGNIFICANT***"
            elif p_value < 0.01:
                sig = "**VERY SIGNIFICANT**"
            elif p_value < 0.05:
                sig = "*SIGNIFICANT*"
            else:
                sig = "NOT SIGNIFICANT"
            
            ttest_results.append({
                'Group': group_col,
                'Outcome': value_col,
                f'{group1_label} Mean': f'{mean1:.2f}',
                f'{group2_label} Mean': f'{mean2:.2f}',
                'Difference': f'{mean_diff:.2f}',
                't-statistic': f'{t_stat:.4f}',
                'P-value': f'{p_value:.6f}',
                "Cohen's d": f'{cohens_d:.4f}',
                'Effect Size': effect_size,
                'Significance': sig
            })
    
    ttest_df = pd.DataFrame(ttest_results)
    st.dataframe(ttest_df, use_container_width=True)
    
    st.subheader("Group Comparisons - Violin Plots")
    
    num_tests = len(binary_groups)
    cols_per_row = 2
    num_rows = (num_tests + cols_per_row - 1) // cols_per_row
    
    for i in range(num_rows):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx < num_tests:
                with cols[j]:
                    group_col, value_col = binary_groups[idx]
                    
                    data_for_plot = df_processed[[group_col, value_col]].copy()
                    data_for_plot[group_col] = data_for_plot[group_col].astype(str)
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.violinplot(data=data_for_plot, x=group_col, y=value_col, ax=ax)
                    ax.set_title(f'{group_col} → {value_col}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
    
    with st.expander("📖 Understanding T-Test Results"):
        st.markdown("""
        **P-values:**
        - p < 0.001 (***): Extremely strong evidence of difference
        - p < 0.01 (**): Very strong evidence
        - p < 0.05 (*): Statistically significant difference
        - p > 0.05 (ns): No significant difference
        
        **Cohen's d (Effect Size):**
        - 0.0-0.2: Negligible difference
        - 0.2-0.5: Small difference
        - 0.5-0.8: Medium difference
        - > 0.8: Large difference
        
        **Key Points:**
        - p-value tells if difference exists
        - Cohen's d tells how important the difference is
        - Need BOTH for complete understanding
        """)

# Footer
st.markdown("---")
st.markdown("📊 E-Commerce Customer Behavior Analysis | Week 5 Statistical Analysis")
# 
# st.title("Data200 Project Dashboard")
# 
# st.header("Upload Dataset")
# 
# uploaded_file = st.file_uploader("Choose a CSV file")
# 
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
# 
#     st.subheader("Dataset Preview")
#     st.write(df.head())
# 
#     st.subheader("Basic Statistics")
#     st.write(df.describe())
# 
#     st.subheader("Column Selection for Plot")
#     column = st.selectbox("Select column", df.columns)
# 
#     st.line_chart(df[column])