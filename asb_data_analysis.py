import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind

# Ensure UTF-8 encoding for output to prevent Unicode errors
sys.stdout.reconfigure(encoding='utf-8')

# Load dataset
file_path = "MPS_Antisocial_Behaviour.csv"

# Handling dtype warning by setting low_memory=False
df = pd.read_csv(file_path, low_memory=False)

# Display missing values before cleaning
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fix column data types if needed (Example: Convert mixed types)
if 'Response_Time' in df.columns:
    df['Response_Time'] = pd.to_numeric(df['Response_Time'], errors='coerce')

# ✅ Correct the 'Hour' column (extract hour from HH:MM format)
if 'Hour' in df.columns:
    df['Hour'] = pd.to_datetime(df['Hour'], format='%H:%M', errors='coerce').dt.hour

# Fill missing values for numerical columns
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill missing values for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# Drop rows where more than 50% of values are missing
threshold = len(df.columns) // 2
df.dropna(thresh=threshold, inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Display missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Check if dataset is empty
if df.empty:
    print("⚠️ WARNING: Dataset is empty after cleaning. Please check data source.")
else:
    print(f"\n✅ Data cleaning complete. Rows remaining: {len(df)}")

# Save cleaned data to a new CSV file
df.to_csv("cleaned_ASB_data.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_ASB_data.csv'")

# --- Data Visualization ---

# Ensure 'Opening_Type_1' column exists before plotting
if 'Opening_Type_1' in df.columns:
    plt.figure(figsize=(12,6))
    sns.countplot(y=df['Opening_Type_1'], order=df['Opening_Type_1'].value_counts().index, hue=df['Opening_Type_1'], palette="coolwarm", legend=False)
    plt.title("Most Frequent ASB Types")
    plt.xlabel("Count")
    plt.ylabel("ASB Type")
    plt.show()
else:
    print("⚠️ 'Opening_Type_1' column not found. Skipping ASB type visualization.")

# Ensure 'Hour' column exists and is numeric before plotting
if 'Hour' in df.columns and pd.api.types.is_numeric_dtype(df['Hour']):
    plt.figure(figsize=(12,6))
    sns.histplot(df['Hour'], bins=24, kde=True, color="blue")
    plt.title("ASB Incidents by Hour")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Incident Count")
    plt.xticks(range(0, 24))  # Ensure x-axis shows hours correctly
    plt.show()
else:
    print("⚠️ 'Hour' column not found or not numeric. Skipping ASB incidents by hour plot.")

# ✅ Borough-wise ASB visualization
if 'Safer_Neighborhood_Team_Borough_Name' in df.columns:
    plt.figure(figsize=(14,7))
    sns.countplot(y=df['Safer_Neighborhood_Team_Borough_Name'], order=df['Safer_Neighborhood_Team_Borough_Name'].value_counts().index, palette="viridis")
    plt.title("ASB Incidents by Borough")
    plt.xlabel("Count")
    plt.ylabel("Borough")
    plt.show()
else:
    print("⚠️ 'Safer_Neighborhood_Team_Borough_Name' column not found. Skipping borough-wise ASB visualization.")

# --- Hypothesis Testing ---

# ✅ 1️⃣ Chi-Square Test for ASB Incidents Across Boroughs
print("\n📊 Performing Chi-Square Test for ASB Distribution Across Boroughs...\n")
if 'Safer_Neighborhood_Team_Borough_Name' in df.columns:
    # Create a frequency table of ASB incidents per borough
    contingency_table = df['Safer_Neighborhood_Team_Borough_Name'].value_counts().to_frame()

    # Perform the Chi-Square Test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    # Print results
    print(f"Chi-Square Statistic: {chi2_stat}")
    print(f"P-value: {p_value}")
    print(f"Degrees of Freedom: {dof}")

    # Decision Rule
    if p_value < 0.05:
        print("❌ Reject H₀: ASB incidents are not uniformly distributed across boroughs.")
    else:
        print("✅ Fail to Reject H₀: ASB incidents are uniformly distributed across boroughs.")
else:
    print("⚠️ 'Safer_Neighborhood_Team_Borough_Name' column not found. Chi-Square test skipped.")

# ✅ 2️⃣ T-Test: Comparing High vs. Low ASB Boroughs
print("\n📊 Performing T-Test for ASB Incidents Between High and Low Boroughs...\n")
if 'Safer_Neighborhood_Team_Borough_Name' in df.columns:
    # Count incidents per borough
    borough_counts = df['Safer_Neighborhood_Team_Borough_Name'].value_counts()

    # Define top 10 and bottom 10 boroughs based on ASB incident counts
    top_boroughs = borough_counts.nlargest(10).index
    bottom_boroughs = borough_counts.nsmallest(10).index

    # Filter dataset for the selected boroughs
    top_asb = df[df['Safer_Neighborhood_Team_Borough_Name'].isin(top_boroughs)]['Safer_Neighborhood_Team_Borough_Name'].value_counts()
    bottom_asb = df[df['Safer_Neighborhood_Team_Borough_Name'].isin(bottom_boroughs)]['Safer_Neighborhood_Team_Borough_Name'].value_counts()

    # Perform T-Test
    t_stat, p_val = ttest_ind(top_asb, bottom_asb, equal_var=False)

    # Print results
    print(f"T-Statistic: {t_stat}")
    print(f"P-value: {p_val}")

    # Decision Rule
    if p_val < 0.05:
        print("❌ Reject H₀: High ASB boroughs have significantly more incidents than low ASB boroughs.")
    else:
        print("✅ Fail to Reject H₀: No significant difference between high and low ASB boroughs.")
else:
    print("⚠️ 'Safer_Neighborhood_Team_Borough_Name' column not found. T-Test skipped.")

