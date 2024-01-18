import streamlit as st
import pandas as pd
import numpy as np

# Function to get non-NaN flattened data from selected columns or entire dataframe
def get_flattened_non_nan_data(df, selected_columns):
    if "All Columns" in selected_columns or not selected_columns:
        flattened = df.to_numpy().flatten()
    else:
        flattened = df[selected_columns].to_numpy().flatten()
    return flattened[~pd.isnull(flattened)]

# Function to sample rows from a dataframe
def sample_rows(df, num_samples_per_row, num_rows, selected_columns, include_breadcrumbs):
    flattened_non_nan = get_flattened_non_nan_data(df, selected_columns)
    sampled_df = pd.DataFrame(index=range(num_rows), columns=[f'sample_{i+1}' for i in range(num_samples_per_row)])

    # Sample with breadcrumbs if requested
    for i in range(num_rows):
        for j in range(num_samples_per_row):
            if include_breadcrumbs:
                col = np.random.choice(df.columns)
                val = df[col].dropna().sample(n=1).values[0]
                sampled_df.iloc[i, j] = f"{col}: {val}"
            else:
                sampled_df.iloc[i, j] = np.random.choice(flattened_non_nan)

    return sampled_df

# Helper function to pad dataframes with empty values if they have fewer rows
def pad_dataframe(df, target_rows):
    additional_rows = target_rows - len(df)
    if additional_rows > 0:
        padding = pd.DataFrame('', index=range(additional_rows), columns=df.columns)
        df = pd.concat([df, padding], ignore_index=True)
    return df

# Streamlit app layout
st.title('CSV File Sampler')

# Step 1: File Upload
uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True, type='csv')

# This will hold the individual dataframes
dataframes = []
max_num_rows = 0

# Step 2: Process each CSV separately
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"File: {uploaded_file.name}")
        df = pd.read_csv(uploaded_file)
        all_columns_option = ['All Columns']
        column_options = all_columns_option + df.columns.tolist()
        selected_columns = st.multiselect(f"Select columns from {uploaded_file.name} (select 'All Columns' for all)", 
                                         column_options, 
                                         default=all_columns_option, 
                                         key=f'select_{uploaded_file.name}')
        include_breadcrumbs = st.checkbox("Include breadcrumbs", key=f'breadcrumbs_{uploaded_file.name}')
        num_samples_per_row = st.number_input(f"Number of samples per row from {uploaded_file.name}", min_value=1, value=1, key=f'samples_{uploaded_file.name}')
        num_rows = st.number_input(f"Number of rows for {uploaded_file.name}", min_value=1, value=10, key=f'rows_{uploaded_file.name}')
        max_num_rows = max(max_num_rows, num_rows)  # Update max_num_rows

        # Sample the dataframe
        sampled_df = sample_rows(df, num_samples_per_row, num_rows, selected_columns, include_breadcrumbs)
        
        # Display the individual dataframe
        st.write("Sampled Data for this CSV with Breadcrumbs:" if include_breadcrumbs else "Sampled Data for this CSV:")
        st.dataframe(sampled_df)
        
        # Add the individual dataframe to our list
        dataframes.append(sampled_df)

if st.button('Merge DataFrames'):
    combined_df = pd.DataFrame()

    # Concatenate each dataframe with unique column names
    for i, df in enumerate(dataframes):
        df = pad_dataframe(df, max_num_rows)  # Pad the dataframe
        # Rename the columns to ensure they are unique
        df.columns = [f'{df.columns[j]}_{i+1}' for j in range(len(df.columns))]
        combined_df = pd.concat([combined_df, df], axis=1)

    # Display the merged dataframe with padding
    st.write("Combined Sampled Data:")
    st.dataframe(combined_df)
