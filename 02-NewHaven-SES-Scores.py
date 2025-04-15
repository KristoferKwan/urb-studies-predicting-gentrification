#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SES (Socio-Economic Status) Score Calculation for New Haven, CT
This script calculates SES scores for New Haven census tracts, 
similar to the London gentrification model.
Modified to work with the direct Census API data format.
"""

# For reproducibility
import random
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Set random seed for reproducibility
r_state = 42
random.seed(r_state)
np.random.seed(r_state)

# Create directories if they don't exist
data_dir = os.path.join('data', 'newhaven')
src_dir = os.path.join(data_dir, 'src')
canonical_dir = os.path.join(data_dir, 'canonical')
analytical_dir = os.path.join(data_dir, 'analytical')
scores_dir = os.path.join(canonical_dir, 'scores')

for d in [data_dir, src_dir, canonical_dir, analytical_dir, scores_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

def load_census_data():
    """
    Loads the processed census data for New Haven
    """
    try:
        # Try to load the filtered New Haven county data
        df_2020 = pd.read_csv(os.path.join(src_dir, 'newhaven_census_data_2020.csv'))
        df_2010 = pd.read_csv(os.path.join(src_dir, 'newhaven_census_data_2010.csv'))
        
        print(f"Loaded census data: 2020 ({len(df_2020)} tracts), 2010 ({len(df_2010)} tracts)")
        return df_2020, df_2010
    
    except FileNotFoundError:
        print("Error: Census data files not found. Please run the 01-NewHaven-Census-Data-modified.py script first.")
        return None, None

def prepare_ses_variables(df_2020, df_2010):
    """
    Prepares the variables needed for SES score calculation
    
    Following the original model, we use:
    1. Median household income
    2. Median house value
    3. Percentage of residents with higher education (Bachelor's or higher)
    4. Percentage of residents in "favorable" occupations (to be determined based on available data)
    """
    if df_2020 is None or df_2010 is None:
        return None, None
    
    # Process 2020 data
    df_2020_ses = pd.DataFrame()
    
    # Extract tract identifiers - using the NAME column from Census API
    df_2020_ses['tract_id'] = df_2020['NAME'].apply(lambda x: x.split(',')[0].strip())
    
    # Median household income - all values are strings in the Census API output
    df_2020_ses['median_income'] = pd.to_numeric(df_2020['B19013_001E'], errors='coerce')
    
    # Median house value
    df_2020_ses['median_house_value'] = pd.to_numeric(df_2020['B25077_001E'], errors='coerce')
    
    # Calculate percentage with higher education (Bachelor's or higher)
    total_pop = pd.to_numeric(df_2020['B01003_001E'], errors='coerce')
    bachelors = pd.to_numeric(df_2020['B15003_022E'], errors='coerce')
    masters = pd.to_numeric(df_2020['B15003_023E'], errors='coerce')
    professional = pd.to_numeric(df_2020['B15003_024E'], errors='coerce')
    doctorate = pd.to_numeric(df_2020['B15003_025E'], errors='coerce')
    
    df_2020_ses['pct_higher_edu'] = ((bachelors + masters + professional + doctorate) / total_pop) * 100
    
    # We would need occupational data to calculate "favorable" occupations
    # For now, let's use a placeholder (this should be replaced with actual occupation data)
    # Consider adding occupation data in a future version
    df_2020_ses['pct_favorable_occs'] = np.nan
    
    # Process 2010 data (similar to 2020)
    df_2010_ses = pd.DataFrame()
    
    # Extract tract identifiers
    df_2010_ses['tract_id'] = df_2010['NAME'].apply(lambda x: x.split(',')[0].strip())
    
    # Median household income
    df_2010_ses['median_income'] = pd.to_numeric(df_2010['B19013_001E'], errors='coerce')
    
    # Median house value
    df_2010_ses['median_house_value'] = pd.to_numeric(df_2010['B25077_001E'], errors='coerce')
    
    # Calculate percentage with higher education (Bachelor's or higher)
    total_pop_2010 = pd.to_numeric(df_2010['B01003_001E'], errors='coerce')
    
    # The 2010 variable IDs are different - need to sum male and female counts
    try:
        # 2010 has separate male/female variables
        male_bachelors = pd.to_numeric(df_2010['B15002_015E'], errors='coerce')
        male_masters = pd.to_numeric(df_2010['B15002_016E'], errors='coerce')
        male_professional = pd.to_numeric(df_2010['B15002_017E'], errors='coerce')
        male_doctorate = pd.to_numeric(df_2010['B15002_018E'], errors='coerce')
        
        female_bachelors = pd.to_numeric(df_2010['B15002_032E'], errors='coerce')
        female_masters = pd.to_numeric(df_2010['B15002_033E'], errors='coerce')
        female_professional = pd.to_numeric(df_2010['B15002_034E'], errors='coerce')
        female_doctorate = pd.to_numeric(df_2010['B15002_035E'], errors='coerce')
        
        higher_edu_2010 = male_bachelors + male_masters + male_professional + male_doctorate + \
                          female_bachelors + female_masters + female_professional + female_doctorate
    except KeyError:
        # Fallback if the variables don't exist
        print("Warning: Could not find 2010 education variables. Using placeholders.")
        higher_edu_2010 = pd.Series([0] * len(df_2010))
    
    df_2010_ses['pct_higher_edu'] = (higher_edu_2010 / total_pop_2010) * 100
    
    # Placeholder for favorable occupations
    df_2010_ses['pct_favorable_occs'] = np.nan
    
    # Save processed data
    df_2020_ses.to_csv(os.path.join(canonical_dir, 'newhaven_ses_vars_2020.csv'), index=False)
    df_2010_ses.to_csv(os.path.join(canonical_dir, 'newhaven_ses_vars_2010.csv'), index=False)
    
    print(f"Saved processed SES variables to {canonical_dir}")
    
    return df_2020_ses, df_2010_ses

def calculate_ses_scores(df_2020_ses, df_2010_ses):
    """
    Calculates SES scores using PCA, similar to the London model
    """
    if df_2020_ses is None or df_2010_ses is None:
        return None
    
    # Filter to include only tracts that exist in both datasets
    common_tracts = set(df_2020_ses['tract_id']).intersection(set(df_2010_ses['tract_id']))
    print(f"Found {len(common_tracts)} census tracts that exist in both 2010 and 2020 datasets")
    
    # Filter both datasets to only include the common tracts
    df_2010_ses = df_2010_ses[df_2010_ses['tract_id'].isin(common_tracts)].copy()
    df_2020_ses = df_2020_ses[df_2020_ses['tract_id'].isin(common_tracts)].copy()
    
    # Create a combined dataset for PCA
    combined_data = pd.DataFrame()
    
    # Add 2010 data
    df_2010_copy = df_2010_ses.copy()
    df_2010_copy['year'] = 2010
    
    # Add 2020 data
    df_2020_copy = df_2020_ses.copy()
    df_2020_copy['year'] = 2020
    
    combined_data = pd.concat([df_2010_copy, df_2020_copy])
    
    # Extract variables for PCA
    # Note: In a real implementation, you would need to handle missing data appropriately
    pca_vars = ['median_income', 'median_house_value', 'pct_higher_edu']
    
    # Check which variables we can actually use
    available_vars = [var for var in pca_vars if not combined_data[var].isna().all()]
    print(f"Using the following variables for PCA: {available_vars}")
    
    # For this demo, let's continue even if we're missing the occupation data
    pca_data = combined_data[available_vars].copy()
    
    # Scale the data
    pca_data_scaled = scale(pca_data.fillna(pca_data.mean()))
    
    # Run PCA
    pca = PCA(n_components=1)
    scores = pca.fit_transform(pca_data_scaled)
    
    # Print the PCA component weights to understand the SES score formula
    print("PCA component weights (how each variable contributes to the SES score):")
    for var, weight in zip(available_vars, pca.components_[0]):
        print(f"  {var}: {weight:.4f}")
    
    # Add scores back to the dataframe
    combined_data['ses_score'] = scores
    
    # Split back into separate years
    df_2010_ses_scored = combined_data[combined_data['year'] == 2010].copy()
    df_2020_ses_scored = combined_data[combined_data['year'] == 2020].copy()
    
    # Ensure the tracts are in the same order for both datasets
    df_2010_ses_scored = df_2010_ses_scored.sort_values('tract_id').reset_index(drop=True)
    df_2020_ses_scored = df_2020_ses_scored.sort_values('tract_id').reset_index(drop=True)
    
    # Calculate SES ascent (change between 2010 and 2020)
    ses_scores = pd.DataFrame()
    ses_scores['tract_id'] = df_2020_ses_scored['tract_id']
    ses_scores['SES_2010'] = df_2010_ses_scored['ses_score'].values
    ses_scores['SES_2020'] = df_2020_ses_scored['ses_score'].values
    ses_scores['SES_ASCENT'] = ses_scores['SES_2020'] - ses_scores['SES_2010']
    
    # Calculate percentiles
    ses_scores['SES_PR_2010'] = ses_scores['SES_2010'].rank(pct=True) * 100
    ses_scores['SES_PR_2020'] = ses_scores['SES_2020'].rank(pct=True) * 100
    ses_scores['SES_PR_ASCENT'] = ses_scores['SES_ASCENT'].rank(pct=True) * 100
    
    # Save SES scores
    ses_scores.to_csv(os.path.join(scores_dir, 'newhaven_ses_scores.csv'), index=False)
    print(f"Saved SES scores to {os.path.join(scores_dir, 'newhaven_ses_scores.csv')}")
    
    return ses_scores

def plot_ses_changes(ses_scores):
    """
    Plots SES changes to visualize neighborhood change
    """
    if ses_scores is None:
        return
    
    # Create plots directory
    plots_dir = os.path.join(data_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Plot SES scores distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(ses_scores['SES_2010'], color='blue', alpha=0.5, label='2010')
    sns.histplot(ses_scores['SES_2020'], color='red', alpha=0.5, label='2020')
    plt.title('Distribution of SES Scores in New Haven, CT')
    plt.xlabel('SES Score')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'ses_score_distribution.png'))
    
    # Plot SES ascent (change)
    plt.figure(figsize=(12, 8))
    sns.histplot(ses_scores['SES_ASCENT'], color='green')
    plt.title('Distribution of SES Ascent (Change) in New Haven, CT')
    plt.xlabel('SES Ascent')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plots_dir, 'ses_ascent_distribution.png'))
    
    # Plot SES 2010 vs 2020
    plt.figure(figsize=(10, 10))
    plt.scatter(ses_scores['SES_2010'], ses_scores['SES_2020'], alpha=0.7)
    plt.plot([ses_scores['SES_2010'].min(), ses_scores['SES_2010'].max()], 
             [ses_scores['SES_2010'].min(), ses_scores['SES_2010'].max()], 
             'k--', alpha=0.5)
    plt.title('SES Scores: 2010 vs 2020')
    plt.xlabel('SES 2010')
    plt.ylabel('SES 2020')
    plt.axis('equal')
    plt.savefig(os.path.join(plots_dir, 'ses_2010_vs_2020.png'))
    
    print(f"Saved plots to {plots_dir}")

def main():
    """
    Main function to calculate SES scores
    """
    print("Starting SES score calculation for New Haven, CT...")
    
    # Load census data
    df_2020, df_2010 = load_census_data()
    
    if df_2020 is not None and df_2010 is not None:
        # Prepare variables for SES calculation
        df_2020_ses, df_2010_ses = prepare_ses_variables(df_2020, df_2010)
        
        # Calculate SES scores
        ses_scores = calculate_ses_scores(df_2020_ses, df_2010_ses)
        
        # Plot SES changes
        plot_ses_changes(ses_scores)
        
        print("SES score calculation complete!")
        print("\nNext steps:")
        print("1. Run the 03-NewHaven-Prediction-Model.py script to train the gentrification prediction model")
        print("2. For more accurate results, collect additional occupation data for the 'favorable occupations' metric")
    else:
        print("Cannot proceed without census data.")

if __name__ == "__main__":
    main() 