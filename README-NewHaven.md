# New Haven Gentrification Prediction Model - Modified Scripts

This directory contains modified scripts for the New Haven Gentrification Prediction Model. These scripts address issues with the original implementation, particularly with the Census data collection.

## Modifications

1. **Census Data Collection**: We've modified the data collection script to use the direct Census API instead of `cenpy`, which had issues with accessing certain URLs.

2. **SES Score Calculation**: Updated to work with the data format from the direct Census API approach.

3. **Package Dependencies**: Added support for proper installation in a new Python environment.

## Prerequisites

### Python Environment Setup

This project requires Python 3.7+ and several packages. Use pyenv to create and manage a separate environment:

```bash
# Create a new pyenv environment
pyenv virtualenv 3.11.6 newhavenmodel

# Activate the environment
pyenv activate newhavenmodel

# Install requirements
pip install -r requirements.txt

# Install additional dependency to fix warning
pip install python-Levenshtein
```

### Census API Key

You'll need to register for a free API key from the US Census Bureau:

1. Go to https://api.census.gov/data/key_signup.html
2. Fill out the form and submit it
3. Check your email for the API key
4. Add your key to the `01-NewHaven-Census-Data-modified.py` script

## Running the Analysis

Execute the following scripts in sequence:

### 1. Data Collection

```bash
python 01-NewHaven-Census-Data-modified.py
```

This script:

- Downloads census data from the 2010 and 2020 ACS 5-year estimates
- Retrieves census tract boundary files
- Filters data to New Haven city census tracts
- Saves all data to the `data/newhaven` directory structure

### 2. SES Score Calculation

```bash
python 02-NewHaven-SES-Scores-modified.py
```

This script:

- Calculates SES (Socio-Economic Status) scores for each tract
- Computes SES changes between 2010 and 2020
- Creates visualizations of SES score distributions
- Saves results to the `data/newhaven/canonical/scores` directory

### 3. Prediction Model

The original `03-NewHaven-Prediction-Model.py` script should work with the outputs from the modified scripts, but you may need to check paths and data formats if you encounter any issues.

## Directory Structure

The scripts create and use the following directory structure:

```
data/
  newhaven/
    src/                    # Raw data from Census API
    canonical/              # Processed data files
      scores/               # SES scores
    analytical/             # Data ready for modeling
    model/                  # Trained model and feature importance
    plots/                  # Visualizations
```

## Troubleshooting

If you encounter any issues:

1. **Census API Connection**: Make sure your Census API key is valid and correctly inserted in the script.

2. **Data Structure**: Check the data formats in each step to ensure they match the expected formats.

3. **Missing Data**: The scripts include handling for missing data, but you may need to adjust parameters for your specific case.

4. **Spatial Analysis**: For more accurate city filtering, consider implementing a spatial join with the city boundary instead of the simple text filter.

## Future Improvements

1. Add collection of occupational data to better calculate the "favorable occupations" metric

2. Implement spatial filtering of tracts using actual city boundaries

3. Add more census variables to improve model accuracy

4. Create interactive maps of gentrification risk
