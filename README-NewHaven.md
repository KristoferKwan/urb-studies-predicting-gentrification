# Predicting Neighborhood Change in New Haven, CT

This project adapts the machine learning approach from the Urban Studies paper ["Understanding urban gentrification through Machine Learning: Predicting neighbourhood change in London"](https://doi.org/10.1177/0042098018789054) to predict gentrification patterns in New Haven, Connecticut using census tract-level data.

## Project Overview

This adaptation uses US Census data to build a model that can predict which neighborhoods in New Haven are at risk of gentrification. The approach follows these main steps:

1. **Data Collection**: Gather census data for New Haven at the tract level for two time periods (2010 and 2020)
2. **SES Score Calculation**: Calculate a socioeconomic status (SES) score for each census tract using key variables:
   - Median household income
   - Median house value
   - Percentage of residents with higher education
   - Percentage of residents in professional occupations
3. **Feature Engineering**: Process and transform additional census variables as features for the model
4. **Model Training**: Use an Extra Trees Regressor to predict SES score changes based on neighborhood characteristics
5. **Visualization**: Create maps and charts to visualize results and identify at-risk neighborhoods

## Setup Instructions

### Prerequisites

You need Python 3.7+ and the following packages:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
geopandas
requests
census
us
cenpy
ipykernel
jupyter
```

Install these packages using:

```bash
pip install -r requirements.txt
```

### Getting a Census API Key

To download US Census data, you'll need to register for a free API key:

1. Go to https://api.census.gov/data/key_signup.html
2. Fill out the form to request an API key
3. Check your email for the API key
4. Add your key to the `01-NewHaven-Census-Data.py` script

## Running the Analysis

Execute the following scripts in sequence:

1. **Data Collection**:

   ```bash
   python 01-NewHaven-Census-Data.py
   ```

   This downloads census data for New Haven at the tract level.

2. **SES Score Calculation**:

   ```bash
   python 02-NewHaven-SES-Scores.py
   ```

   This calculates SES scores for each census tract.

3. **Model Training and Prediction**:
   ```bash
   python 03-NewHaven-Prediction-Model.py
   ```
   This trains and evaluates the prediction model.

## Data Structure

The project creates the following directory structure:

```
data/
  newhaven/
    src/                    # Raw data from Census API
    canonical/              # Processed data files
    analytical/             # Data ready for modeling
    model/                  # Trained model and feature importance
    plots/                  # Visualizations
```

## Model Details

The model uses an **Extra Trees Regressor** (a variant of Random Forest) to predict neighborhood change. Features include demographic, housing, economic, and social indicators derived from census data.

The key metrics for evaluating model performance include:

- R² (coefficient of determination)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Explained variance

## Understanding Gentrification Risk

The model outputs a predicted SES change score for each census tract. Higher positive values indicate higher likelihood of gentrification. Tracts can be categorized as:

- **High Risk**: Tracts with high predicted SES change (top 25%)
- **Moderate Risk**: Tracts with moderate predicted SES change (25-75%)
- **Low Risk**: Tracts with low predicted SES change (bottom 25%)

## Customizing for Other Cities

To adapt this model for other cities:

1. Change the FIPS codes in the data collection script
2. Adjust the filtering criteria to match your target city's tract names/codes
3. Run the pipeline as described above

## Acknowledgments

This project adapts the methodology developed by the original researchers:

- Jon Reades
- Jordan De Souza
- Phil Hubbard

If you use this adaptation in your work, please cite the original Urban Studies paper.
