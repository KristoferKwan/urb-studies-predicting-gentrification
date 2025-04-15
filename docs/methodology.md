---
layout: default
title: Methodology
---

# Methodology

## Data Sources

Our analysis relies on the following data sources:

- Census data from 2001 and 2011
- Socio-economic indicators
- Geographic boundary data for neighborhood units

## Data Processing

The data processing pipeline includes several key steps:

1. **Geographic Conversion**: Converting data between different boundary systems using the UK Data Service's Geo-Convert tool
2. **Feature Engineering**: Creating relevant socio-economic indicators
3. **Data Standardization**: Normalizing variables for consistent scale

## Machine Learning Approach

We employed several machine learning algorithms to predict neighborhood change:

- Random Forest
- Support Vector Machines
- Gradient Boosting
- Neural Networks

Each model was evaluated using cross-validation to ensure robust results. Hyperparameter optimization was conducted to find the optimal model configuration.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision and Recall
- F1 Score
- ROC AUC

## Visualization

Results were visualized using:

- QGIS for geospatial mapping
- R for statistical charts and graphs

[Return to Home](index.html)
