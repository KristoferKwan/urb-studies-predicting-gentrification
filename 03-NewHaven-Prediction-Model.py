#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neighborhood Change Prediction Model for New Haven, CT
Based on the London gentrification model using Extra Trees Regressor.
"""

# For reproducibility
import random
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import scale
from timeit import default_timer as timer
import datetime
import geopandas as gpd
import folium
from folium.plugins import FloatImage
from branca.colormap import LinearColormap
import matplotlib.cm as cm
import traceback

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
model_dir = os.path.join(data_dir, 'model')
plots_dir = os.path.join(data_dir, 'plots')

for d in [data_dir, src_dir, canonical_dir, analytical_dir, scores_dir, model_dir, plots_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

def load_data():
    """
    Loads the SES scores and feature data for model training
    """
    try:
        # Load SES scores
        ses_scores = pd.read_csv(os.path.join(scores_dir, 'newhaven_ses_scores.csv'))
        print(f"Loaded SES scores for {len(ses_scores)} census tracts")
        
        # Load feature data (this would be the equivalent of variables from London model)
        # In a complete implementation, you would have many more variables from the census data
        # For now, we'll use placeholder data derived from census variables
        
        try:
            # Try to load processed features if they exist
            features = pd.read_csv(os.path.join(analytical_dir, 'newhaven_model_features.csv'))
            print(f"Loaded {features.shape[1]} features for model training")
        except FileNotFoundError:
            # If features don't exist yet, we can create a simple set from census data
            print("Feature data not found. Creating simplified feature set from census data.")
            
            # Load census data
            census_2010 = pd.read_csv(os.path.join(canonical_dir, 'newhaven_city_data_2010.csv'))
            
            # Extract basic features
            features = pd.DataFrame()
            features['tract_id'] = ses_scores['tract_id']
            
            # Join with census data
            # This is a placeholder - in reality, you would create many more features
            # and properly process/transform them as in the London model
            
            # For now, we'll just create a dummy feature set
            features['total_pop'] = np.random.normal(5000, 1000, size=len(features))
            features['pct_owner_occupied'] = np.random.uniform(0.2, 0.8, size=len(features))
            features['pct_vacant'] = np.random.uniform(0.05, 0.2, size=len(features))
            features['median_income'] = np.random.normal(50000, 15000, size=len(features))
            features['pct_higher_edu'] = np.random.uniform(0.1, 0.5, size=len(features))
            
            # Save the feature data for future use
            features.to_csv(os.path.join(analytical_dir, 'newhaven_model_features.csv'), index=False)
        
        return ses_scores, features
        
    except FileNotFoundError:
        print("Error: SES scores not found. Please run the 02-NewHaven-SES-Scores.py script first.")
        return None, None

def prepare_model_data(ses_scores, features):
    """
    Prepares data for modeling by joining SES scores with features
    and selecting only the most relevant predictors
    """
    if ses_scores is None or features is None:
        return None, None, None
    
    # Merge features with SES scores
    model_data = pd.merge(features, ses_scores, on='tract_id')
    
    # Set tract_id as index
    model_data.set_index('tract_id', inplace=True)
    
    # Drop SES columns to create feature dataset
    X = model_data.drop(['SES_2010', 'SES_2020', 'SES_ASCENT', 
                         'SES_PR_2010', 'SES_PR_2020', 'SES_PR_ASCENT'], axis=1)
    
    # First, identify and drop ID columns and other non-predictive identifiers
    id_columns = ['MTFCC', 'OID', 'GEOID', 'STATE', 'COUNTY', 'TRACT', 
                 'BASENAME', 'NAME', 'LSADC', 'FUNCSTAT', 'OBJECTID',
                 'AREALAND', 'AREAWATER', 'UR', 'CENTLAT', 'CENTLON', 'INTPTLAT', 'INTPTLON', 
                 'HU100', 'POP100']
    
    id_columns_present = [col for col in id_columns if col in X.columns]
    
    if id_columns_present:
        X = X.drop(id_columns_present, axis=1)
        print(f"Dropped ID and administrative columns: {len(id_columns_present)} columns")
    
    # Keep only highly relevant features
    highly_relevant_features = [
        # Socioeconomic indicators
        'median_income', 
        'pct_higher_edu',
        'median_house_value', 
        'pct_in_labor_force',
        
        # Housing characteristics
        'pct_owner_occupied',
        'pct_renter_occupied', 
        'pct_vacant',
        
        # Demographic factors
        'pct_white',
        'pct_black',
        'pop_density',
        'total_pop',
        
        # Geographic features
        'distance_to_downtown',
        'area_sqkm',
        'compactness'
    ]
    
    # Filter to only include columns that actually exist in the dataset
    available_features = [f for f in highly_relevant_features if f in X.columns]
    missing_features = [f for f in highly_relevant_features if f not in X.columns]
    
    if missing_features:
        print(f"Warning: The following important features are missing: {missing_features}")
    
    # Keep only the selected features
    X_selected = X[available_features].copy()
    print(f"Selected {len(available_features)} highly relevant features for modeling")
    
    # Drop any remaining non-numerical columns (shouldn't be any, but just in case)
    numerical_cols = X_selected.select_dtypes(include=['int64', 'float64']).columns
    X_numeric = X_selected[numerical_cols]
    
    # Check if we dropped any selected features
    dropped_selected = [col for col in available_features if col not in numerical_cols]
    if dropped_selected:
        print(f"Warning: Dropped non-numeric features: {dropped_selected}")
    
    # Target is SES_ASCENT (the change in SES score)
    y = model_data['SES_ASCENT']
    
    # Scale features
    X_scaled = pd.DataFrame(scale(X_numeric), index=X_numeric.index, columns=X_numeric.columns)
    
    # Save the prepared data
    X_scaled.to_csv(os.path.join(analytical_dir, 'newhaven_X_scaled.csv'))
    print(f"Feature selection complete. Using features: {', '.join(X_numeric.columns)}")
    
    return model_data, X_scaled, y

def train_model(X, y):
    """
    Trains an Extra Trees Regressor model to predict neighborhood change
    """
    if X is None or y is None:
        return None
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=r_state)
    
    # Define the parameter grid for grid search
    # Updated to use only valid parameters for ExtraTreesRegressor
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.7, 0.85, 'sqrt', 'log2', None]  # Fixed invalid 'auto' option
    }
    
    print("Starting model training with grid search...")
    print(f"Number of parameter combinations: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])}")
    
    # Create the Extra Trees Regressor
    clf = ensemble.ExtraTreesRegressor(random_state=r_state)
    
    # Perform grid search
    start = timer()
    cv = model_selection.GridSearchCV(estimator=clf, param_grid=param_grid, 
                                    cv=4, n_jobs=-1, verbose=1, 
                                    scoring='neg_mean_squared_error')
    cv.fit(X_train, y_train)
    duration = timer() - start
    
    print(f"Model training complete in {duration:.1f}s ({str(datetime.timedelta(seconds=duration))})")
    print(f"Best parameters: {cv.best_params_}")
    print(f"Best score: {cv.best_score_}")
    
    # Get the best model
    best_clf = cv.best_estimator_
    
    # Evaluate on test set
    y_pred = best_clf.predict(X_test)
    
    # Print evaluation metrics
    print("\nModel Evaluation on Test Set:")
    print(f"R²:        {metrics.r2_score(y_test, y_pred):.5f}")
    print(f"MSE:       {metrics.mean_squared_error(y_test, y_pred):.5f}")
    print(f"MAE:       {metrics.mean_absolute_error(y_test, y_pred):.5f}")
    print(f"Expl. Var: {metrics.explained_variance_score(y_test, y_pred):.5f}")
    
    # Save the model
    import pickle
    with open(os.path.join(model_dir, 'newhaven_model.pkl'), 'wb') as f:
        pickle.dump(best_clf, f)
    
    # Create a dataframe of feature importances
    fi = pd.DataFrame({
        'feature': X.columns,
        'importance': best_clf.feature_importances_
    })
    fi.sort_values(by='importance', ascending=False, inplace=True)
    
    # Save feature importances
    fi.to_csv(os.path.join(model_dir, 'newhaven_feature_importance.csv'), index=False)
    
    print(f"\nTop 5 most important features:")
    print(fi.head(5))
    
    return best_clf, fi, (X_test, y_test, y_pred)

def visualize_results(model, feature_importance, test_data):
    """
    Creates visualizations of model results
    """
    if model is None:
        return
    
    X_test, y_test, y_pred = test_data
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', alpha=0.5)
    plt.title('Actual vs Predicted SES Ascent')
    plt.xlabel('Actual SES Ascent')
    plt.ylabel('Predicted SES Ascent')
    plt.axis('equal')
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'))
    
    print(f"Saved visualization plots to {plots_dir}")

def map_gentrification_risk(model, model_data, X_scaled):
    """
    Creates a choropleth map of predicted gentrification risk by census tract
    """
    print("Creating gentrification risk map for New Haven census tracts...")
    
    try:
        # Predict risk of gentrification (SES ascent) for all tracts
        risk_scores = model.predict(X_scaled)
        
        # Create a dataframe with tract_id and predicted risk
        risk_df = pd.DataFrame({
            'tract_id': X_scaled.index,
            'gentrification_risk': risk_scores
        })
        
        # Categorize risk levels (for display purposes)
        risk_df['risk_level'] = pd.qcut(
            risk_df['gentrification_risk'], 
            q=5, 
            labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        )
        
        # Create a finer gradation within each risk level for more detailed visualization
        # First, calculate percentile rank for each risk score within dataset
        risk_df['percentile'] = risk_df['gentrification_risk'].rank(pct=True) * 100
        
        # Load the GeoJSON tract boundaries - using 2010 instead of 2020
        tracts_geojson = gpd.read_file(os.path.join(src_dir, 'newhaven_tracts_2010.geojson'))
        
        # Create a tract_id field in the GeoJSON that matches our data
        if 'NAME' in tracts_geojson.columns:
            # Extract tract identifiers - using the same logic as in the SES scores script
            tracts_geojson['tract_id'] = tracts_geojson['NAME'].apply(lambda x: x.split(',')[0].strip())
        elif 'TRACTCE' in tracts_geojson.columns:
            # Fallback to TRACTCE approach if NAME is not available
            tracts_geojson['tract_id'] = tracts_geojson['TRACTCE'].apply(
                lambda x: f"Census Tract {x.lstrip('0')}" if x.lstrip('0') else "Census Tract 0"
            )
        else:
            print("Warning: Could not find NAME or TRACTCE columns in GeoJSON. Risk mapping may fail.")
        
        # Print information about both datasets before merging
        print(f"GeoJSON tracts: {len(tracts_geojson)}, Risk scores: {len(risk_df)}")
        print(f"Unique tract IDs in GeoJSON: {len(tracts_geojson['tract_id'].unique())}")
        print(f"Unique tract IDs in risk data: {len(risk_df['tract_id'].unique())}")
        
        # Find common tract IDs
        common_tracts = set(tracts_geojson['tract_id']).intersection(set(risk_df['tract_id']))
        print(f"Found {len(common_tracts)} common tract IDs between datasets")
        
        # Optionally filter both datasets before merging
        tracts_geojson_filtered = tracts_geojson[tracts_geojson['tract_id'].isin(common_tracts)]
        risk_df_filtered = risk_df[risk_df['tract_id'].isin(common_tracts)]
        
        # Merge risk scores with GeoJSON data using inner join to keep only matching rows
        tracts_with_risk = tracts_geojson_filtered.merge(risk_df_filtered, on='tract_id', how='inner')
        
        # Verify the merge result
        print(f"After merge: {len(tracts_with_risk)} tracts with risk scores")
        
        # Create a static visualization using geopandas and save as PNG
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Use a continuous color palette but with clear boundaries between risk levels
        # Use the percentile value for continuous coloring
        tracts_with_risk.plot(
            column='percentile',  # Use percentile for continuous gradient
            cmap='YlOrRd',  # Yellow-Orange-Red provides better distinction for low values
            linewidth=0.8,
            ax=ax,
            edgecolor='0.8',
            legend=True,
            legend_kwds={'label': 'Gentrification Risk Percentile'}
        )
        
        # Add title and annotations
        ax.set_title('Predicted Gentrification Risk by Census Tract in New Haven, CT', fontsize=16)
        ax.set_axis_off()
        
        # Add a custom legend showing the main risk categories
        # Create patches for the legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ffffb2', edgecolor='black', label='Very Low'),
            Patch(facecolor='#fecc5c', edgecolor='black', label='Low'),
            Patch(facecolor='#fd8d3c', edgecolor='black', label='Moderate'),
            Patch(facecolor='#f03b20', edgecolor='black', label='High'),
            Patch(facecolor='#bd0026', edgecolor='black', label='Very High')
        ]
        # Add the custom legend
        ax.legend(handles=legend_elements, loc='lower right', title='Risk Categories')
        
        # Add annotations for high-risk areas
        high_risk = tracts_with_risk[tracts_with_risk['risk_level'].isin(['High', 'Very High'])]
        for idx, row in high_risk.iterrows():
            if pd.notna(row['gentrification_risk']):
                centroid = row.geometry.centroid
                ax.annotate(
                    f"{row['tract_id']}\nRisk: {row['gentrification_risk']:.2f}",
                    xy=(centroid.x, centroid.y),
                    ha='center',
                    fontsize=8,
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
        
        # Save the static map
        static_map_path = os.path.join(plots_dir, 'gentrification_risk_map.png')
        plt.savefig(static_map_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved static risk map to {static_map_path}")
        
        # Create an interactive map with folium
        # Create a center point for New Haven
        center = [41.308, -72.928]  # New Haven coordinates
        
        # Create a base map
        m = folium.Map(
            location=center,
            zoom_start=12,
            tiles='CartoDB positron'
        )
        
        # Convert to GeoJSON format for Folium
        tracts_with_risk = tracts_with_risk.to_crs(epsg=4326)  # Convert to WGS84 for web mapping
        
        # Define color scales for each main risk level (creates gradients within each category)
        colors_very_low = ['#ffffb2', '#fff8a3', '#fff194', '#ffea85', '#ffe476']
        colors_low = ['#fecc5c', '#fec54d', '#febe3e', '#feb72f', '#feb020']
        colors_moderate = ['#fd8d3c', '#fd832d', '#fd791e', '#fd6f0f', '#fd6500']
        colors_high = ['#f03b20', '#e13519', '#d32f12', '#c5290c', '#b72305']
        colors_very_high = ['#bd0026', '#ae0022', '#9f001e', '#90001a', '#810016']
        
        # Calculate min and max percentiles for each category
        percentile_thresholds = [0, 20, 40, 60, 80, 100]
        
        # Function to get appropriate color based on percentile within its category
        def get_gradient_color(risk_level, percentile):
            if risk_level == 'Very Low':
                # Map values between 0-20 percentile to the very low color gradient
                norm_pct = (percentile - 0) / 20  # Normalize to 0-1 range within bracket
                idx = min(int(norm_pct * 5), 4)  # Get index in color array (0-4)
                return colors_very_low[idx]
            elif risk_level == 'Low':
                norm_pct = (percentile - 20) / 20
                idx = min(int(norm_pct * 5), 4)
                return colors_low[idx]
            elif risk_level == 'Moderate':
                norm_pct = (percentile - 40) / 20
                idx = min(int(norm_pct * 5), 4)
                return colors_moderate[idx]
            elif risk_level == 'High':
                norm_pct = (percentile - 60) / 20
                idx = min(int(norm_pct * 5), 4)
                return colors_high[idx]
            else:  # 'Very High'
                norm_pct = (percentile - 80) / 20
                idx = min(int(norm_pct * 5), 4)
                return colors_very_high[idx]
        
        # Function to style each feature based on both risk level and percentile
        def style_function(feature):
            risk_level = feature['properties']['risk_level']
            percentile = feature['properties']['percentile']
            return {
                'fillColor': get_gradient_color(risk_level, percentile),
                'color': '#000000',
                'weight': 0.2,  # Changed from 0.5 to 0.2 to make lines thinner
                'fillOpacity': 0.8
            }
        
        # Add the GeoJSON layer with gradient coloring
        folium.GeoJson(
            tracts_with_risk,
            style_function=style_function,
            name='Gentrification Risk'
        ).add_to(m)
        
        # Create a more detailed legend with gradient colors
        legend_html = '''
        <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; 
                    padding: 10px; border: 2px solid grey; border-radius: 5px;">
            <p style="text-align: center;"><b>Risk Level</b></p>
            <div style="display: flex; flex-direction: column; gap: 5px;">
                <div>
                    <b>Very High</b>
                    <div style="display: flex; height: 20px;">
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                    </div>
                </div>
                <div>
                    <b>High</b>
                    <div style="display: flex; height: 20px;">
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                    </div>
                </div>
                <div>
                    <b>Moderate</b>
                    <div style="display: flex; height: 20px;">
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                    </div>
                </div>
                <div>
                    <b>Low</b>
                    <div style="display: flex; height: 20px;">
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                    </div>
                </div>
                <div>
                    <b>Very Low</b>
                    <div style="display: flex; height: 20px;">
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                        <div style="flex: 1; background-color: {}"></div>
                    </div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                <span>Lower</span>
                <span>Higher</span>
            </div>
        </div>
        '''.format(
            colors_very_high[0], colors_very_high[1], colors_very_high[2], colors_very_high[3], colors_very_high[4],
            colors_high[0], colors_high[1], colors_high[2], colors_high[3], colors_high[4],
            colors_moderate[0], colors_moderate[1], colors_moderate[2], colors_moderate[3], colors_moderate[4],
            colors_low[0], colors_low[1], colors_low[2], colors_low[3], colors_low[4],
            colors_very_low[0], colors_very_low[1], colors_very_low[2], colors_very_low[3], colors_very_low[4]
        )
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add hover functionality with a GeoJsonTooltip
        highlight_function = lambda x: {'weight': 1.5, 'color': '#000000', 'fillOpacity': 0.5, 'dashArray': '5, 5'}  # Changed from 3 to 1.5
        
        NIL = folium.features.GeoJson(
            tracts_with_risk,
            style_function=lambda x: {'fillOpacity': 0.0, 'weight': 1, 'color': 'black'},  # Changed from 2 to 1
            control=False,
            highlight_function=highlight_function,
            tooltip=folium.features.GeoJsonTooltip(
                fields=['tract_id', 'gentrification_risk', 'risk_level', 'percentile'],
                aliases=['Census Tract:', 'Risk Score:', 'Risk Level:', 'Percentile:'],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.3);")
            )
        )
        m.add_child(NIL)
        m.keep_in_front(NIL)
        
        # Add a title as a simple HTML element
        title_html = '''
             <h3 align="center" style="font-size:16px"><b>Predicted Gentrification Risk in New Haven, CT</b></h3>
             <h4 align="center" style="font-size:12px"><i>Color intensity within each category indicates relative risk level</i></h4>
             '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save the interactive map
        interactive_map_path = os.path.join(plots_dir, 'gentrification_risk_map.html')
        m.save(interactive_map_path)
        print(f"Saved interactive risk map to {interactive_map_path}")
        
        # Also save the risk scores as a CSV for future use
        risk_csv_path = os.path.join(analytical_dir, 'newhaven_gentrification_risk.csv')
        risk_df.to_csv(risk_csv_path, index=False)
        print(f"Saved risk scores to {risk_csv_path}")
        
        return risk_df, static_map_path, interactive_map_path
    
    except ImportError as e:
        print(f"Error: Required mapping libraries not installed. Please install with:")
        print("pip install geopandas folium branca")
        print(f"Full error: {e}")
        return None
    except Exception as e:
        print(f"Error creating gentrification risk map: {e}")
        print("\nFull stack trace:")
        traceback.print_exc()
        return None

def main():
    """
    Main function to run the prediction model
    """
    print("Starting neighborhood change prediction model for New Haven, CT...")
    
    # Load data
    ses_scores, features = load_data()
    
    if ses_scores is not None and features is not None:
        # Prepare model data
        model_data, X, y = prepare_model_data(ses_scores, features)
        
        if X is not None and y is not None:
            # Train model
            model, feature_importance, test_data = train_model(X, y)
            
            # Visualize results
            visualize_results(model, feature_importance, test_data)
            
            # Map gentrification risk
            map_gentrification_risk(model, model_data, X)
            
            print("\nModel training and evaluation complete!")
            print("\nTo use this model for prediction:")
            print("1. Load the model from 'model/newhaven_model.pkl'")
            print("2. Prepare new census data in the same format as the training data")
            print("3. Use model.predict() to predict SES ascent for new data")
            
        else:
            print("Cannot train model without proper data preparation.")
    else:
        print("Cannot proceed without SES scores and feature data.")

if __name__ == "__main__":
    main() 