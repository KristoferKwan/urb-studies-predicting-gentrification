#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Census Data Collection for New Haven, CT
This script collects census data for New Haven census tracts from the US Census API.
Modified to avoid cenpy FIPS loading issues.
"""

# For reproducibility
import random
import numpy as np
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from census import Census
from us import states
import json
import zipfile
import io

# Set random seed for reproducibility
r_state = 42
random.seed(r_state)
np.random.seed(r_state)

# Create directories if they don't exist
data_dir = os.path.join('data', 'newhaven')
src_dir = os.path.join(data_dir, 'src')
canonical_dir = os.path.join(data_dir, 'canonical')
analytical_dir = os.path.join(data_dir, 'analytical')

for d in [data_dir, src_dir, canonical_dir, analytical_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# Define New Haven County FIPS code (New Haven is in New Haven County, CT)
# CT FIPS = 09, New Haven County = 009
ct_fips = '09'
newhaven_county_fips = '009'

# To use the Census API, you need an API key from https://api.census.gov/data/key_signup.html
# Use the API key that was already in the original script
api_key = '8b222a8cfc46d9eb6687aa247c20c9849ce6dcc7'

def download_census_data_direct():
    """
    Downloads census data for New Haven census tracts directly using Census API
    """
    print("Downloading census data for New Haven, CT using direct Census API...")
    
    # Use the Census API directly
    census = Census(api_key)
    
    # Define variables to collect (similar to the original list)
    variables = [
        'NAME',
        'B19013_001E',  # Median household income
        'B25077_001E',  # Median house value
        'B15003_022E',  # Bachelor's degree
        'B15003_023E',  # Master's degree
        'B15003_024E',  # Professional degree
        'B15003_025E',  # Doctorate degree
        'B23025_001E',  # Total labor force population 16+ years
        'B23025_002E',  # In labor force
        'B01003_001E',  # Total population
        'B25002_001E',  # Housing units
        'B25002_002E',  # Occupied housing units
        'B25002_003E',  # Vacant housing units
        'B25003_001E',  # Tenure
        'B25003_002E',  # Owner occupied
        'B25003_003E',  # Renter occupied
        'B02001_002E',  # White alone
        'B02001_003E'   # Black or African American alone
    ]
    
    # Query ACS5 2020 data for New Haven County
    newhaven_data = census.acs5.state_county_tract(
        fields=variables,
        state_fips=ct_fips,
        county_fips=newhaven_county_fips,
        tract='*',
        year=2020
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(newhaven_data)
    
    # Replace Census placeholder for missing values with NaN
    df = df.replace(-666666666, np.nan)
    df = df.replace(-666666666.0, np.nan)
    
    # Save raw data
    df.to_csv(os.path.join(src_dir, 'newhaven_census_data_2020.csv'), index=False)
    print(f"Saved 2020 census data to {os.path.join(src_dir, 'newhaven_census_data_2020.csv')}")
    
    return df

def download_census_2010_data_direct():
    """
    Downloads 2010 census data directly using Census API
    """
    print("Downloading 2010 census data for New Haven, CT using direct Census API...")
    
    # Use the Census API directly
    census = Census(api_key)
    
    # Labor force component variables from B23001 table
    labor_force_vars = [
        'B23001_007E',  # Male 16-19 in labor force
        'B23001_014E',  # Male 20-21 in labor force  
        'B23001_021E',  # Male 22-24 in labor force
        'B23001_028E',  # Male 25-29 in labor force
        'B23001_035E',  # Male 30-34 in labor force
        'B23001_042E',  # Male 35-44 in labor force
        'B23001_049E',  # Male 45-54 in labor force
        'B23001_056E',  # Male 55-59 in labor force
        'B23001_063E',  # Male 60-61 in labor force
        'B23001_070E',  # Male 62-64 in labor force
        'B23001_075E',  # Male 65-69 in labor force
        'B23001_080E',  # Male 70-74 in labor force
        'B23001_085E',  # Male 75+ in labor force
        'B23001_093E',  # Female 16-19 in labor force
        'B23001_100E',  # Female 20-21 in labor force
        'B23001_107E',  # Female 22-24 in labor force  
        'B23001_114E',  # Female 25-29 in labor force
        'B23001_121E',  # Female 30-34 in labor force
        'B23001_128E',  # Female 35-44 in labor force
        'B23001_135E',  # Female 45-54 in labor force
        'B23001_142E',  # Female 55-59 in labor force
        'B23001_149E',  # Female 60-61 in labor force
        'B23001_156E',  # Female 62-64 in labor force
        'B23001_161E',  # Female 65-69 in labor force
        'B23001_166E',  # Female 70-74 in labor force
        'B23001_171E',  # Female 75+ in labor force
    ]
    
    # Base variables for 2010 ACS - adjusted to use available variables
    base_vars = [
        'NAME',
        'B19013_001E',  # Median household income
        'B25077_001E',  # Median house value
        'B15002_015E',  # Male: Bachelor's degree (2010 equivalent)
        'B15002_016E',  # Male: Master's degree (2010 equivalent)
        'B15002_017E',  # Male: Professional degree (2010 equivalent)
        'B15002_018E',  # Male: Doctorate degree (2010 equivalent)
        'B15002_032E',  # Female: Bachelor's degree (2010 equivalent)
        'B15002_033E',  # Female: Master's degree (2010 equivalent)
        'B15002_034E',  # Female: Professional degree (2010 equivalent)
        'B15002_035E',  # Female: Doctorate degree (2010 equivalent)
        'B23001_001E',  # Total population 16+ years
        'B01003_001E',  # Total population
        'B25002_001E',  # Housing units
        'B25002_002E',  # Occupied housing units
        'B25002_003E',  # Vacant housing units
        'B25003_001E',  # Tenure
        'B25003_002E',  # Owner occupied
        'B25003_003E',  # Renter occupied
        'B02001_002E',  # White alone
        'B02001_003E'   # Black or African American alone
    ]
    
    # Combine base variables with labor force variables
    variables = base_vars + labor_force_vars
    
    try:
        # Query ACS5 2010 data for New Haven County
        newhaven_data_2010 = census.acs5.state_county_tract(
            fields=variables,
            state_fips=ct_fips,
            county_fips=newhaven_county_fips,
            tract='*',
            year=2010
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(newhaven_data_2010)
        
        # Replace Census placeholder for missing values with NaN
        df = df.replace(-666666666, np.nan)
        df = df.replace(-666666666.0, np.nan)
        
        # Convert labor force variables to numeric
        for var in labor_force_vars:
            df[var] = pd.to_numeric(df[var], errors='coerce')
        
        # Create a new column that represents the sum of all labor force variables
        # This will be equivalent to B23025_002E (In labor force)
        df['B23025_002E'] = df[labor_force_vars].sum(axis=1)
        
        # Save raw data
        df.to_csv(os.path.join(src_dir, 'newhaven_census_data_2010.csv'), index=False)
        print(f"Saved 2010 census data to {os.path.join(src_dir, 'newhaven_census_data_2010.csv')}")
        
        return df
    except Exception as e:
        print(f"Error downloading 2010 data: {e}")
        # Return empty DataFrame to allow the script to continue
        return pd.DataFrame()

def download_tract_boundaries():
    """
    Downloads census tract boundaries for New Haven using TIGERweb services
    """
    print("Obtaining census tract boundaries for New Haven, CT...")
    
    try:
        # For 2020 tract boundaries using GeoJSON format
        tiger_url_2020 = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Census2020/MapServer/8/query"
        params = {
            'where': f"STATE='09' AND COUNTY='009'",  # Connecticut and New Haven County
            'outFields': '*',
            'outSR': '4326',  # WGS84 coordinate system
            'f': 'geojson'
        }
        
        response = requests.get(tiger_url_2020, params=params)
        if response.status_code == 200:
            geojson_2020 = response.json()
            # Save GeoJSON
            with open(os.path.join(src_dir, 'newhaven_tracts_2020.geojson'), 'w') as f:
                json.dump(geojson_2020, f)
            print(f"Downloaded 2020 tract boundaries to {os.path.join(src_dir, 'newhaven_tracts_2020.geojson')}")
        else:
            print(f"Failed to download 2020 tract boundaries: {response.status_code}")
            print(response.text)
            
        # For 2010 tract boundaries
        tiger_url_2010 = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Census2010/MapServer/14/query"
        
        response = requests.get(tiger_url_2010, params=params)
        if response.status_code == 200:
            geojson_2010 = response.json()
            # Save GeoJSON
            with open(os.path.join(src_dir, 'newhaven_tracts_2010.geojson'), 'w') as f:
                json.dump(geojson_2010, f)
            print(f"Downloaded 2010 tract boundaries to {os.path.join(src_dir, 'newhaven_tracts_2010.geojson')}")
        else:
            print(f"Failed to download 2010 tract boundaries: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error downloading tract boundaries: {e}")

def filter_new_haven_city(county_data):
    """
    Filters county-level data to only include New Haven city tracts
    """
    # Filter based on the 'NAME' field containing 'New Haven'
    # Note: This is a simple approach - for more accuracy, you'd use the spatial boundaries
    new_haven_tracts = county_data[county_data['NAME'].str.contains('New Haven')]
    
    # Save filtered data
    new_haven_tracts.to_csv(os.path.join(canonical_dir, 'newhaven_city_data.csv'), index=False)
    print(f"Filtered to {len(new_haven_tracts)} New Haven city tracts")
    print(f"Saved filtered data to {os.path.join(canonical_dir, 'newhaven_city_data.csv')}")
    
    return new_haven_tracts

def create_model_features():
    """
    Creates model features from census geometric and socioeconomic data for the prediction model
    """
    print("Creating model features from census data...")
    
    try:
        # Import geopandas
        import geopandas as gpd
        from shapely.geometry import Point
        
        # Load GeoJSON directly into a GeoDataFrame
        tracts_2010 = gpd.read_file(os.path.join(src_dir, 'newhaven_tracts_2010.geojson'))
        print(f"Loaded {len(tracts_2010)} census tracts from 2010 GeoJSON")
        
        # Create a standard tract_id field
        if 'TRACTCE' in tracts_2010.columns:
            tracts_2010['tract_id'] = tracts_2010['TRACTCE'].apply(
                lambda x: f"Census Tract {x.lstrip('0')}" if x.lstrip('0') else "Census Tract 0"
            )
        elif 'NAME' in tracts_2010.columns:
            tracts_2010['tract_id'] = tracts_2010['NAME']
        
        # Create downtown point
        downtown = Point(-72.9279, 41.3083)
        
        # Ensure geometry is in a projected CRS for accurate measurements
        # Convert to a projected coordinate system (UTM zone 18N is appropriate for CT)
        tracts_2010 = tracts_2010.to_crs(epsg=32618)  # UTM zone 18N
        downtown_point = gpd.GeoDataFrame(geometry=[downtown], crs="EPSG:4326").to_crs(epsg=32618)
        
        # Calculate geometric features
        # Area in square kilometers
        tracts_2010['area_sqkm'] = tracts_2010.geometry.area / 1000000
        
        # Perimeter in kilometers
        tracts_2010['perimeter_km'] = tracts_2010.geometry.length / 1000
        
        # Compactness ratio (circle = 1, less compact shapes < 1)
        tracts_2010['compactness'] = (4 * np.pi * tracts_2010['area_sqkm']) / (tracts_2010['perimeter_km'] ** 2)
        
        # Count number of parts in MultiPolygons (1 for simple Polygons)
        tracts_2010['num_polygons'] = tracts_2010.geometry.apply(lambda g: len(g.geoms) if hasattr(g, 'geoms') else 1)
        
        # Distance to downtown in kilometers
        tracts_2010['distance_to_downtown'] = tracts_2010.geometry.centroid.distance(downtown_point.geometry[0]) / 1000
        
        # Add centroid coordinates (in original degrees)
        centroids = tracts_2010.geometry.centroid.to_crs(epsg=4326)
        tracts_2010['centroid_lon'] = centroids.x
        tracts_2010['centroid_lat'] = centroids.y
        
        # Create features DataFrame without the geometry column
        features = tracts_2010.drop(columns='geometry').copy()
        
        # Load census data
        try:
            # Try to load 2010 census data first (better for prediction model baseline)
            census_data = pd.read_csv(os.path.join(src_dir, 'newhaven_census_data_2010.csv'))
            year = '2010'
        except:
            # Fall back to 2020 data if 2010 is not available
            census_data = pd.read_csv(os.path.join(src_dir, 'newhaven_census_data_2020.csv'))
            year = '2020'
            
        print(f"Using {year} census data for socioeconomic features")
        
        # Extract tract IDs to match format
        census_data['tract_id'] = census_data['NAME'].apply(
            lambda x: f"Census Tract {x.split(',')[0].strip().split()[-1]}"
        )
        
        # Convert all relevant columns to numeric
        for col in census_data.columns:
            if col not in ['NAME', 'state', 'county', 'tract', 'tract_id']:
                census_data[col] = pd.to_numeric(census_data[col], errors='coerce')
                
        # Create socioeconomic features
        socio_features = pd.DataFrame()
        socio_features['tract_id'] = census_data['tract_id']
        
        # Total population
        socio_features['total_pop'] = census_data['B01003_001E']
        
        # Housing occupancy and tenure
        socio_features['pct_owner_occupied'] = (census_data['B25003_002E'] / census_data['B25003_001E']) * 100
        socio_features['pct_renter_occupied'] = (census_data['B25003_003E'] / census_data['B25003_001E']) * 100
        socio_features['pct_vacant'] = (census_data['B25002_003E'] / census_data['B25002_001E']) * 100
        
        # Income
        socio_features['median_income'] = census_data['B19013_001E']
        
        # Race/ethnicity
        socio_features['pct_white'] = (census_data['B02001_002E'] / census_data['B01003_001E']) * 100
        socio_features['pct_black'] = (census_data['B02001_003E'] / census_data['B01003_001E']) * 100
        
        # Education (combining bachelor's and higher degrees)
        if year == '2010':
            # 2010 has separate male/female education columns
            higher_edu_pop = (
                census_data['B15002_015E'] + census_data['B15002_016E'] + 
                census_data['B15002_017E'] + census_data['B15002_018E'] +
                census_data['B15002_032E'] + census_data['B15002_033E'] + 
                census_data['B15002_034E'] + census_data['B15002_035E']
            )
            # Need to calculate the total 25+ population for 2010 ACS data
            # This would require more variables from the census, so we'll use a simpler approach
            socio_features['pct_higher_edu'] = higher_edu_pop / census_data['B01003_001E'] * 100
        else:
            # 2020 has combined education columns
            higher_edu_pop = (
                census_data['B15003_022E'] + census_data['B15003_023E'] + 
                census_data['B15003_024E'] + census_data['B15003_025E']
            )
            # Again, we'd need the total 25+ population, but we'll simplify
            socio_features['pct_higher_edu'] = higher_edu_pop / census_data['B01003_001E'] * 100
            
        # Labor force participation
        socio_features['pct_in_labor_force'] = (census_data['B23025_002E'] / census_data['B23001_001E']) * 100
        
        # Housing values
        socio_features['median_house_value'] = census_data['B25077_001E']
        
        # Population density
        socio_features['pop_density'] = socio_features['total_pop'] / features['area_sqkm']
        
        # Merge geometric features with socioeconomic features
        final_features = pd.merge(features, socio_features, on='tract_id', how='left')
        
        # Filter to just New Haven city tracts
        if os.path.exists(os.path.join(canonical_dir, 'newhaven_city_data.csv')):
            city_data = pd.read_csv(os.path.join(canonical_dir, 'newhaven_city_data.csv'))
            city_data['tract_id'] = city_data['NAME'].apply(
                lambda x: f"Census Tract {x.split(',')[0].strip().split()[-1]}"
            )
            newhaven_tract_ids = city_data['tract_id'].unique()
            final_features = final_features[final_features['tract_id'].isin(newhaven_tract_ids)]
            
        # Save features to CSV
        final_features.to_csv(os.path.join(analytical_dir, 'newhaven_model_features.csv'), index=False)
        print(f"Saved model features to {os.path.join(analytical_dir, 'newhaven_model_features.csv')}")
        print(f"Created {len(final_features)} features with {len(final_features.columns)} variables")
        
        return final_features
    
    except ImportError:
        print("Error: geopandas not installed. Please install it with:")
        print("pip install geopandas")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error creating model features: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to execute data collection
    """
    print("Starting data collection for New Haven, CT...")
    
    if api_key == 'YOUR_CENSUS_API_KEY':
        print("ERROR: Please set your Census API key in the script.")
        print("Get one at: https://api.census.gov/data/key_signup.html")
        return
    
    # Download census data using direct Census API (avoids cenpy issues)
    newhaven_data_2020 = download_census_data_direct()
    newhaven_data_2010 = download_census_2010_data_direct()
    
    # Download tract boundaries
    download_tract_boundaries()
    
    # Filter to New Haven city only
    if not newhaven_data_2020.empty:
        newhaven_city_2020 = filter_new_haven_city(newhaven_data_2020)
        print(f"Successfully processed 2020 data with {len(newhaven_city_2020)} New Haven city tracts")
    
    if not newhaven_data_2010.empty:
        newhaven_city_2010 = filter_new_haven_city(newhaven_data_2010)
        print(f"Successfully processed 2010 data with {len(newhaven_city_2010)} New Haven city tracts")
    
    # Create model features from geographic data
    create_model_features()
    
    print("\nInitial data collection complete!")
    print("\nNext steps:")
    print("1. For more accurate city filtering, use the boundary files with a spatial join")
    print("2. Run the 02-NewHaven-SES-Scores.py script to calculate socioeconomic status scores")
    print("3. Run the 03-NewHaven-Prediction-Model.py script to train the gentrification prediction model")
    
if __name__ == "__main__":
    main() 