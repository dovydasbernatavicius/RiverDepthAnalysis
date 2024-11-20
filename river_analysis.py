import os
import random
import geopandas as gpd
import rasterio
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Define the base directory for river data
BASE_DIR = r"C:\Users\dober\PycharmProjects\RiverDepthAnalysis\data_rivers"

# River folders and their respective band file names
river_data = {
    "jura": {
        "shp_file": "Jura-1.01-dv_SHP-230713.shp",
        "bands": [
            "Jura-1.01-nir-230713.tif",
            "Jura-1.01-orto-230713.tif",
            "Jura-1.01-red_e-230713.tif",
            "Jura-1.01-red-230713.tif"
        ]
    },
    "musa": {
        "shp_file": "Musa-0.548-dv_SHP-230711.shp",
        "bands": [
            "Musa-0.548-nir-230711.tif",
            "Musa-0.548-orto-230711.tif",
            "Musa-0.548-red_e-230711.tif",
            "Musa-0.548-red-230711.tif"
        ]
    },
    "verkne": {
        "shp_file": "Verkne-1.42-dv_SHP-230718.shp",
        "bands": [
            "Verkne-1.42-nir-230718.tif",
            "Verkne-1.42-orto-230718.tif",
            "Verkne-1.42-red_e-230718.tif",
            "Verkne-1.42-red-230718.tif"
        ]
    }
}

# Combine all river data for training
global_X = []
global_y = []
coordinates = []

for river_name, river_info in river_data.items():
    # Construct the path to the shapefile and bands
    shp_path = os.path.join(BASE_DIR, f"data_{river_name}", river_info["shp_file"])
    band_paths = [os.path.join(BASE_DIR, f"data_{river_name}", band) for band in river_info["bands"]]

    # Print loading shapefile message
    print(f"Loading shapefile from: {shp_path}")

    # Load the shapefile using geopandas
    gdf = gpd.read_file(shp_path)

    # Define the depth column name as it appears in your shapefile
    depth_column_name = 'Depth'

    # Filter out geometries that have no depth value
    valid_geometries = gdf[gdf[depth_column_name].notnull()]

    # Ensure there are valid geometries with depth values
    if not valid_geometries.empty:
        # Open the bands once and store them in memory
        band_data_list = []
        for band_path in band_paths:
            with rasterio.open(band_path) as src:
                band_data_list.append(src.read(1))

        # Collect data points from valid geometries
        for geometry, depth in zip(valid_geometries.geometry, valid_geometries[depth_column_name]):
            if geometry.geom_type == 'Point':
                intensities_for_geometry = []
                for i, band_data in enumerate(band_data_list):
                    row, col = rasterio.open(band_paths[i]).index(geometry.x, geometry.y)
                    intensities_for_geometry.append(band_data[row, col])
                global_X.append(intensities_for_geometry)
                global_y.append(depth)
                coordinates.append((geometry.x, geometry.y))

# Convert global lists to numpy arrays
global_X = np.array(global_X)
global_y = np.array(global_y)
coordinates = np.array(coordinates)

# Normalize input data
scaler = StandardScaler()
global_X = scaler.fit_transform(global_X)

# Train models on the full dataset
print("Training the models...")

# Model 1: Random Forest Regression
print("Training Random Forest model...")
model_rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
model_rf.fit(global_X, global_y)

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(model_rf, global_X, global_y, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-Validation MAE for Random Forest: {-np.mean(rf_cv_scores):.4f} +/- {np.std(rf_cv_scores):.4f}")

# Model 2: XGBoost Regression
print("Training XGBoost model...")
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', max_depth=8, learning_rate=0.03, n_estimators=300, subsample=0.8, n_jobs=-1)
model_xgb.fit(global_X, global_y)

# Cross-validation for XGBoost
xgb_cv_scores = cross_val_score(model_xgb, global_X, global_y, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-Validation MAE for XGBoost: {-np.mean(xgb_cv_scores):.4f} +/- {np.std(xgb_cv_scores):.4f}")

# Model 3: Gaussian Process Regression for Interpolation (Training using part of the data)
print("Training Gaussian Process model...")
gp_indices = random.sample(range(len(global_y)), int(0.5 * len(global_y)))  # Use 50% of data for training GP to reduce overfitting
gp_train_coords = coordinates[gp_indices]
gp_train_depths = global_y[gp_indices]

# Adjust kernel to limit flexibility (smaller length scale and variance)
kernel = C(0.5, (1e-3, 1e1)) * RBF(5, (1e-2, 1e1))
model_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
model_gp.fit(gp_train_coords, gp_train_depths)

successful_predictions = 0

# Predict depth at random points
for i in range(10):
    # Randomly select a river for prediction
    selected_river = random.choice(list(river_data.keys()))
    river_info = river_data[selected_river]

    # Construct the path to the shapefile and bands
    shp_path = os.path.join(BASE_DIR, f"data_{selected_river}", river_info["shp_file"])
    band_paths = [os.path.join(BASE_DIR, f"data_{selected_river}", band) for band in river_info["bands"]]

    # Print loading shapefile message
    print(f"Loading shapefile from: {shp_path}")

    # Load the shapefile using geopandas
    gdf = gpd.read_file(shp_path)

    # Filter out geometries that have no depth value
    valid_geometries = gdf[gdf[depth_column_name].notnull()]

    # Ensure there are valid geometries with depth values
    if not valid_geometries.empty:
        # Randomly select a valid point from the filtered geometries
        selected_geometry = valid_geometries.sample(n=1).geometry.values[0]

        # Extract coordinates based on geometry type
        if selected_geometry.geom_type == 'Point':
            selected_point = selected_geometry
        elif selected_geometry.geom_type in ['LineString', 'Polygon']:
            selected_point = selected_geometry.coords[0] if selected_geometry.geom_type == 'LineString' else random.choice(list(selected_geometry.exterior.coords))
        else:
            print("Unsupported geometry type.")
            continue

        # Get the actual depth value at the selected point
        selected_depth = valid_geometries.loc[valid_geometries.geometry == selected_geometry, depth_column_name].values[0]

        # Print selected river, point, and depth
        print(f"Selected river: {selected_river.capitalize()}")
        print(f"Selected point: {selected_point.x}, {selected_point.y}")
        print(f"Actual Depth at selected point: {selected_depth}")

        # Open the bands once and store them in memory
        band_data_list = []
        for band_path in band_paths:
            with rasterio.open(band_path) as src:
                band_data_list.append(src.read(1))

        # Extract pixel intensities for each of the four bands at the selected point
        intensities = []
        for i, band_data in enumerate(band_data_list):
            row, col = rasterio.open(band_paths[i]).index(selected_point.x, selected_point.y)
            intensities.append(band_data[row, col])

        # Normalize the features for prediction
        intensities = scaler.transform([intensities])[0]

        # Predict depth using the trained models
        predicted_depth_rf = model_rf.predict([intensities])[0]
        predicted_depth_xgb = model_xgb.predict([intensities])[0]
        predicted_depth_gp = model_gp.predict([[selected_point.x, selected_point.y]])[0]

        print(f"Predicted Depth (Random Forest) at selected point: {predicted_depth_rf}")
        print(f"Predicted Depth (XGBoost) at selected point: {predicted_depth_xgb}")
        print(f"Predicted Depth (Gaussian Process Interpolation) at selected point: {predicted_depth_gp}")

        # Calculate and print MAE for the selected point
        mae_rf = mean_absolute_error([selected_depth], [predicted_depth_rf])
        mae_xgb = mean_absolute_error([selected_depth], [predicted_depth_xgb])
        mae_gp = mean_absolute_error([selected_depth], [predicted_depth_gp])
        print(f"Mean Absolute Error (Random Forest) for selected point: {mae_rf:.4f}")
        print(f"Mean Absolute Error (XGBoost) for selected point: {mae_xgb:.4f}")
        print(f"Mean Absolute Error (Gaussian Process Interpolation) for selected point: {mae_gp:.4f}")

        successful_predictions += 1

    else:
        print("No valid geometries with depth information found.")

# Summary of predictions
print(f"Successful predictions: {successful_predictions}/10")
