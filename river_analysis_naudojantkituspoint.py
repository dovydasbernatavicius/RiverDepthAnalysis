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

# Define the base directory for river data
BASE_DIR = r"C:\Users\dober\PycharmProjects\RiverDepthAnalysis\data_rivers"

# River folders and their respective band file names
river_data = {
    "jura": {
        "shp_file": "Jura-1.01-dv_SHP-230713.shp",  # Updated for Jura
        "bands": [
            "Jura-1.01-nir-230713.tif",
            "Jura-1.01-orto-230713.tif",
            "Jura-1.01-red_e-230713.tif",
            "Jura-1.01-red-230713.tif"
        ]
    },
    "musa": {
        "shp_file": "Musa-0.548-dv_SHP-230711.shp",  # Updated for Musa
        "bands": [
            "Musa-0.548-nir-230711.tif",
            "Musa-0.548-orto-230711.tif",
            "Musa-0.548-red_e-230711.tif",
            "Musa-0.548-red-230711.tif"
        ]
    },
    "verkne": {
        "shp_file": "Verkne-1.42-dv_SHP-230718.shp",  # Updated for Verkne
        "bands": [
            "Verkne-1.42-nir-230718.tif",
            "Verkne-1.42-orto-230718.tif",
            "Verkne-1.42-red_e-230718.tif",
            "Verkne-1.42-red-230718.tif"
        ]
    }
}

# Repeat the entire process 30 times
for iteration in range(1):
    print(f"Iteration {iteration + 1}/30")

    # Combine all river data for training
    global_X = []
    global_y = []
    coordinates = []

    for river_name, river_info in river_data.items():
        # Construct the path to the shapefile and bands
        shp_path = os.path.join(BASE_DIR, f"data_{river_name}", river_info["shp_file"])
        band_paths = [os.path.join(BASE_DIR, f"data_{river_name}", band) for band in river_info["bands"]]

        # Load the shapefile using geopandas
        print(f"Loading shapefile from: {shp_path}")
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

            # Now, collect data points from all available valid geometries
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
    # Model 1: Random Forest Regression
    model_rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1)  # Increased number of trees to 1000 for higher precision
    model_rf.fit(global_X, global_y)

    # Cross-validation for Random Forest
    rf_cv_scores = cross_val_score(model_rf, global_X, global_y, cv=5, scoring='neg_mean_absolute_error')
    print(f"Cross-Validation MAE for Random Forest: {-np.mean(rf_cv_scores):.4f} +/- {np.std(rf_cv_scores):.4f}")

    # Model 2: XGBoost Regression
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror', max_depth=8, learning_rate=0.03, n_estimators=300, subsample=0.8, n_jobs=-1)  # Adjusted hyperparameters for better performance
    model_xgb.fit(global_X, global_y)

    # Cross-validation for XGBoost
    xgb_cv_scores = cross_val_score(model_xgb, global_X, global_y, cv=5, scoring='neg_mean_absolute_error')
    print(f"Cross-Validation MAE for XGBoost: {-np.mean(xgb_cv_scores):.4f} +/- {np.std(xgb_cv_scores):.4f}")

    # Interpolate riverbed contour
    grid_x, grid_y = np.mgrid[coordinates[:, 0].min():coordinates[:, 0].max():1000j,
                              coordinates[:, 1].min():coordinates[:, 1].max():1000j]

    # Interpolate using griddata from scipy
    interpolated_depth = griddata(coordinates, global_y, (grid_x, grid_y), method='cubic')

    # Clip interpolated depth values to a realistic range (0 to 4 meters)
    interpolated_depth = np.clip(interpolated_depth, 0, 4)

    # Plot the interpolated riverbed contour along with the original depth points
    plt.figure(figsize=(12, 8))
    plt.contourf(grid_x, grid_y, interpolated_depth, cmap='viridis', levels=50)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=global_y, cmap='coolwarm', edgecolor='k', s=20, label='Shapefile Depth Points')
    plt.colorbar(label='Interpolated Depth (m)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Interpolated Riverbed Depth Contour with Shapefile Depth Points (Iteration {iteration + 1})')
    plt.legend()
    plt.show()
    # Randomly select a river for prediction
    selected_river = random.choice(list(river_data.keys()))
    river_info = river_data[selected_river]

    # Construct the path to the shapefile and bands
    shp_path = os.path.join(BASE_DIR, f"data_{selected_river}", river_info["shp_file"])
    band_paths = [os.path.join(BASE_DIR, f"data_{selected_river}", band) for band in river_info["bands"]]

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

        print(f"Predicted Depth (Random Forest) at selected point: {predicted_depth_rf}")
        print(f"Predicted Depth (XGBoost) at selected point: {predicted_depth_xgb}")

        # Calculate and print MAE for the selected point
        mae_rf = mean_absolute_error([selected_depth], [predicted_depth_rf])
        mae_xgb = mean_absolute_error([selected_depth], [predicted_depth_xgb])
        print(f"Mean Absolute Error (Random Forest) for selected point: {mae_rf:.4f}")
        print(f"Mean Absolute Error (XGBoost) for selected point: {mae_xgb:.4f}")

        # Interpolation around the selected point
        point_buffer = 50  # Buffer size for interpolation grid around the point
        point_grid_x, point_grid_y = np.mgrid[(selected_point.x - point_buffer):(selected_point.x + point_buffer):100j,
                                              (selected_point.y - point_buffer):(selected_point.y + point_buffer):100j]
        interpolated_point_depth = griddata(coordinates, global_y, (point_grid_x, point_grid_y), method='cubic')
        interpolated_point_depth = np.clip(interpolated_point_depth, 0, 4)

        # Plot the interpolation around the selected point
        plt.figure(figsize=(8, 6))
        plt.contourf(point_grid_x, point_grid_y, interpolated_point_depth, cmap='viridis', levels=50)
        plt.scatter([selected_point.x], [selected_point.y], c='red', marker='x', s=100, label='Selected Point')
        plt.colorbar(label='Interpolated Depth (m)')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Detailed Interpolation Around Selected Point (Iteration {iteration + 1})')
        plt.legend()
        plt.show()

    else:
        print("No valid geometries with depth information found.")
