import os
import random
import geopandas as gpd
import rasterio
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

# Randomly select a river
selected_river = random.choice(list(river_data.keys()))
river_info = river_data[selected_river]

# Construct the path to the shapefile and bands
shp_path = os.path.join(BASE_DIR, f"data_{selected_river}", river_info["shp_file"])
band_paths = [os.path.join(BASE_DIR, f"data_{selected_river}", band) for band in river_info["bands"]]

# Load the shapefile using geopandas
print(f"Loading shapefile from: {shp_path}")
gdf = gpd.read_file(shp_path)

# Print the columns to verify the structure
print("Columns in the shapefile:", gdf.columns)

# Define the depth column name as it appears in your shapefile
depth_column_name = 'Depth'  # Updated to match the correct case

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
        selected_point = selected_geometry.coords[0] if selected_geometry.geom_type == 'LineString' else random.choice(
            list(selected_geometry.exterior.coords))
    else:
        print("Unsupported geometry type.")
        exit()

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

    # Now, perform the analysis: plot depth vs. intensity for all available data points
    X = []
    y = []

    for geometry, depth in zip(valid_geometries.geometry, valid_geometries[depth_column_name]):
        intensities_for_geometry = []
        for i, band_data in enumerate(band_data_list):
            row, col = rasterio.open(band_paths[i]).index(geometry.x, geometry.y)
            intensities_for_geometry.append(band_data[row, col])
        X.append(intensities_for_geometry)
        y.append(depth)

    X = np.array(X)
    y = np.array(y)

    # Model 1: Linear Regression to predict depth based on intensity values
    model = LinearRegression()
    model.fit(X, y)

    # Use the trained model to predict the depth for the selected point
    predicted_depth_model1 = model.predict([intensities])[0]
    print(f"Predicted Depth (Linear Regression) at selected point: {predicted_depth_model1}")

    # Model 2: Intensity-based depth prediction using custom scaling
    # Normalize the intensity values (scale between 0 and 1)
    max_intensity = np.max(intensities)
    min_intensity = np.min(intensities)
    normalized_intensities = [(i - min_intensity) / (max_intensity - min_intensity) for i in intensities]

    # Calculate the predicted depth using a custom scaling based on intensity averages
    mean_intensity = np.mean(normalized_intensities)

    # Custom dynamic depth mapping (you can adjust the range)
    min_depth = 0.0  # Min depth value (adjust as needed)
    max_depth = 1.0  # Max depth value (adjust as needed)
    predicted_depth_model2 = (min_depth + (mean_intensity * (max_depth - min_depth))) / 2.5

    print(f"Predicted Depth (Intensity-based, adjusted) at selected point: {predicted_depth_model2}")

    # Model 3: Dynamic scaling based on data range (more adaptive scaling)
    scaling_factor = 0.5  # Adjust the scaling factor to tune prediction closer to the actual value
    predicted_depth_model3 = mean_intensity * scaling_factor
    print(f"Predicted Depth (Dynamic Scaling) at selected point: {predicted_depth_model3}")

    # Calculate percentage error for all models
    percentage_error_model1 = abs((predicted_depth_model1 - selected_depth) / selected_depth) * 100
    percentage_error_model2 = abs((predicted_depth_model2 - selected_depth) / selected_depth) * 100
    percentage_error_model3 = abs((predicted_depth_model3 - selected_depth) / selected_depth) * 100

    print(f"Percentage Error (Linear Regression Model): {percentage_error_model1:.2f}%")
    print(f"Percentage Error (Intensity-based Model, adjusted): {percentage_error_model2:.2f}%")
    print(f"Percentage Error (Dynamic Scaling Model): {percentage_error_model3:.2f}%")

else:
    print("No valid geometries with depth information found.")
