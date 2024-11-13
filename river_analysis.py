import os
import random
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np

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

    # Get the actual depth value at the selected point (just for display)
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

    # Normalize the intensity values (scale between 0 and 1)
    max_intensity = np.max(intensities)
    min_intensity = np.min(intensities)
    normalized_intensities = [(i - min_intensity) / (max_intensity - min_intensity) for i in intensities]

    # Print out the normalized intensities for the selected point
    print(f"Normalized Intensities at selected point: {normalized_intensities}")

    # Calculate the predicted depth using a simple scaling based on intensity averages
    # Custom scaling: Find the mean of the normalized intensities and use a new scaling heuristic
    mean_intensity = np.mean(normalized_intensities)

    # Create a dynamic depth mapping based on the intensity values (this is the key change)
    # Here, you can define your own mapping logic to fit the data.

    # Example: A dynamic scaling factor based on a linear relationship between intensity and depth
    # Adjusting this range based on data can help achieve better results
    min_depth = 0.0  # Min depth value (can adjust)
    max_depth = 1.0  # Max depth value (can adjust)
    predicted_depth = min_depth + (mean_intensity * (max_depth - min_depth))

    print(f"Predicted Depth from intensity (after dynamic scaling): {predicted_depth}")

    # Calculate percentage error
    percentage_error = abs((predicted_depth - selected_depth) / selected_depth) * 100
    print(f"Percentage Error: {percentage_error:.2f}%")

    # Visualize the four bands around the selected point
    window_size = 300  # Size of the window around the point
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for i, band_data in enumerate(band_data_list):
        # Get the coordinates of the pixel for the selected point
        row, col = rasterio.open(band_paths[i]).index(selected_point.x, selected_point.y)

        # Define the window for display
        window = band_data[row - window_size // 2: row + window_size // 2,
                 col - window_size // 2: col + window_size // 2]

        # Plot the window
        ax = axes[i // 2, i % 2]
        ax.imshow(window, cmap='gray')
        ax.set_title(f'Band: {os.path.basename(band_paths[i])} ({selected_river.capitalize()})')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

else:
    print("No valid geometries with depth information found.")