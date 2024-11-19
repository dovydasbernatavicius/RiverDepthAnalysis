import os
import random
import geopandas as gpd
import rasterio
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

# Load data function
def load_data(river_data, base_dir):
    global_X, global_y, coordinates, groups = [], [], [], []

    for river_idx, (river_name, river_info) in enumerate(river_data.items()):
        shp_path = os.path.join(base_dir, f"data_{river_name}", river_info["shp_file"])
        band_paths = [os.path.join(base_dir, f"data_{river_name}", band) for band in river_info["bands"]]

        # Load the shapefile using geopandas
        gdf = gpd.read_file(shp_path)
        depth_column_name = 'Depth'

        # Filter valid geometries with depth values
        valid_geometries = gdf[gdf[depth_column_name].notnull()]

        if not valid_geometries.empty:
            band_data_list = [rasterio.open(band).read(1) for band in band_paths]

            for geometry, depth in zip(valid_geometries.geometry, valid_geometries[depth_column_name]):
                if geometry.geom_type == 'Point':
                    intensities = []
                    for i, band_data in enumerate(band_data_list):
                        row, col = rasterio.open(band_paths[i]).index(geometry.x, geometry.y)
                        intensities.append(band_data[row, col])
                    global_X.append(intensities)
                    global_y.append(depth)
                    coordinates.append((geometry.x, geometry.y))
                    groups.append(river_idx)

    return np.array(global_X), np.array(global_y), np.array(coordinates), np.array(groups)

# Train and evaluate model using LeaveOneGroupOut
def evaluate_model_with_logo(model, X, y, groups):
    logo = LeaveOneGroupOut()
    mae_scores, rmse_scores, r2_scores = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        model.fit(X[train_idx], y[train_idx])
        predictions = model.predict(X[test_idx])
        mae_scores.append(mean_absolute_error(y[test_idx], predictions))
        rmse_scores.append(np.sqrt(mean_squared_error(y[test_idx], predictions)))
        r2_scores.append(r2_score(y[test_idx], predictions))

    return np.mean(mae_scores), np.mean(rmse_scores), np.mean(r2_scores)

# Benchmark comparison using average depth as baseline
def benchmark(y):
    mean_depth = np.mean(y)
    return mean_absolute_error(y, [mean_depth] * len(y))

# Plot feature importance
def plot_feature_importance(model, feature_names, title):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        sns.barplot(x=feature_names, y=importance)
        plt.title(title)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.show()

# Main processing and enhancements
global_X, global_y, coordinates, groups = load_data(river_data, BASE_DIR)

# Normalize data
scaler = StandardScaler()
global_X = scaler.fit_transform(global_X)

# Define models
model_rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', max_depth=8, learning_rate=0.03, n_estimators=300, subsample=0.8, n_jobs=-1)

# Train and cross-validate models
rf_mae, rf_rmse, rf_r2 = evaluate_model_with_logo(model_rf, global_X, global_y, groups)
xgb_mae, xgb_rmse, xgb_r2 = evaluate_model_with_logo(model_xgb, global_X, global_y, groups)

# Print benchmark and cross-validation results
benchmark_mae = benchmark(global_y)
print("Benchmark MAE (Mean Depth):")
print(f"{benchmark_mae:.4f}")
print("Random Forest MAE:")
print(f"MAE={rf_mae:.4f}")
print("Random Forest RMSE:")
print(f"RMSE={rf_rmse:.4f}")
print("Random Forest R²:")
print(f"R²={rf_r2:.4f}")
print("XGBoost MAE:")
print(f"MAE={xgb_mae:.4f}")
print("XGBoost RMSE:")
print(f"RMSE={xgb_rmse:.4f}")
print("XGBoost R²:")
print(f"R²={xgb_r2:.4f}")

# Plot feature importance
plot_feature_importance(model_rf, feature_names=[f'Band {i+1}' for i in range(global_X.shape[1])], title='Random Forest Feature Importance')
plot_feature_importance(model_xgb, feature_names=[f'Band {i+1}' for i in range(global_X.shape[1])], title='XGBoost Feature Importance')

# Save models
joblib.dump(model_rf, "random_forest_model.pkl")
joblib.dump(model_xgb, "xgboost_model.pkl")

# Interpolate and plot riverbed depth
grid_x, grid_y = np.mgrid[coordinates[:, 0].min():coordinates[:, 0].max():1000j,
                          coordinates[:, 1].min():coordinates[:, 1].max():1000j]
interpolated_depth = griddata(coordinates, global_y, (grid_x, grid_y), method='cubic')
interpolated_depth = np.clip(interpolated_depth, 0, 4)

plt.figure(figsize=(12, 8))
plt.contourf(grid_x, grid_y, interpolated_depth, cmap='viridis', levels=50)
plt.scatter(coordinates[:, 0], coordinates[:, 1], c=global_y, cmap='coolwarm', edgecolor='k', s=20)
plt.colorbar(label='Interpolated Depth (m)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Interpolated Riverbed Depth Contour')
plt.show()

# Random river prediction
selected_river = random.choice(list(river_data.keys()))
river_info = river_data[selected_river]
shp_path = os.path.join(BASE_DIR, f"data_{selected_river}", river_info["shp_file"])
band_paths = [os.path.join(BASE_DIR, f"data_{selected_river}", band) for band in river_info["bands"]]
gdf = gpd.read_file(shp_path)
depth_column_name = 'Depth'
valid_geometries = gdf[gdf[depth_column_name].notnull()]

if not valid_geometries.empty:
    selected_geometry = valid_geometries.sample(n=1).geometry.values[0]
    if selected_geometry.geom_type == 'Point':
        selected_point = selected_geometry

    selected_depth = valid_geometries.loc[valid_geometries.geometry == selected_geometry, depth_column_name].values[0]
    band_data_list = [rasterio.open(band).read(1) for band in band_paths]
    intensities = [band[row, col] for band in band_data_list for row, col in [rasterio.open(band_paths[0]).index(selected_point.x, selected_point.y)]]
    intensities = scaler.transform([intensities])[0]

    predicted_depth_rf = model_rf.predict([intensities])[0]
    predicted_depth_xgb = model_xgb.predict([intensities])[0]
    mae_rf = mean_absolute_error([selected_depth], [predicted_depth_rf])
    mae_xgb = mean_absolute_error([selected_depth], [predicted_depth_xgb])

    print("-------------------------------------------------")
    print(f"Benchmark MAE (Mean Depth): {benchmark_mae:.4f}")
    print(f"Random Forest MAE: {rf_mae:.4f}")
    print(f"Random Forest RMSE: {rf_rmse:.4f}")
    print(f"Random Forest R²: {rf_r2:.4f}")
    print(f"XGBoost MAE: {xgb_mae:.4f}")
    print(f"XGBoost RMSE: {xgb_rmse:.4f}")
    print(f"XGBoost R²: {xgb_r2:.4f}")
    print("-------------------------------------------------")
    print(f"Selected River: {selected_river.capitalize()}")
    print(f"Actual Depth: {selected_depth:.4f}")
    print(f"Random Forest Predicted Depth: {predicted_depth_rf:.4f}")
    print(f"XGBoost Predicted Depth: {predicted_depth_xgb:.4f}")
    print(f"Random Forest MAE: {mae_rf:.4f}")
    print(f"XGBoost MAE: {mae_xgb:.4f}")
    print("-------------------------------------------------")

else:
    print("No valid geometries with depth information found.")
