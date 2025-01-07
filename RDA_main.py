import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# --------------------------------------------------------------------------
# 1) SPECIFY YOUR DATA
# --------------------------------------------------------------------------
BASE_DIR = r"C:\Users\dober\PycharmProjects\RDA_Python\river_data"

river_data = {
    "jura": [
        {
            "shp_file": "Jura-1.01-dv_SHP-230713.shp",
            "bands": [
                "Jura-1.01-nir-230713.tif",
                "Jura-1.01-orto-230713.tif",
                "Jura-1.01-red_e-230713.tif",
                "Jura-1.01-red-230713.tif",
                "Jura-1.01-green-230713.tif"
            ]
        },
        {
            "shp_file": "Jura-3.24-dv_SHP-230914.shp",
            "bands": [
                "Jura-3.24-nir-230914.tif",
                "Jura-3.24-orto-230914.tif",
                "Jura-3.24-red_e-230914.tif",
                "Jura-3.24-red-230914.tif",
                "Jura-3.24-green-230914.tif"
            ]
        }
    ],
    "musa": [
        {
            "shp_file": "Musa-0.548-dv_SHP-230711.shp",
            "bands": [
                "Musa-0.548-nir-230711.tif",
                "Musa-0.548-orto-230711.tif",
                "Musa-0.548-red_e-230711.tif",
                "Musa-0.548-red-230711.tif",
                "Musa-0.548-green-230711.tif"
            ]
        },
        {
            "shp_file": "Musa-0.506-dva_SHP-240923.shp",
            "bands": [
                "Musa-0.506-nir-240923.tif",
                "Musa-0.506-orto-240923.tif",
                "Musa-0.506-red_e-240923.tif",
                "Musa-0.506-red-240923.tif",
                "Musa-0.506-green-240923.tif"
            ]
        }
    ],
    "susve": [
        {
            "shp_file": "Susve-0.491-dv_SHP-230922.shp",
            "bands": [
                "Susve-0.491-nir-230922.tif",
                "Susve-0.491-orto-230922.tif",
                "Susve-0.491-red-230922.tif",
                "Susve-0.491-green-230922.tif",
                "Susve-0.491-red_e-230922.tif"
            ]
        },
        {
            "shp_file": "Susve-0.697-dva_SHP-240626.shp",
            "bands": [
                "Susve-0.697-nir-240626.tif",
                "Susve-0.697-orto-240626.tif",
                "Susve-0.697-red-240626.tif",
                "Susve-0.697-green-240626.tif",
                "Susve-0.697-red_e-240626.tif"
            ]
        }
    ],
    "verkne": [
        {
            "shp_file": "Verkne-1.42-dv_SHP-230718.shp",
            "bands": [
                "Verkne-1.42-nir-230718.tif",
                "Verkne-1.42-orto-230718.tif",
                "Verkne-1.42-red_e-230718.tif",
                "Verkne-1.42-red-230718.tif",
                "Verkne-1.42-green-230718.tif"
            ]
        }
    ]
}

def check_files_exist():
    """Check that each shapefile/band actually exists."""
    missing_files = False
    for river_name, datasets in river_data.items():
        for dataset in datasets:
            shp_path = os.path.join(BASE_DIR, f"data_{river_name}", dataset["shp_file"])
            if not os.path.isfile(shp_path):
                print(f"[WARNING] Missing shapefile: {shp_path}")
                missing_files = True

            for band_file in dataset["bands"]:
                band_path = os.path.join(BASE_DIR, f"data_{river_name}", band_file)
                if not os.path.isfile(band_path):
                    print(f"[WARNING] Missing band: {band_path}")
                    missing_files = True

    if missing_files:
        print("[ERROR] Some required files are missing.")
        return False
    print("[INFO] All required files found.")
    return True

# --------------------------------------------------------------------------
# HELPER: AVERAGE A large NEIGHBORHOOD
# --------------------------------------------------------------------------
def average_patch(band_data, row, col, patch_size=3):
    """
    For each band_data (2D array), average the pixel intensities in a patch
    of size (2*patch_size + 1) around (row, col). Default patch_size=3 => 7x7 region.
    """
    rows, cols = band_data.shape
    rmin = max(0, row - patch_size)
    rmax = min(rows, row + patch_size + 1)
    cmin = max(0, col - patch_size)
    cmax = min(cols, col + patch_size + 1)

    patch = band_data[rmin:rmax, cmin:cmax]
    return float(np.mean(patch))  # average intensity

def load_training_data(train_rivers, patch_size=3):
    """
    Load shapefiles + raster band data from the specified rivers (all their datasets).
    For each geometry, gather a 'patch_size' region average intensities to reduce noise.
    """
    global_X = []
    global_y = []

    for river_name in train_rivers:
        for dataset in river_data.get(river_name, []):
            shp_file = dataset["shp_file"]
            band_list = dataset["bands"]

            shp_path = os.path.join(BASE_DIR, f"data_{river_name}", shp_file)
            if not os.path.isfile(shp_path):
                print(f"[WARNING] Shapefile missing: {shp_path}")
                continue

            # Read the shapefile
            gdf = gpd.read_file(shp_path)
            if "Depth" not in gdf.columns:
                print(f"[WARNING] No 'Depth' column in {shp_file}. Skipping...")
                continue

            valid_gdf = gdf[gdf["Depth"].notnull()]
            if valid_gdf.empty:
                print(f"[WARNING] All Depth values are null in {shp_file}. Skipping...")
                continue

            # Load band data in memory
            band_arrays = []
            band_transforms = []
            for band_file in band_list:
                band_path = os.path.join(BASE_DIR, f"data_{river_name}", band_file)
                with rasterio.open(band_path) as src:
                    band_arrays.append(src.read(1))
                    band_transforms.append((src.transform, src.width, src.height))

            # For each geometry, get the average intensity in a patch
            for idx, row_data in valid_gdf.iterrows():
                geom = row_data.geometry
                depth_val = row_data["Depth"]
                if geom is None or geom.is_empty or geom.geom_type != 'Point':
                    continue

                x_coord, y_coord = geom.x, geom.y
                intensities = []
                out_of_bounds = False

                for i, band_data in enumerate(band_arrays):
                    transform, w, h = band_transforms[i]
                    col, row_idx = ~transform * (x_coord, y_coord)
                    row_idx = int(round(row_idx))
                    col = int(round(col))

                    # Check if the center is within the raster
                    if 0 <= row_idx < h and 0 <= col < w:
                        avg_val = average_patch(band_data, row_idx, col, patch_size=patch_size)
                        intensities.append(avg_val)
                    else:
                        out_of_bounds = True
                        break

                if not out_of_bounds:
                    global_X.append(intensities)
                    global_y.append(depth_val)

    return np.array(global_X), np.array(global_y)

def load_test_data(test_river, patch_size=3):
    """
    Combines all shapefiles for 'test_river' into one GeoDataFrame,
    plus parallel band arrays. Then we do the patch-based average
    in the predict_random_point function.
    """
    import pandas as pd

    combined_gdf = gpd.GeoDataFrame()
    list_of_band_data = []
    list_of_band_info = []

    if test_river not in river_data:
        print(f"[ERROR] Test river '{test_river}' not in dictionary.")
        return combined_gdf, list_of_band_data, list_of_band_info

    for dataset in river_data[test_river]:
        shp_file = dataset["shp_file"]
        band_list = dataset["bands"]

        shp_path = os.path.join(BASE_DIR, f"data_{test_river}", shp_file)
        if not os.path.isfile(shp_path):
            print(f"[WARNING] Missing shapefile: {shp_path}")
            continue

        gdf = gpd.read_file(shp_path)
        if "Depth" not in gdf.columns:
            valid_gdf = gdf
        else:
            valid_gdf = gdf[gdf["Depth"].notnull()]

        if not valid_gdf.empty:
            combined_gdf = pd.concat([combined_gdf, valid_gdf], ignore_index=True)

        # Load bands
        band_arrays = []
        band_info = []
        for band_file in band_list:
            band_path = os.path.join(BASE_DIR, f"data_{test_river}", band_file)
            with rasterio.open(band_path) as src:
                band_arrays.append(src.read(1))
                band_info.append((src.transform, src.width, src.height))
        list_of_band_data.append(band_arrays)
        list_of_band_info.append(band_info)

    # Convert to GeoDataFrame if not already
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry="geometry", crs=gdf.crs)
    return combined_gdf, list_of_band_data, list_of_band_info

def train_models(X, y):
    """
    Aggressively overfit both RandomForest and XGBoost.
    We skip cross-validation entirely to maximize training set fit.
    """
    # 1) Print naive average depth
    naive_depth = np.mean(y)
    print(f"[BENCHMARK] Training set average depth = {naive_depth:.4f}")

    # 2) Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3) Overfit RandomForest
    rf_model = RandomForestRegressor(
        n_estimators=2000,    # huge forest
        max_depth=None,       # no depth limit -> can overfit
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_scaled, y)

    # 4) Overfit XGBoost
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=50,           # very deep
        learning_rate=0.005,    # smaller LR -> can overfit with enough rounds
        n_estimators=3000,      # many boosting rounds
        subsample=1.0,          # no row subsampling
        colsample_bytree=1.0,   # no feature subsampling
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_scaled, y)

    return rf_model, xgb_model, scaler, naive_depth

def predict_random_point(
    combined_gdf,
    list_of_band_data,
    list_of_band_info,
    rf_model,
    xgb_model,
    scaler,
    patch_size=3,
    naive_depth=None,
    num_bands_expected=5
):
    """Pick random geometry from test shapefile, average patch, predict, compare to naive depth."""
    from sklearn.metrics import mean_absolute_error

    if combined_gdf.empty:
        print("[ERROR] Test shapefile has no valid Depth data.")
        return

    selected_row = combined_gdf.sample(n=1).iloc[0]
    geom = selected_row.geometry
    if geom is None or geom.is_empty or geom.geom_type != 'Point':
        print("[WARNING] Invalid geometry. Skipping.")
        return

    x_coord, y_coord = geom.x, geom.y
    actual_depth = selected_row.get('Depth', None)

    print("\n[TEST] Random geometry from test river:")
    print(f"   Coordinates: ({x_coord:.3f}, {y_coord:.3f})")
    print(f"   Actual Depth: {actual_depth}")

    intensities = None
    for band_arrays, band_info in zip(list_of_band_data, list_of_band_info):
        if len(band_arrays) != num_bands_expected:
            continue

        feat_vector = []
        in_bounds = True
        for i, band_data in enumerate(band_arrays):
            transform, w, h = band_info[i]
            col, row_idx = ~transform * (x_coord, y_coord)
            row_idx = int(round(row_idx))
            col = int(round(col))

            if 0 <= row_idx < h and 0 <= col < w:
                avg_val = average_patch(band_data, row_idx, col, patch_size=patch_size)
                feat_vector.append(avg_val)
            else:
                in_bounds = False
                break

        if in_bounds and len(feat_vector) == num_bands_expected:
            intensities = feat_vector
            break

    if intensities is None:
        print("[WARNING] No valid raster coverage or mismatch.")
        return

    # Scale + Predict
    intensities_scaled = scaler.transform([intensities])[0]
    rf_pred = rf_model.predict([intensities_scaled])[0]
    xgb_pred = xgb_model.predict([intensities_scaled])[0]

    print(f"[RandomForest] Predicted Depth: {rf_pred:.4f}")
    print(f"[XGBoost]     Predicted Depth: {xgb_pred:.4f}")

    if actual_depth is not None:
        mae_rf = mean_absolute_error([actual_depth], [rf_pred])
        mae_xgb = mean_absolute_error([actual_depth], [xgb_pred])
        print(f"[RandomForest] MAE: {mae_rf:.4f}")
        print(f"[XGBoost]     MAE: {mae_xgb:.4f}")

        if naive_depth is not None:
            mae_naive = mean_absolute_error([actual_depth], [naive_depth])
            print(f"[BENCHMARK - AvgDepth]  Predicted Depth: {naive_depth:.4f} | MAE: {mae_naive:.4f}")

def main():
    if not check_files_exist():
        return

    # Use bigger patch_size (3 => 7x7 neighborhood, or 5 => 11x11 if you want to overfit more)
    patch_size = 20

    # 1) Decide which rivers to train on vs. test on
    train_rivers = ["musa", "susve", "verkne"]
    test_river = "jura"

    # 2) Load training data
    X_train, y_train = load_training_data(train_rivers, patch_size=patch_size)
    print(f"Training samples: {X_train.shape[0]} | Features/sample: {X_train.shape[1]}")

    if len(X_train) == 0:
        print("[ERROR] No training data found.")
        return

    # 3) Train models (overfit style)
    rf_model, xgb_model, scaler, naive_depth = train_models(X_train, y_train)

    # 4) Load test data
    combined_gdf, list_of_band_data, list_of_band_info = load_test_data(test_river, patch_size=patch_size)
    print(f"Test shapefile(s) have {len(combined_gdf)} valid geometries with Depth.")

    # 5) Predict random points multiple times
    for i in range(10):
        print(f"\n--- Random Prediction #{i+1} ---")
        predict_random_point(
            combined_gdf,
            list_of_band_data,
            list_of_band_info,
            rf_model,
            xgb_model,
            scaler,
            patch_size=patch_size,
            naive_depth=naive_depth,
            num_bands_expected=5
        )

    print("\nDone.")

if __name__ == "__main__":
    main()
