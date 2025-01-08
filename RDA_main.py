import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

# --------------------------------------------------------------------------
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
# HELPER: gather multiple sub-pixels around (row_center, col_center)
# --------------------------------------------------------------------------
def collect_local_pixels(band_arrays, row_center, col_center, radius=300, n_samples=200):
    """
    Collect up to n_samples random sub-pixels in a (2*radius+1) x (2*radius+1) box.
    Each sub-pixel is [band_count] intensities, appended to the results list.
    """
    results = []
    if not band_arrays:
        return results

    rows, cols = band_arrays[0].shape

    rmin = max(0, row_center - radius)
    rmax = min(rows, row_center + radius + 1)
    cmin = max(0, col_center - radius)
    cmax = min(cols, col_center + radius + 1)

    box_height = rmax - rmin
    box_width = cmax - cmin
    total_box_pixels = box_height * box_width

    # If region smaller than n_samples, just gather all
    if total_box_pixels <= n_samples:
        sub_indices = [
            (r, c)
            for r in range(rmin, rmax)
            for c in range(cmin, cmax)
        ]
    else:
        # random sample
        sub_indices = []
        for _ in range(n_samples):
            rr = random.randint(rmin, rmax - 1)
            cc = random.randint(cmin, cmax - 1)
            sub_indices.append((rr, cc))

    # For each sub-pixel
    for (rr, cc) in sub_indices:
        intensities = []
        out_of_bounds = False
        for band_data in band_arrays:
            if 0 <= rr < band_data.shape[0] and 0 <= cc < band_data.shape[1]:
                intensities.append(float(band_data[rr, cc]))
            else:
                out_of_bounds = True
                break
        if (not out_of_bounds) and len(intensities) == len(band_arrays):
            results.append(intensities)

    return results

# --------------------------------------------------------------------------
# HELPER 2: AVERAGE A large NEIGHBORHOOD (only used if needed)
# --------------------------------------------------------------------------
def average_patch(band_data, row, col, patch_size=3):
    rows, cols = band_data.shape
    rmin = max(0, row - patch_size)
    rmax = min(rows, row + patch_size + 1)
    cmin = max(0, col - patch_size)
    cmax = min(cols, col + patch_size + 1)

    patch = band_data[rmin:rmax, cmin:cmax]
    return float(np.mean(patch))

# --------------------------------------------------------------------------
# HELPER 3: For optional debugging (display 500x500)
# --------------------------------------------------------------------------
def display_cropped_bands(
    band_arrays,
    row_cols,
    radius=300,
    region_size=500
):
    """
    Show each band's image but only a 500x500 area around the center pixel.
    We'll just place a red dot at the center. The radius-based approach
    means we don't necessarily show the sub-pixel boxes. This is for debugging.
    """
    import matplotlib.patches as patches

    num_bands = len(band_arrays)
    fig, axs = plt.subplots(1, num_bands, figsize=(15, 5))

    for i, band_data in enumerate(band_arrays):
        ax = axs[i] if num_bands > 1 else axs
        row_idx, col_idx = row_cols[i]
        H, W = band_data.shape

        half = region_size // 2
        row_start = max(0, row_idx - half)
        col_start = max(0, col_idx - half)
        row_end = min(H, row_start + region_size)
        col_end = min(W, col_start + region_size)
        row_start = max(0, row_end - region_size)
        col_start = max(0, col_end - region_size)

        cropped = band_data[row_start:row_end, col_start:col_end]
        ax.imshow(cropped, cmap='gray')

        local_row = row_idx - row_start
        local_col = col_idx - col_start

        # Red dot for center
        ax.scatter(local_col, local_row, c='red', s=30)

        ax.set_title(f"Band {i+1}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------
# TRAINING: multi-sub-pixel approach
# --------------------------------------------------------------------------
def load_training_data(
    train_rivers,
    radius=300,    # gather sub-pixels in 300-pixel radius
    n_samples=200  # # sub-pixels to sample
):
    global_X = []
    global_y = []

    for river_name in train_rivers:
        datasets = river_data.get(river_name, [])
        for dataset in datasets:
            shp_file = dataset["shp_file"]
            band_list = dataset["bands"]

            shp_path = os.path.join(BASE_DIR, f"data_{river_name}", shp_file)
            if not os.path.isfile(shp_path):
                continue

            gdf = gpd.read_file(shp_path)
            if "Depth" not in gdf.columns:
                continue
            valid_gdf = gdf[gdf["Depth"].notnull()]
            if valid_gdf.empty:
                continue

            # load bands
            band_arrays = []
            transforms = []
            widths = []
            heights = []
            for band_file in band_list:
                band_path = os.path.join(BASE_DIR, f"data_{river_name}", band_file)
                with rasterio.open(band_path) as src:
                    band_arrays.append(src.read(1))
                    transforms.append(src.transform)
                    widths.append(src.width)
                    heights.append(src.height)

            # For each geometry
            for _, row_data in valid_gdf.iterrows():
                geom = row_data.geometry
                depth_val = row_data["Depth"]
                if geom is None or geom.is_empty or geom.geom_type != 'Point':
                    continue

                x_coord, y_coord = geom.x, geom.y
                # transform => pixel
                transform_1 = transforms[0]
                w_1 = widths[0]
                h_1 = heights[0]

                col, row_idx = ~transform_1 * (x_coord, y_coord)
                row_idx = int(round(row_idx))
                col = int(round(col))

                # If in range
                if 0 <= row_idx < h_1 and 0 <= col < w_1:
                    local_pixels = collect_local_pixels(
                        band_arrays, row_idx, col,
                        radius=radius,
                        n_samples=n_samples
                    )
                    # each sub-pixel is appended
                    for feats in local_pixels:
                        global_X.append(feats)
                        global_y.append(depth_val)

    return np.array(global_X), np.array(global_y)

# --------------------------------------------------------------------------
# TEST LOADING
# --------------------------------------------------------------------------
def load_test_data(test_river):
    import pandas as pd

    combined_gdf = gpd.GeoDataFrame()
    list_of_band_data = []
    list_of_band_info = []

    if test_river not in river_data:
        return combined_gdf, list_of_band_data, list_of_band_info

    for dataset in river_data[test_river]:
        shp_file = dataset["shp_file"]
        band_list = dataset["bands"]

        shp_path = os.path.join(BASE_DIR, f"data_{test_river}", shp_file)
        if not os.path.isfile(shp_path):
            continue

        gdf = gpd.read_file(shp_path)
        if "Depth" not in gdf.columns:
            valid_gdf = gdf
        else:
            valid_gdf = gdf[gdf["Depth"].notnull()]

        if not valid_gdf.empty:
            combined_gdf = pd.concat([combined_gdf, valid_gdf], ignore_index=True)

        band_arrays = []
        band_info = []
        for band_file in band_list:
            band_path = os.path.join(BASE_DIR, f"data_{test_river}", band_file)
            with rasterio.open(band_path) as src:
                band_arrays.append(src.read(1))
                band_info.append((src.transform, src.width, src.height))
        list_of_band_data.append(band_arrays)
        list_of_band_info.append(band_info)

    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry="geometry", crs=gdf.crs)
    return combined_gdf, list_of_band_data, list_of_band_info

# --------------------------------------------------------------------------
# TRAIN
# --------------------------------------------------------------------------
def train_models(X, y, n_estimators=300, max_depth=20):
    naive_depth = np.mean(y)
    print(f"[BENCHMARK] Training set average depth = {naive_depth:.4f}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_scaled, y)

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=max_depth,
        learning_rate=0.005,
        n_estimators=n_estimators,
        tree_method='hist',   # try a more memory-friendly approach
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_scaled, y)

    return rf_model, xgb_model, scaler, naive_depth


# --------------------------------------------------------------------------
# PREDICT: gather sub-pixels in a 300 px radius, average predictions
# --------------------------------------------------------------------------
def predict_random_point_subpixels(
    combined_gdf,
    list_of_band_data,
    list_of_band_info,
    rf_model,
    xgb_model,
    scaler,
    radius=300,
    n_samples=200,
    naive_depth=None,
    num_bands_expected=5
):
    """
    1) Random geometry from test shapefile
    2) Gather multiple sub-pixels around that geometry
    3) Predict each sub-pixel's depth
    4) Average predictions for final result
    """
    from sklearn.metrics import mean_absolute_error

    if combined_gdf.empty:
        print("[ERROR] No valid Depth data for test.")
        return

    # pick random geometry
    selected_row = combined_gdf.sample(n=1).iloc[0]
    geom = selected_row.geometry
    if not geom or geom.is_empty or geom.geom_type != 'Point':
        print("[WARNING] Invalid geometry. Skipping.")
        return

    x_coord, y_coord = geom.x, geom.y
    actual_depth = selected_row["Depth"]
    print("\n[TEST: Sub-Pixel Approach] Coordinates: (%.3f, %.3f)" % (x_coord, y_coord))
    print("   Actual Depth:", actual_depth)

    # Identify coverage
    subpixel_features = []
    coverage_bands = None
    coverage_row_cols = None
    found_coverage = False

    for band_arrays, band_info in zip(list_of_band_data, list_of_band_info):
        if len(band_arrays) != num_bands_expected:
            continue

        # We'll do the transform from the first band
        transform_1, w_1, h_1 = band_info[0]
        col, row_idx = ~transform_1 * (x_coord, y_coord)
        row_idx = int(round(row_idx))
        col = int(round(col))

        if 0 <= row_idx < h_1 and 0 <= col < w_1:
            # gather sub-pixels from all bands
            # first, let's see if each band is big enough
            for i, bd in enumerate(band_arrays):
                if row_idx < 0 or row_idx >= bd.shape[0]:
                    break
                if col < 0 or col >= bd.shape[1]:
                    break
            else:
                # we have coverage
                # gather sub-pixels for all bands
                sub_pixels = collect_local_pixels(
                    band_arrays,
                    row_center=row_idx,
                    col_center=col,
                    radius=radius,
                    n_samples=n_samples
                )
                if sub_pixels:
                    # we store them
                    subpixel_features = sub_pixels
                    coverage_bands = band_arrays
                    coverage_row_cols = (row_idx, col)
                    found_coverage = True
                    break

    if not found_coverage or not subpixel_features:
        print("[WARNING] No valid coverage for sub-pixel approach.")
        return

    # Optional: display the area (just a single red dot at center)
    # For debugging
    # We'll pass row_cols as a single list of coords (one per band, same center).
    row_cols_list = []
    for i in range(num_bands_expected):
        row_cols_list.append((coverage_row_cols[0], coverage_row_cols[1]))

    display_cropped_bands(coverage_bands, row_cols_list, radius=radius)

    # Now sub-pixel_features is many lines of shape [n_bands].
    # We scale & predict each sub-pixel
    subpixel_features = np.array(subpixel_features)
    subpixel_scaled = scaler.transform(subpixel_features)

    preds_rf = rf_model.predict(subpixel_scaled)
    preds_xgb = xgb_model.predict(subpixel_scaled)

    # final = average
    final_rf = preds_rf.mean()
    final_xgb = preds_xgb.mean()

    print("[RANDOM FOREST] Sub-pixel predictions => final=%.4f" % final_rf)
    print("[XGBOOST]       Sub-pixel predictions => final=%.4f" % final_xgb)

    if actual_depth is not None:
        mae_rf = mean_absolute_error([actual_depth], [final_rf])
        mae_xgb = mean_absolute_error([actual_depth], [final_xgb])
        print("[RandomForest] MAE:", mae_rf)
        print("[XGBoost]     MAE:", mae_xgb)
        if naive_depth is not None:
            mae_naive = mean_absolute_error([actual_depth], [naive_depth])
            print("[BENCHMARK]   MAE (naive=%.3f): %.3f" % (naive_depth, mae_naive))

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    if not check_files_exist():
        return

    radius = 200      # smaller radius instead of 300
    n_samples = 100    # gather fewer sub-pixels per geometry

    train_rivers = ["musa", "jura", "verkne"]
    test_river = "susve"

    # load training with multi-sub-pixel approach but smaller
    X_train, y_train = load_training_data(
        train_rivers,
        radius=radius,
        n_samples=n_samples
    )
    print(f"Training samples: {X_train.shape[0]} | Features/sample: {X_train.shape[1]}")
    if len(X_train) == 0:
        print("[ERROR] No training data found.")
        return

    # Train with smaller n_estimators and a max_depth
    rf_model, xgb_model, scaler, naive_depth = train_models(X_train, y_train,
                                                           n_estimators=300,
                                                           max_depth=20)

    # Load test
    combined_gdf, list_of_band_data, list_of_band_info = load_test_data(test_river)
    print("Test shapefile(s) have", len(combined_gdf), "valid Depth geometries.")

    # do sub-pixel predictions
    for i in range(50):
        predict_random_point_subpixels(
            combined_gdf,
            list_of_band_data,
            list_of_band_info,
            rf_model,
            xgb_model,
            scaler,
            radius=radius,
            n_samples=n_samples,
            naive_depth=naive_depth,
            num_bands_expected=5
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
