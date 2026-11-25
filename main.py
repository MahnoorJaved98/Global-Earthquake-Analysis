
# ============================================================
# GLOBAL EARTHQUAKE ANALYSIS
# Complete Python Data Science Project
# Data source: USGS Earthquake Catalog
# Period: 2015-2025
# ============================================================

import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import folium
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from datetime import timedelta
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

# ============================================================
# PROJECT SETTINGS
# ============================================================
START_DATE = pd.Timestamp("2015-01-01", tz="UTC")
END_DATE = pd.Timestamp("2026-01-01", tz="UTC")

MIN_MAGNITUDE = 4.5

API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"
MAP_DIR = OUTPUT_DIR / "cartopy_maps"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True,exist_ok=True)

sns.set_theme(style="whitegrid")

session = requests.Session()
session.headers.update({"User-Agent": "GlobalEarthquakeAnalysis/1.0"})

# ============================================================
# FETCH DATA FROM USGS API
# ============================================================

def fetch_interval(start, end):
    print(f"Fetching: {start.date()} to {end.date()}")
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": MIN_MAGNITUDE,
        "orderby": "time-asc",
        "limit": 20000
    }
    response = session.get(API_URL, params=params, timeout=90)
    response.raise_for_status()
    features = response.json().get("features", [])

    if len(features) >= 20000:
        duration = end - start
        if duration <= timedelta(hours=1):
            raise RuntimeError(
                "Too many events in a one-hour interval. "
                "Use pagination or a higher magnitude threshold.")

        midpoint = start + duration / 2
        print("Large response. Splitting interval...")
        first_half = fetch_interval(start, midpoint)
        time.sleep(0.1)
        second_half = fetch_interval(midpoint, end)
        return first_half + second_half
    return features


def download_earthquake_data():
    print("\nDOWNLOADING GLOBAL EARTHQUAKE DATA")
    all_features = []
    current = START_DATE

    while current < END_DATE:
        next_month = min(current + pd.offsets.MonthBegin(1), END_DATE)
        features = fetch_interval(current, next_month)
        all_features.extend(features)
        print(f"Records collected: "f"{len(all_features):,}")
        current = next_month
        time.sleep(0.15)

    print("\nDownload completed!")
    return all_features

# ============================================================
# CONVERT GEOJSON INTO PANDAS DATAFRAME
# ============================================================
def create_dataframe(features):
    records = []

    for feature in features:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates")
        if not coordinates or len(coordinates) < 3:
            continue
        longitude = coordinates[0]
        latitude = coordinates[1]
        depth = coordinates[2]
        records.append({
            "id": feature.get("id"),
            "time": pd.to_datetime(properties.get("time"),unit="ms",
                utc=True,errors="coerce"),
            "latitude": latitude,
            "longitude": longitude,
            "depth_km": depth,
            "magnitude": properties.get("mag"),
            "place": properties.get("place"),
            "mag_type": properties.get("magType"),
            "status": properties.get("status"),
            "significance": properties.get("sig"),
            "tsunami": properties.get("tsunami"),
            "event_type": properties.get("type"),
            "felt": properties.get("felt"),
            "alert": properties.get("alert")})

    df = pd.DataFrame(records)
    print("\nOriginal dataframe shape:")
    print(df.shape)
    return df

# ============================================================
# CLEAN THE DATASET
# ============================================================

def clean_data(df):
    print("\nCLEANING DATASET")
    df = df.copy()
    df = df.drop_duplicates(subset="id")
    df = df[df["latitude"].between(-90, 90)]
    df = df[df["longitude"].between(-180, 180)]
    df = df[df["magnitude"] >= MIN_MAGNITUDE]
    numeric_columns = ["latitude", "longitude", "depth_km", "magnitude",
                       "significance"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce")
    df = df.dropna(subset=["time", "latitude", "longitude", "magnitude"])

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.to_period("M").astype(str)
    df["date"] = df["time"].dt.floor("D")

    df["depth_category"] = pd.cut(df["depth_km"], bins=[-np.inf, 70, 300, np.inf],
                                  labels=["Shallow", "Intermediate", "Deep"])
    df["grid_lat"] = (np.floor((df["latitude"] + 90) / 10) * 10 - 90)
    df["grid_lon"] = (np.floor((df["longitude"] + 180) / 10) * 10 - 180)
    df = df.sort_values("time").reset_index(drop=True)

    print("\nCleaned dataframe shape:")
    print(df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDataset information:")
    df.info()
    df.to_csv(OUTPUT_DIR / "earthquakes.csv", index=False)
    return df

# ============================================================
# GENERAL DATASET EXPLORATION
# ============================================================

def explore_data(df):
    print("GENERAL DATASET EXPLORATION")

    print("\nTotal earthquakes:")
    print(f"{len(df):,}")

    print("\nDate range:")
    print("Start:", df["time"].min())
    print("End:", df["time"].max())

    print("\nDataset dimensions:")
    print(df.shape)

    print("\nMagnitude statistics:")
    print(df["magnitude"].describe())

    print("\nDepth statistics:")
    print(df["depth_km"].describe())

    print("\nEarthquakes per year:")
    yearly_counts = (df["year"].value_counts().sort_index())
    print(yearly_counts)

    print("\nTen largest earthquakes:")
    largest_earthquakes = df.nlargest(10, "magnitude")[["time", "place", "magnitude", "depth_km"]]
    print(largest_earthquakes.to_string(index=False))

    print("\nEarthquakes by depth category:")
    depth_counts = (df["depth_category"].value_counts(dropna=False))
    print(depth_counts)

    print("\nGeneral exploration completed.")
    print(f"Charts saved in: {CHART_DIR}")

# ============================================================
# PLOTTING WITH CARTOPY
# ============================================================

def create_cartopy_maps(df):
    map_df = df.copy()
    numeric_columns = ["longitude","latitude","magnitude","depth_km"]
    for column in numeric_columns:
        map_df[column] = pd.to_numeric(map_df[column],errors="coerce")
    map_df = map_df.dropna(subset=["longitude","latitude","magnitude"])
    map_df = map_df[map_df["longitude"].between(-180, 180)
        & map_df["latitude"].between(-90, 90)].copy()
    MAP_CRS = ccrs.PlateCarree()
    print("\nRecords available for mapping:")
    print(f"{len(map_df):,}")
    def create_map(title):
        fig = plt.figure(figsize=(18, 10))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.LAND,facecolor="#E8E8E8")
        ax.add_feature(cfeature.OCEAN,facecolor="#DCEAF3")
        ax.add_feature(cfeature.COASTLINE,linewidth=0.6)
        ax.add_feature(cfeature.BORDERS,linewidth=0.4,linestyle=":")
        ax.gridlines(draw_labels=False,linewidth=0.4,color="gray",alpha=0.5,
        linestyle="--")
        ax.set_title(title,fontsize=20,fontweight="bold",pad=20)
        return fig, ax

    print("\nGenerating Map 1: Global Earthquake Density")
    fig, ax = create_map("Global Earthquake Density | 2015–2025")
    lon_bins = np.linspace(-180, 180, 361)
    lat_bins = np.linspace(-90, 90, 181)
    counts, lon_edges, lat_edges = np.histogram2d(map_df["longitude"],map_df["latitude"],bins=[lon_bins,lat_bins])
    counts = counts.T
    masked_counts = np.ma.masked_where(counts == 0,counts)
    maximum_count = max(2,int(counts.max()))
    mesh = ax.pcolormesh(lon_edges,lat_edges,masked_counts,transform=MAP_CRS,
        cmap="inferno",norm=LogNorm(vmin=1,vmax=maximum_count),shading="auto")
    plt.colorbar(mesh,ax=ax,orientation="horizontal",pad=0.06,shrink=0.65,
        label="Earthquake count per 1° grid cell (log scale)")
    plt.savefig(MAP_DIR / "01_global_density.png",dpi=300,bbox_inches="tight")
    plt.show()
    plt.close()

    print("\nGenerating Map 2: Earthquake Magnitudes")
    fig, ax = create_map("Global Earthquake Magnitudes | 2015–2025")
    sample = map_df.sample(n=min(20000, len(map_df)),random_state=42)
    scatter = ax.scatter(sample["longitude"],sample["latitude"],
        c=sample["magnitude"],s=np.clip(sample["magnitude"] ** 2,3,100),
        cmap="plasma",alpha=0.65,transform=MAP_CRS,edgecolors="none")
    plt.colorbar(scatter,ax=ax,orientation="horizontal",pad=0.06,shrink=0.65,
        label="Earthquake magnitude")
    plt.savefig(MAP_DIR / "02_magnitude_map.png",dpi=300,bbox_inches="tight")
    plt.show()
    plt.close()

    print("\nGenerating Map 3: Earthquake Depth")
    fig, ax = create_map("Earthquake Depth Distribution | 2015–2025")
    depth_df = map_df.dropna(subset=["depth_km"]).copy()
    depth_df["depth_category"] = pd.cut(depth_df["depth_km"],
        bins=[-np.inf,70,300,np.inf],labels=["Shallow","Intermediate","Deep"])
    depth_categories = {"Shallow": "#F28E2B","Intermediate": "#4E79A7",
        "Deep": "#7A5195"}
    depth_sample = depth_df.sample(n=min(50000, len(depth_df)),random_state=42)
    for category, color in depth_categories.items():
        subset = depth_sample[depth_sample["depth_category"] == category]
        total_category = (depth_df["depth_category"] == category).sum()
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            s=3,
            color=color,
            alpha=0.5,
            label=f"{category} ({total_category:,})",
            transform=MAP_CRS)
    ax.legend(loc="lower left",fontsize=11,frameon=True)
    plt.savefig(MAP_DIR / "03_depth_map.png",dpi=300,bbox_inches="tight")
    plt.show()
    plt.close()

    print("\nGenerating Map 4: Major Earthquakes")
    major = map_df[map_df["magnitude"] >= 6].copy()
    fig, ax = create_map("Major Earthquakes | Magnitude 6+ | 2015–2025")
    if not major.empty:
        scatter = ax.scatter(major["longitude"],major["latitude"],
            c=major["magnitude"],s=np.clip(major["magnitude"] ** 3,15,500),
            cmap="inferno",alpha=0.8,edgecolors="black",linewidths=0.25,
            transform=MAP_CRS)
        plt.colorbar(scatter,ax=ax,orientation="horizontal",pad=0.06,
            shrink=0.65,label="Earthquake magnitude")
    else:
        ax.text(0.5,0.5,"No magnitude 6+ earthquakes found",transform=ax.transAxes,
            ha="center",va="center")
    ax.text(0.02,0.04,f"Total events: {len(major):,}",transform=ax.transAxes,
        fontsize=12,bbox=dict(facecolor="white",alpha=0.8))
    plt.savefig(MAP_DIR / "04_major_earthquakes.png",dpi=300,bbox_inches="tight")
    plt.show()
    plt.close()

    print("\nGenerating Map 5: Regional Earthquake Concentration")
    fig, ax = create_map("Global Earthquake Concentration by Region")
    map_df["grid_lat_map"] = (np.floor((map_df["latitude"] + 90) / 10)
                              * 10 - 90).clip(-90, 80)
    map_df["grid_lon_map"] = (np.floor((map_df["longitude"] + 180) / 10)
                              * 10 - 180).clip(-180, 170)
    regional_counts = (map_df.groupby(["grid_lat_map", "grid_lon_map"])
                       .size().unstack(fill_value=0))
    lat_starts = np.arange(-90, 90, 10)
    lon_starts = np.arange(-180, 180, 10)
    regional_counts = regional_counts.reindex(index=lat_starts,
                                              columns=lon_starts, fill_value=0)
    regional_values = regional_counts.to_numpy()
    regional_values = np.ma.masked_where(regional_values == 0, regional_values)
    regional_maximum = max(2, int(regional_counts.to_numpy().max()))
    lat_edges_10 = np.arange(-90, 91, 10)
    lon_edges_10 = np.arange(-180, 181, 10)
    mesh = ax.pcolormesh(lon_edges_10, lat_edges_10, regional_values, transform=MAP_CRS, cmap="YlOrRd",
    norm=LogNorm(vmin=1, vmax=regional_maximum), shading="auto")
    plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, shrink=0.65,
                 label="Earthquake count per 10° grid cell (log scale)")
    plt.savefig(MAP_DIR / "05_regional_concentration.png", dpi=300,
                bbox_inches="tight")
    plt.show()
    plt.close()
    print("\nAll Cartopy maps generated successfully!")
    print(f"Maps saved in: {MAP_DIR.resolve()}")

# ============================================================
# ANALYZING TIME TRENDS
# ============================================================

def analyze_time_trends(df):
    print("\nANALYSING TIME TRENDS")
    monthly = (df.set_index("time").resample("MS").size()
               .reset_index(name="earthquake_count"))
    monthly.to_csv(OUTPUT_DIR / "monthly_trends.csv", index=False)
    yearly = (df.groupby("year").size().reset_index(name="earthquake_count"))
    yearly.to_csv(OUTPUT_DIR / "yearly_trends.csv", index=False)

    # Monthly trend chart.
    plt.figure(figsize=(15, 6))
    plt.plot(monthly["time"], monthly["earthquake_count"], marker="o")
    plt.title("Global Earthquake Frequency (2015-2025)", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Number of Earthquakes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "01_monthly_trends.png", dpi=300)
    plt.close()

    # Yearly trend chart.
    plt.figure(figsize=(11, 6))
    sns.barplot(data=yearly, x="year", y="earthquake_count")
    plt.title("Earthquakes Per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Earthquakes")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "02_yearly_trends.png", dpi=300)
    plt.close()
    print("\nYearly earthquake counts:")
    print(yearly)

# ============================================================
# MAGNITUDE ANALYSIS
# ============================================================

def analyze_magnitude(df):
    print("\nANALYSING EARTHQUAKE MAGNITUDES")
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x="magnitude", bins=40, kde=True)
    plt.title("Distribution of Earthquake Magnitudes")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "03_magnitude_distribution.png", dpi=300)
    plt.close()
    df["magnitude_category"] = pd.cut(df["magnitude"], bins=[2.5, 4.0, 5.0, 6.0,7.0, np.inf],
                                      labels=["Minor", "Light", "Moderate", "Strong", "Major"],
                                      include_lowest=True)
    magnitude_counts = (df["magnitude_category"].value_counts().sort_index())
    magnitude_counts.rename_axis("magnitude_category").reset_index(
        name="earthquake_count").to_csv(OUTPUT_DIR / "magnitude_categories.csv", index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=magnitude_counts.index, y=magnitude_counts.values)
    plt.title("Earthquakes by Magnitude Category")
    plt.xlabel("Magnitude Category")
    plt.ylabel("Number of Earthquakes")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "04_magnitude_categories.png", dpi=300)
    plt.close()
    print("\nMagnitude categories:")
    print(magnitude_counts)

# ============================================================
# DEPTH ANALYSIS
# ============================================================

def analyze_depth(df):
    print("\nANALYSING EARTHQUAKE DEPTH")
    depth_counts = (df["depth_category"].value_counts())
    depth_counts.to_csv(OUTPUT_DIR / "depth_categories.csv")
    plt.figure(figsize=(9, 6))
    sns.countplot(data=df, x="depth_category", order=["Shallow", "Intermediate",
                                                      "Deep"])
    plt.title("Earthquakes by Depth Category")
    plt.xlabel("Depth Category")
    plt.ylabel("Number of Earthquakes")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "05_depth_categories.png", dpi=300)
    plt.close()

    # Depth versus magnitude.
    plt.figure(figsize=(11, 6))
    sample = df.sample(min(15000, len(df)), random_state=42)
    sns.scatterplot(data=sample, x="depth_km", y="magnitude", alpha=0.4, s=20)
    plt.title("Earthquake Magnitude vs Depth")
    plt.xlabel("Depth (km)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "06_depth_vs_magnitude.png", dpi=300)
    plt.close()
    print("\nDepth category distribution:")
    print(depth_counts)

# --------------------------------------------------------
# INTERACTIVE GLOBAL MAP
# --------------------------------------------------------

def create_interactive_map(df):
    print("\nCREATING INTERACTIVE EARTHQUAKE MAP")
    sample = df.sample(min(5000, len(df)), random_state=42)
    earthquake_map = folium.Map(location=[0, 0], zoom_start=2, tiles="OpenStreetMap")
    for row in sample.itertuples():
        magnitude = row.magnitude
        if magnitude >= 6:
            color = "red"
        elif magnitude >= 5:
            color = "orange"
        elif magnitude >= 4:
            color = "blue"
        else:
            color = "green"
        popup = (
            f"Place: {row.place}<br>"
            f"Magnitude: {magnitude}<br>"
            f"Depth: {row.depth_km} km<br>"
            f"Date: {row.time}")
        folium.CircleMarker(
            location=[row.latitude, row.longitude], radius=max(2, magnitude),
            color=color, fill=True, fill_opacity=0.6, popup=popup
        ).add_to(earthquake_map)
    map_path = (OUTPUT_DIR / "earthquake_map.html")
    earthquake_map.save(str(map_path))
    print(f"Interactive map saved: {map_path}")

# --------------------------------------------------------
# GEOGRAPHIC CLUSTERING
# --------------------------------------------------------

def analyze_clusters(df):
    print("\nRUNNING GEOGRAPHIC CLUSTERING")
    # Use a reproducible sample to control memory usage.
    sample = df.sample(min(20000, len(df)), random_state=42).copy()
    # Convert coordinates to radians.
    coordinates = np.radians(sample[["latitude", "longitude"]].values)
    # Earth radius in kilometres.
    earth_radius = 6371.0088
    # Maximum distance between neighbouring points: 50 km.
    epsilon = 50 / earth_radius
    model = DBSCAN(eps=epsilon, min_samples=10, metric="haversine",
                   algorithm="ball_tree", n_jobs=-1)
    sample["cluster"] = model.fit_predict(coordinates)
    sample.to_csv(OUTPUT_DIR / "earthquake_clusters.csv", index=False)
    # Ignore noise points (-1).
    clusters = sample[sample["cluster"] != -1]
    if clusters.empty:
        print("No clusters found with these settings.")
        return
    cluster_summary = (clusters.groupby("cluster").agg(
        event_count=("id", "count"),
        average_latitude=("latitude", "mean"),
        average_longitude=("longitude", "mean"),
        average_magnitude=("magnitude", "mean"),
        maximum_magnitude=("magnitude", "max")
    ).reset_index().sort_values("event_count", ascending=False))
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    print("\nLargest geographic clusters:")
    print(cluster_summary.head(15))
    plt.figure(figsize=(13, 7))
    sns.barplot(data=cluster_summary.head(15), x="event_count", y="cluster")
    plt.title("Largest Geographic Earthquake Clusters")
    plt.xlabel("Number of Sampled Earthquakes")
    plt.ylabel("Cluster ID")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "08_geographic_clusters.png", dpi=300)
    plt.close()

# --------------------------------------------------------
# GLOBAL DAILY ANOMALY DETECTION
# --------------------------------------------------------

def detect_global_anomalies(df):
    print("\nDETECTING GLOBAL ACTIVITY ANOMALIES")
    daily = (df.groupby("date").size().rename("earthquake_count"))
    # Include days with zero recorded earthquakes.
    all_dates = pd.date_range(START_DATE, END_DATE - pd.Timedelta(days=1),
                              freq="D")
    daily = daily.reindex(all_dates, fill_value=0)
    daily.index.name = "date"
    # Use the previous 90 days as the baseline.
    # shift(1) prevents today's activity from
    # influencing its own baseline.
    daily_df = daily.to_frame()
    daily_df["rolling_mean"] = (daily_df["earthquake_count"].shift(1)
                                .rolling(window=90, min_periods=30).mean())
    daily_df["rolling_std"] = (daily_df["earthquake_count"].shift(1)
                               .rolling(window=90, min_periods=30).std())
    daily_df["z_score"] = (
                                  daily_df["earthquake_count"] - daily_df["rolling_mean"]
                          ) / daily_df["rolling_std"].replace(0, np.nan)
    daily_df["is_anomaly"] = ((daily_df["z_score"] >= 3)
                              & (daily_df["earthquake_count"] >= 5))
    daily_df.to_csv(OUTPUT_DIR / "global_daily_activity.csv")
    anomalies = daily_df[daily_df["is_anomaly"]].copy()
    anomalies = anomalies.sort_values("z_score", ascending=False)
    anomalies.to_csv(OUTPUT_DIR / "global_anomalies.csv")
    print("\nMost unusual global activity days:")
    print(anomalies.head(20))
    # Plot daily counts and rolling baseline.
    plt.figure(figsize=(15, 7))
    plt.plot(daily_df.index, daily_df["earthquake_count"], alpha=0.5,
             label="Daily earthquake count")
    plt.plot(daily_df.index, daily_df["rolling_mean"], linewidth=2,
             label="Previous 90-day average")
    plt.scatter(anomalies.index, anomalies["earthquake_count"],
                color="red", label="Unusual activity", zorder=5)
    plt.title("Detecting Unusual Global Earthquake Activity")
    plt.xlabel("Date")
    plt.ylabel("Earthquake Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "09_global_anomalies.png", dpi=300)
    plt.close()

# --------------------------------------------------------
# REGIONAL ANOMALY DETECTION
# --------------------------------------------------------

def detect_regional_anomalies(df):
    print("\nDETECTING REGIONAL ANOMALIES")
    daily_regions = (df.groupby(["grid_lat", "grid_lon", "date"]
                                ).size().reset_index(name="earthquake_count"))
    all_dates = pd.date_range(START_DATE, END_DATE - pd.Timedelta(days=1),
                              freq="D")
    anomaly_results = []
    # Analyse each 10-degree geographic cell.
    for (grid_lat, grid_lon), group in daily_regions.groupby(
            ["grid_lat", "grid_lon"]):
        series = (group.set_index("date")["earthquake_count"].reindex(
            all_dates, fill_value=0))
        rolling_mean = (series.shift(1).rolling(90, min_periods=30).mean())
        rolling_std = (series.shift(1).rolling(90, min_periods=30)
                       .std())
        z_score = (series - rolling_mean) / rolling_std.replace(0, np.nan)
        # Require at least five events and a Z-score
        # of three or more.
        unusual = ((z_score >= 3) & (series >= 5))
        for date in series.index[unusual]:
            anomaly_results.append({
                "date": date,
                "grid_lat": grid_lat,
                "grid_lon": grid_lon,
                "earthquake_count": series.loc[date],
                "historical_average": rolling_mean.loc[date],
                "z_score": z_score.loc[date]
            })
    anomalies = pd.DataFrame(anomaly_results)
    if not anomalies.empty:
        anomalies = anomalies.sort_values("z_score", ascending=False)
    anomalies.to_csv(OUTPUT_DIR / "regional_anomalies.csv", index=False)
    print("\nRegional anomalies:")
    print(anomalies.head(20))

# --------------------------------------------------------
# CORRELATION ANALYSIS
# --------------------------------------------------------

def analyze_correlations(df):
    print("\nANALYSING CORRELATIONS")
    columns = ["magnitude", "depth_km", "significance", "felt"]
    correlation = df[columns].corr(method="pearson")
    correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    plt.figure(figsize=(9, 7))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Earthquake Variable Correlations")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "10_correlation_matrix.png", dpi=300)
    plt.close()
    print(correlation)

# --------------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------------

def generate_report(df):
    print("\nGENERATING FINAL REPORT")
    report = {"Total Earthquakes": len(df),
              "First Recorded Date": str(df["time"].min()),
              "Last Recorded Date": str(df["time"].max()),
              "Average Magnitude": round(df["magnitude"].mean(), 2),
              "Maximum Magnitude": round(df["magnitude"].max(), 2),
              "Average Depth (km)": round(df["depth_km"].mean(), 2),
              "Number of Geographic Cells": df[["grid_lat", "grid_lon"]
              ].drop_duplicates().shape[0],
              "Number of Years": df["year"].nunique()}
    report_df = pd.DataFrame(list(report.items()), columns=["Metric",
                                                            "Value"])
    report_df.to_csv(OUTPUT_DIR / "final_report.csv", index=False)
    print("\n" + "=" * 55)
    print("FINAL PROJECT REPORT")
    print("=" * 55)
    for metric, value in report.items():
        print(f"{metric}: {value}")

# --------------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------------

def main():
    print("GLOBAL EARTHQUAKE DATA ANALYSIS")
    # STEP 1: Download data.
    features = download_earthquake_data()
    # STEP 2: Convert GeoJSON to DataFrame.
    df = create_dataframe(features)
    # STEP 3: Clean data.
    df = clean_data(df)
    # STEP 4: Explore dataset.
    explore_data(df)
    create_cartopy_maps(df)
    # STEP 5: Time-series analysis.
    analyze_time_trends(df)
    # STEP 6: Magnitude analysis.
    analyze_magnitude(df)
    # STEP 7: Depth analysis.
    analyze_depth(df)
    # STEP 8: Geographic analysis.
    analyze_geography(df)
    # STEP 9: Interactive map.
    create_interactive_map(df)
    # STEP 10: Geographic clustering.
    analyze_clusters(df)
    # STEP 11: Global anomalies.
    detect_global_anomalies(df)
    # STEP 12: Regional anomalies.
    detect_regional_anomalies(df)
    # STEP 13: Correlations.
    analyze_correlations(df)
    # STEP 14: Final report.
    generate_report(df)
    print("\n")
    print("=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
