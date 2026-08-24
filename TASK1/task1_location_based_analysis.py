"""
Task 1 (Location-based Analysis) - Cognifyz ML Internship
----------------------------------------------------------
Geographical analysis pipeline for a Zomato-like restaurant dataset.

What this script produces (in TASK1/output/):
1) Interactive map (Plotly Scatter Mapbox) colored by rating and sized by votes
2) Group-by tables:
   - stats by City
   - stats by Country Code
3) Hotspots (top localities/cities by restaurant count)
4) Cross-country comparisons (plots + summary stats) for:
   Philippines, Brazil, US, Australia, India, Singapore

Run:
  python task1_location_based_analysis.py
Optionally:
  python task1_location_based_analysis.py --countries India Brazil ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px


def parse_cuisines(value: object) -> List[str]:
    """Split the `Cuisines` field into a list of cuisine tokens."""
    if pd.isna(value):
        return []
    parts = [p.strip() for p in str(value).split(",")]
    return [p for p in parts if p]


def clean_lat_lon(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean/validate lat-long for geographic analysis.

    Data-quality rules used here:
    - Drop missing lat/lon (not expected in this dataset, but good practice)
    - Drop out-of-range values (lat outside [-90, 90], lon outside [-180, 180])
    - Drop the "default placeholder" coordinate (0, 0), which often indicates
      missing geo data and will create a misleading hotspot.
    """

    df = df.copy()

    # Coerce to numeric; if the CSV had "bad strings", we'd convert them to NaN.
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    coord_missing = df["Latitude"].isna() | df["Longitude"].isna()
    coord_out_of_range = (
        (df["Latitude"] < -90)
        | (df["Latitude"] > 90)
        | (df["Longitude"] < -180)
        | (df["Longitude"] > 180)
    )
    coord_zero_zero = (df["Latitude"] == 0) & (df["Longitude"] == 0)

    drop_mask = coord_missing | coord_out_of_range | coord_zero_zero
    stats = {
        "rows_before": int(len(df)),
        "coord_missing": int(coord_missing.sum()),
        "coord_out_of_range": int(coord_out_of_range.sum()),
        "coord_zero_zero": int(coord_zero_zero.sum()),
        "rows_after": int((~drop_mask).sum()),
        "rows_dropped": int(drop_mask.sum()),
    }

    df_clean = df.loc[~drop_mask].copy()

    # Use shorter column names downstream.
    df_clean = df_clean.rename(columns={"Latitude": "lat", "Longitude": "lon"})
    return df_clean, stats


def compute_group_stats(
    df: pd.DataFrame, group_cols: List[str]
) -> pd.DataFrame:
    """
    Compute:
      - average rating
      - average cost for two
      - restaurant count
      - cuisine diversity (unique cuisines count in the group)
    """
    tmp = df.copy()
    tmp["cuisine_list"] = tmp["Cuisines"].apply(parse_cuisines)

    base = (
        tmp.groupby(group_cols)
        .agg(
            avg_rating=("Aggregate rating", "mean"),
            avg_cost_for_two=("Average Cost for two", "mean"),
            restaurant_count=("Restaurant ID", "size"),
        )
        .reset_index()
    )

    # Cuisine diversity: count unique cuisines across the group.
    exploded = tmp[group_cols + ["cuisine_list"]].explode("cuisine_list")
    exploded = exploded.dropna(subset=["cuisine_list"])
    diversity = (
        exploded.groupby(group_cols)["cuisine_list"]
        .nunique()
        .reset_index(name="cuisine_diversity")
    )

    return base.merge(diversity, on=group_cols, how="left")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to the restaurant CSV (defaults to ../Dataset .csv)",
    )
    args = parser.parse_args()

    task_dir = Path(__file__).resolve().parent  # .../TASK1
    repo_dir = task_dir.parent  # .../ML_INTERNSHIP
    dataset_path = Path(args.dataset) if args.dataset else (repo_dir / "Dataset .csv")

    output_dir = task_dir / "output"
    ensure_output_dir(output_dir)

    # ------------------------
    # Step 1: Load + cleaning
    # ------------------------
    df = pd.read_csv(dataset_path)

    # Basic numeric sanity checks; helps prevent subtle map/grouping bugs.
    for col in ["Aggregate rating", "Average Cost for two", "Votes", "Latitude", "Longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_clean, coord_stats = clean_lat_lon(df)

    print("\n[Step 1] Coordinate cleaning summary")
    for k, v in coord_stats.items():
        print(f"  {k}: {v}")

    # Save a cleaned dataset so you can reuse it in later tasks.
    cleaned_path = output_dir / "cleaned_restaurants.csv"
    df_clean.to_csv(cleaned_path, index=False)

    # ------------------------------
    # Step 2: Interactive map view
    # ------------------------------
    # Note: Plotly's Mapbox tiles may require a token in some environments.
    # We use the free "open-street-map" style to reduce friction.
    map_df = df_clean.copy()
    map_df["Restaurant Name"] = map_df["Restaurant Name"].astype(str)
    map_df["City"] = map_df["City"].astype(str)
    map_df["Locality"] = map_df["Locality"].astype(str)

    fig_map = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        color="Aggregate rating",
        size="Votes",
        hover_name="Restaurant Name",
        hover_data={
            "City": True,
            "Locality": True,
            "Cuisines": True,
            "Aggregate rating": True,
            "Votes": True,
            "Average Cost for two": True,
        },
        zoom=1.5,
        height=750,
        color_continuous_scale="Viridis",
        size_max=18,
        title="Restaurant Distribution (colored by rating, sized by votes)",
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin={"l": 0, "r": 0, "t": 50, "b": 0})
    map_path = output_dir / "restaurant_map_rating_votes.html"
    fig_map.write_html(map_path)
    print(f"\n[Step 2] Saved interactive map: {map_path}")

    # ---------------------------------------
    # Step 3: Group stats (City + Country)
    # ---------------------------------------
    city_stats = compute_group_stats(df_clean, group_cols=["City"])
    city_stats_path = output_dir / "group_stats_by_city.csv"
    city_stats.sort_values("restaurant_count", ascending=False).to_csv(city_stats_path, index=False)
    print(f"[Step 3] Saved: {city_stats_path}")

    country_stats = compute_group_stats(df_clean, group_cols=["Country Code"])
    country_stats_path = output_dir / "group_stats_by_country_code.csv"
    country_stats.sort_values("restaurant_count", ascending=False).to_csv(country_stats_path, index=False)
    print(f"[Step 3] Saved: {country_stats_path}")

    # ---------------------------------------
    # Step 4: Density hotspots
    # ---------------------------------------
    # Prefer "Locality Verbose" if present; it tends to be more readable.
    loc_verbose_col = "Locality Verbose"
    if loc_verbose_col in df_clean.columns:
        df_clean["locality_for_hotspot"] = np.where(
            df_clean[loc_verbose_col].notna() & (df_clean[loc_verbose_col].astype(str).str.strip() != ""),
            df_clean[loc_verbose_col].astype(str),
            df_clean["Locality"].astype(str),
        )
    else:
        df_clean["locality_for_hotspot"] = df_clean["Locality"].astype(str)

    locality_hotspots = (
        df_clean.groupby(["Country Code", "City", "locality_for_hotspot"])
        .agg(restaurant_count=("Restaurant ID", "size"))
        .reset_index()
        .sort_values("restaurant_count", ascending=False)
        .head(20)
    )
    hotspots_path = output_dir / "hotspots_top_localities.csv"
    locality_hotspots.to_csv(hotspots_path, index=False)
    print(f"[Step 4] Saved: {hotspots_path}")

    # Simple bar plot for the top localities
    fig_hot = px.bar(
        locality_hotspots,
        x="restaurant_count",
        y="locality_for_hotspot",
        color="City",
        orientation="h",
        title="Top Localities by Restaurant Count",
        height=900,
    )
    fig_hot.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_hot_path = output_dir / "hotspots_top_localities.html"
    fig_hot.write_html(fig_hot_path)
    print(f"[Step 4] Saved: {fig_hot_path}")

    # City hotspots (top cities by count)
    city_hotspots = (
        df_clean.groupby(["Country Code", "City"])
        .agg(restaurant_count=("Restaurant ID", "size"))
        .reset_index()
        .sort_values("restaurant_count", ascending=False)
        .head(20)
    )
    city_hotspots_path = output_dir / "hotspots_top_cities.csv"
    city_hotspots.to_csv(city_hotspots_path, index=False)
    print(f"[Step 4] Saved: {city_hotspots_path}")

    # --------------------------------------------
    # Step 5: Cross-country patterns (6 countries)
    # --------------------------------------------
    # These codes are commonly used in Zomato-like datasets:
    # 1=India, 30=Brazil, 14=Australia, 162=Philippines, 184=Singapore, 216=United States
    code_to_country: Dict[int, str] = {
        1: "India",
        30: "Brazil",
        162: "Philippines",
        184: "Singapore",
        14: "Australia",
        216: "United States",
    }
    df_cc = df_clean[df_clean["Country Code"].isin(code_to_country.keys())].copy()
    df_cc["country"] = df_cc["Country Code"].map(code_to_country)

    # Normalize cost within each country (so "average cost" comparisons are less misleading
    # when currencies differ).
    def cost_zscore(s: pd.Series) -> pd.Series:
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / sd

    df_cc["cost_z"] = df_cc.groupby("country")["Average Cost for two"].transform(cost_zscore)

    # Per-country summary table
    def safe_corr(a: pd.Series, b: pd.Series) -> float:
        if len(a) < 2:
            return float("nan")
        return float(pd.DataFrame({"a": a, "b": b}).corr().iloc[0, 1])

    summary_rows = []
    for country, sub in df_cc.groupby("country"):
        summary_rows.append(
            {
                "country": country,
                "restaurant_count": len(sub),
                "avg_rating": float(sub["Aggregate rating"].mean()),
                "median_rating": float(sub["Aggregate rating"].median()),
                "share_rating_ge_4": float((sub["Aggregate rating"] >= 4).mean()),
                "avg_cost_for_two": float(sub["Average Cost for two"].mean()),
                "median_cost_for_two": float(sub["Average Cost for two"].median()),
                "corr_cost_rating": safe_corr(sub["Average Cost for two"], sub["Aggregate rating"]),
                "corr_votes_rating": safe_corr(sub["Votes"], sub["Aggregate rating"]),
            }
        )

    summary_cc = pd.DataFrame(summary_rows).sort_values("restaurant_count", ascending=False)
    summary_cc_path = output_dir / "cross_country_summary.csv"
    summary_cc.to_csv(summary_cc_path, index=False)
    print(f"[Step 5] Saved: {summary_cc_path}")

    # Plot 1: rating distribution across countries
    fig_box_rating = px.box(
        df_cc,
        x="country",
        y="Aggregate rating",
        color="country",
        title="Rating Distribution by Country (Box Plot)",
        height=500,
    )
    fig_box_rating_path = output_dir / "box_rating_by_country.html"
    fig_box_rating.write_html(fig_box_rating_path)

    # Plot 2: normalized price vs rating
    fig_scatter_price_rating = px.scatter(
        df_cc,
        x="cost_z",
        y="Aggregate rating",
        color="country",
        size="Votes",
        hover_data=["City", "Restaurant Name"],
        title="Normalized Cost vs Rating (cost_z per country)",
        height=650,
    )
    fig_scatter_price_rating.update_layout(
        xaxis_title="Cost (z-score within country)",
        yaxis_title="Aggregate rating",
    )
    fig_scatter_price_rating_path = output_dir / "scatter_costz_vs_rating_by_country.html"
    fig_scatter_price_rating.write_html(fig_scatter_price_rating_path)

    print(f"[Step 5] Saved: {fig_box_rating_path}")
    print(f"[Step 5] Saved: {fig_scatter_price_rating_path}")

    # --------------------------------------------
    # Step 6 (optional for code): surface insights
    # --------------------------------------------
    # We compute a small set of candidates you can directly quote in your portfolio.
    # (Write-up in README/notes is still recommended because you should interpret.)
    print("\n[Step 6] Candidate, data-backed insights (check summary CSV for exact values)")

    # 6.1 Country with strongest positive/negative cost-rating relationship
    cc = summary_cc.dropna(subset=["corr_cost_rating"]).copy()
    if len(cc) >= 2:
        best = cc.sort_values("corr_cost_rating", ascending=False).head(1).iloc[0]
        worst = cc.sort_values("corr_cost_rating", ascending=True).head(1).iloc[0]
        print(
            f"  Strongest positive cost-to-rating correlation: {best['country']} "
            f"(corr={best['corr_cost_rating']:.3f})"
        )
        print(
            f"  Strongest negative cost-to-rating correlation: {worst['country']} "
            f"(corr={worst['corr_cost_rating']:.3f})"
        )

    # 6.2 Highest rating>=4 share
    hi = summary_cc.sort_values("share_rating_ge_4", ascending=False).head(1).iloc[0]
    print(
        f"  Highest share of restaurants with rating>=4.0: {hi['country']} "
        f"(share={hi['share_rating_ge_4']:.3f})"
    )

    # 6.3 Best votes→rating alignment
    vr = summary_cc.dropna(subset=["corr_votes_rating"]).sort_values("corr_votes_rating", ascending=False).head(1).iloc[0]
    print(
        f"  Strongest votes-to-rating relationship: {vr['country']} "
        f"(corr={vr['corr_votes_rating']:.3f})"
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()

