# Task 1: Location-based Analysis (Restaurants)

## Overview
This notebook/script performs a full geographical analysis of the provided restaurant dataset:
- Clean latitude/longitude for trustworthy mapping
- Create an interactive map (Plotly Mapbox)
- Aggregate statistics by `City` and `Country Code`
- Find density hotspots (top localities/cities by count)
- Surface cross-country patterns across 6 target countries

## How to run
From inside `TASK1/`:
```bash
python task1_location_based_analysis.py
```
Outputs are written to `TASK1/output/`.

## Outputs produced
### Interactive visualizations (HTML)
- `output/restaurant_map_rating_votes.html`
  - Scatter map colored by `Aggregate rating`, sized by `Votes`
- `output/hotspots_top_localities.html`
  - Bar chart of top localities by restaurant count
- `output/box_rating_by_country.html`
  - Rating distribution by country (box plot)
- `output/scatter_costz_vs_rating_by_country.html`
  - Normalized cost (`cost_z`) vs rating, colored by country

### Tables (CSV)
- `output/cleaned_restaurants.csv`
  - Dataset after coordinate cleaning
- `output/group_stats_by_city.csv`
- `output/group_stats_by_country_code.csv`
- `output/hotspots_top_localities.csv`
- `output/hotspots_top_cities.csv`
- `output/cross_country_summary.csv`

## Data cleaning decisions (important for geospatial trust)
Before plotting any map patterns, the script filters out coordinates that can create misleading “fake hotspots”:
1. Drops missing lat/lon values (`NaN`) if they exist
2. Drops out-of-range values (lat outside `[-90, 90]`, lon outside `[-180, 180]`)
3. Drops rows where `(Latitude, Longitude) == (0, 0)`

This dataset is especially affected by rule (3): a non-trivial number of records share `0,0` coordinates, which should not be treated as real restaurant locations.

## Data quality caveats to mention in your portfolio
When interpreting cross-country price patterns:
- `Average Cost for two` is numeric but not currency-normalized across countries.
- The script compares price using a per-country normalization (`cost_z`) to reduce bias from currency scale differences.

When interpreting density hotspots:
- Some country groups have small sample sizes (few restaurants), so hotspot/rating metrics for those countries are more sensitive to individual points.

## Suggested portfolio story (what to write)
Use the results to answer “why geography matters” with at least 3–4 *data-backed* insights, such as:
- Which country shows the strongest vs weakest relationship between price and rating?
- Where do high-rated restaurants cluster within the densest country?
- Do user popularity signals (`Votes`) align with ratings similarly across countries?

## Notes
Country code mapping used in the script:
- `1` → India
- `30` → Brazil
- `14` → Australia
- `162` → Philippines
- `184` → Singapore
- `216` → United States

