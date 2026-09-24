# Processed Data Documentation

Chicago Racial Segregation ABM Simulation - vendored tract-level inputs

**Spatial scope**: Chicago city proper (77 Community Areas, 785 Census Tracts, 2010 boundaries)
**Coordinate system**: EPSG:4326 (WGS84; stored as `urn:ogc:def:crs:OGC:1.3:CRS84`)

These files were produced by the authors' data pipeline, which is not included in this
repository. Only the two files below are vendored. For the license terms of each source
see [`../DATA_LICENSE.md`](../DATA_LICENSE.md): the Apache-2.0 license of the code does
**not** cover these data, and the HOLC-derived columns are for non-commercial use only.

---

## Files

### 1. `chicago_tracts.geojson` (~3 MB)
**Master tract-level dataset** — the primary spatial input for the ABM (all `init_mode`s
except `census_2000`).

785 Census Tracts within Chicago city limits, each with:

| Column | Type | Description |
|--------|------|-------------|
| `GEOID10` | str | Census Tract FIPS code (11 digits: state+county+tract) |
| `community_area_num` | float | Chicago Community Area number (1-77) |
| `total_pop` | int | Total population (2010 Census) |
| `nh_white` | int | Non-Hispanic White population |
| `nh_black` | int | Non-Hispanic Black population |
| `nh_asian` | int | Non-Hispanic Asian population |
| `hispanic` | int | Hispanic/Latino population |
| `nh_other` | int | Other races (total minus above four) |
| `pct_nh_white` | float | % Non-Hispanic White |
| `pct_nh_black` | float | % Non-Hispanic Black |
| `pct_nh_asian` | float | % Non-Hispanic Asian |
| `pct_hispanic` | float | % Hispanic |
| `pct_nh_other` | float | % Other |
| `per_capita_income` | float | Per capita income (from ACS 2008-2012, CA-level) |
| `pct_below_poverty` | float | % households below poverty line (CA-level) |
| `pct_unemployed` | float | % aged 16+ unemployed (CA-level) |
| `hardship_index` | float | Composite hardship index 1-100 (CA-level) |
| `crime_count_2010` | int | Total crimes in 2010 (CA-level) |
| `cta_stations` | int | Number of CTA L stations within the tract |
| `cta_bus_stops` | int | Number of CTA bus stops within the tract |
| `schools` | int | Number of CPS schools within the tract |
| `religious_places` | int | Number of religious places (OpenStreetMap) within the tract |
| `police_stations` | int | Number of police stations within the tract |
| `fire_stations` | int | Number of fire stations within the tract |
| `parks` | int | Number of parks (centroid within tract) |
| `holc_grade` | str | Dominant HOLC redlining grade: A (Best) to D (Hazardous) |
| `holc_score` | float | Area-weighted HOLC score: 1.0 (A) to 4.0 (D) |
| `holc_coverage` | float | Fraction of tract covered by HOLC-mapped areas |
| `geometry` | Polygon | Census Tract boundary |

**Key statistics**:
- Total population: 2,654,858
- NH White: 840,769 (31.7%), NH Black: 851,576 (32.1%), Hispanic: 776,926 (29.3%), NH Asian: 141,935 (5.3%)
- 778 of 785 tracts have HOLC redlining scores

**Note**: Socioeconomic and crime data are at Community Area level and shared across all tracts within each CA.

### 2. `chicago_tracts_2000.geojson` (~3 MB)
The same 785 tracts on **2010 boundaries**, with **2000** demographics, used only by
`init_mode="census_2000"`. It has exactly the same schema as `chicago_tracts.geojson`, so
the model can swap it in directly.

- Race/ethnicity: Census 2000 SF1 table P008 (Hispanic/Latino origin by race), moved from
  2000-vintage tracts (TIGER/Line 2000) to 2010 tracts (TIGER/Line 2010) by area-weighted
  interpolation for tracts whose boundaries changed.
- Socioeconomics: Census 2000 SF3 (P082 per capita income, P087 poverty status, P043
  employment status, H020 occupants per room), interpolated the same way;
  `hardship_index` is a simplified composite recomputed from these indicators.
- Crime: community-area crime counts from 2005, the earliest reliable year available
  (the column keeps the name `crime_count_2010` for schema compatibility).
- Amenities (`cta_*`, `schools`, `religious_places`, `police_stations`, `fire_stations`,
  `parks`): reused from the 2010 file as a proxy.
- HOLC columns: identical to the 2010 file (historical grades do not change).

**Key statistics**: total population 2,848,471; NH White 890,843, NH Black 1,028,768,
Hispanic 752,017, NH Asian 121,042; 778 of 785 tracts have HOLC redlining scores.

---

## Data Sources

| Dataset | Source | Access |
|---------|--------|--------|
| Race/ethnicity (P005) | 2010 Decennial Census SF1 | Census Bureau API |
| Race/ethnicity, income, poverty, employment (2000 file) | 2000 Decennial Census SF1 / SF3 | Census Bureau API |
| Tract boundaries | TIGER/Line 2010 (and 2000 for interpolation) | census.gov direct download |
| Community Area boundaries | Chicago Data Portal (igwz-8jzy) | Socrata API |
| Socioeconomic indicators | Chicago Data Portal (kn9c-c2s2; ACS 2008-2012) | Socrata API |
| Crime data | Chicago Data Portal | Socrata API (2010 filter; 2005 for the 2000 file) |
| CTA L stations | Chicago Data Portal (vmyy-m9qj) | Socrata API |
| CTA bus stops | Chicago Data Portal (pxug-u72f) | Shapefile download |
| Schools | Chicago Data Portal (Chicago Public Schools) | Socrata API |
| Religious places | OpenStreetMap via Overpass API | Overpass turbo |
| Police stations | Chicago Data Portal (z8bn-74gv) | Socrata API |
| Fire stations | Chicago Data Portal (28km-gtjn) | Socrata API |
| Parks | Chicago Data Portal (ejsh-fztr) | GeoJSON export |
| HOLC redlining | Mapping Inequality census crosswalk (americanpanorama/mapping-inequality-census-crosswalk) | MIv3Areas_2010TractCrosswalk.gpkg |

---

## Usage in ABM

```python
import geopandas as gpd

# Load the master tract dataset
tracts = gpd.read_file("processed_data/chicago_tracts.geojson")

# Build spatial adjacency (Queen contiguity)
from libpysal.weights import Queen
w = Queen.from_dataframe(tracts)

# Each tract becomes a "cell" in the ABM environment
# Agent archetypes use race, income, and preferences
# HOLC scores encode historical neighborhood reputation
```
