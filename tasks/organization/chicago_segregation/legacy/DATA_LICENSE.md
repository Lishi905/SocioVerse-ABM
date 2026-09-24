# Data notice for the vendored Chicago data

The Apache License 2.0 of this repository covers the **source code only**. It does not
relicense the third-party data vendored in this directory. Each data source below keeps
its own terms; if you redistribute or build on these files, follow those terms and cite
the sources.

| file | contents |
|---|---|
| `processed_data/chicago_tracts.geojson` | 785 Chicago census tracts (2010 boundaries) with 2010 demographics, community-area socioeconomics and crime, amenity counts and HOLC redlining columns |
| `processed_data/chicago_tracts_2000.geojson` | the same tracts and schema with 2000 demographics (used by `init_mode="census_2000"`) |
| `config/archetypes.csv`, `config/archetype_tract_mapping.csv`, `config/persona_parameters.json` | household archetypes, their placement on tracts and behavioral parameters compiled by the authors from the census counts above and the published literature (aggregates only, no microdata) |

Column-level documentation is in [`processed_data/DATA_README.md`](processed_data/DATA_README.md).

## Sources and terms

### U.S. Census Bureau (public domain)
Race and ethnicity counts, tract boundaries and census socioeconomic indicators:
2010 Decennial Census Summary File 1; 2000 Decennial Census Summary Files 1 and 3;
TIGER/Line Shapefiles (2000 and 2010 vintages); American Community Survey 2008-2012
estimates (community-area level, via the City of Chicago Data Portal). These are works of
the U.S. federal government and are in the public domain in the United States.

*This product uses the Census Bureau Data API but is not endorsed or certified by the
Census Bureau.*

### City of Chicago Data Portal (Terms of Use, attribution)
Community-area boundaries and socioeconomic indicators, crime counts, CTA rail stations and
bus stops, Chicago Public Schools, police stations, fire stations and parks, obtained from
the City of Chicago Data Portal (https://data.cityofchicago.org). Used under the City of
Chicago's data Terms of Use (https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html).
The City of Chicago makes no claims as to the content, accuracy, timeliness or
completeness of these data, and they are provided "as is". Attribution: *Data provided by
the City of Chicago Data Portal.*

### OpenStreetMap (ODbL 1.0) — `religious_places`
The `religious_places` column counts places of worship retrieved from OpenStreetMap via the
Overpass API. Map data © OpenStreetMap contributors, available under the Open Database
License 1.0 (https://opendatacommons.org/licenses/odbl/1-0/); see
https://www.openstreetmap.org/copyright. This derived column is made available under the
ODbL 1.0: keep the attribution, and share any adapted version of it under the same license.

### Mapping Inequality census crosswalk (CC BY-NC) — HOLC fields, **non-commercial use only**
The HOLC redlining fields come from the Mapping Inequality census crosswalk
(https://github.com/americanpanorama/mapping-inequality-census-crosswalk, file
`MIv3Areas_2010TractCrosswalk.gpkg`), which combines the Mapping Inequality HOLC polygons
with census tracts and is licensed CC BY-NC (https://creativecommons.org/licenses/by-nc/4.0/).
This applies to:

- the `holc_grade`, `holc_score` and `holc_coverage` columns of both geojson files;
- the `holc_legacy_effect` block of `config/persona_parameters.json`, including
  `empirical_race_by_holc`.

These fields may be used for **non-commercial purposes only**, with attribution. Cite:

> Robert K. Nelson, LaDale Winling, et al., "Mapping Inequality: Redlining in New Deal
> America," ed. Robert K. Nelson and Edward L. Ayers, *American Panorama: An Atlas of
> United States History*, University of Richmond Digital Scholarship Lab,
> https://dsl.richmond.edu/panorama/redlining.

The HOLC grade is part of the tract description the LLM agents read, so removing these
fields changes LLM-mode results. For commercial use, drop the HOLC columns (the model then
falls back to its defaults) or obtain permission from the rights holders.

## Everything else
Apart from the HOLC-derived fields above, the authors' configuration tables in `config/`
add no restrictions beyond those of their public-domain census sources. The data are
provided "as is", without warranty of any kind.
