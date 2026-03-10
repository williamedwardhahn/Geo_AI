# FAU Geoscience & Climate Data with AI

An educational data science curriculum that teaches **Python, climate analysis, and machine learning** through real NOAA weather data. Designed for students with no programming experience.

A collaboration between **William Hahn** (Mathematics & Statistics) and **[Erik Johanson](https://www.geosciences.fau.edu/people/ejohanson.php)** (Geosciences) at Florida Atlantic University.

---

## Why Climate Data?

Climate literacy is essential in the 21st century. Understanding how to access, process, and analyze real climate data empowers students to engage with one of the most important scientific challenges of our time. This curriculum uses climate data as a compelling context to teach computational thinking and data science — the same approach used in our companion [Birdsong project](https://github.com/wjhahn/fau-birdsong) for bioacoustics.

---

## Notebooks

| # | Notebook | Description |
|---|---------|-------------|
| 0 | **Python Basics** | Variables, lists, loops, and functions — using weather stations as context |
| 1 | **Climate as Data** | Fetch NOAA data, daily time series, heatmaps, trends, and anomalies |
| 2 | **Climate Explorer** | Compare 5 US cities visually — heatmap montages, mystery station challenge |
| 3 | **Clustering** | Extract 26 climate features, train a PyTorch autoencoder, 2D embedding of 20 stations |
| 4 | **Satellite & Reanalysis** | GOES-19 imagery, NCEP/NCAR gridded data, animated cartopy maps |
| 5 | **Your Own Data** | Choose any US station, run the full pipeline, see where it clusters — capstone project |

All notebooks run in **Google Colab** — no installation required.

---

## Station Dataset

| City | Station ID | Climate Type |
|------|-----------|-------------|
| Miami, FL | USW00012839 | Tropical |
| New York, NY | USW00094728 | Humid Continental |
| Chicago, IL | USW00094846 | Continental |
| Los Angeles, CA | USW00023174 | Mediterranean |
| Seattle, WA | USW00024233 | Marine |

Notebook 3 expands to **20 stations** covering tropical, subtropical, continental, Mediterranean, marine, arid, semi-arid, and subarctic climates.

---

## Dependencies

All installed automatically in Colab:

- `requests` — HTTP data fetching
- `pandas` — Data manipulation
- `numpy` — Numerical computing
- `matplotlib` — Plotting and visualization
- `seaborn` — Statistical visualization
- `torch` (PyTorch) — Neural networks
- `scikit-learn` — Preprocessing
- `cartopy` — Map projections
- `xarray` — Gridded climate data
- `folium` — Interactive maps

---

## Getting Started

1. Click any notebook link on the [landing page](index.html) or open a `.ipynb` file
2. Click **"Open in Colab"** at the top
3. Run cells from top to bottom — each notebook is self-contained

---

## Data Sources

- **NOAA NCEI** — Daily weather station summaries via the [Access Data Service API](https://www.ncei.noaa.gov/access/services/)
- **NOAA NESDIS** — GOES-19 satellite imagery
- **NOAA PSL** — NCEP/NCAR Reanalysis gridded data

---

## Collaborators

- **William Hahn** — Department of Mathematics & Statistics, FAU
- **Erik Johanson** — Department of Geosciences, FAU

---

## License

Educational use. All data sourced from public NOAA APIs.
