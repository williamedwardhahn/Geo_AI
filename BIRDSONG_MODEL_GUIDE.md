# FAU Geo/Climate Project Guide
## Modeled After the FAU Birdsong Project

This guide documents the FAU Birdsong project architecture and provides a detailed blueprint for building an equivalent educational curriculum for Geoscience and Climate Data.

---

## PART 1: THE BIRDSONG MODEL — WHAT EXISTS

### 1.1 Project Summary

The FAU Birdsong project is an educational data science curriculum created by **Rindy Anderson** (Biological Sciences) and **William Hahn** (Mathematics & Statistics) at Florida Atlantic University. It teaches computational analysis of bird vocalizations using Python, targeting students with zero programming experience.

**Core idea:** Use a compelling scientific domain (birdsong) to teach Python, data processing, visualization, and machine learning — entirely in the browser via Google Colab.

### 1.2 Directory Structure

```
FAU Birdsong/
├── README.md                          # Project documentation
├── index.html                         # Web landing page with Colab links
├── images/                            # Tutorial screenshots
│   ├── xeno_canto_1_homepage.png
│   ├── xeno_canto_2_results.png
│   └── xeno_canto_3_download.png
├── archive/                           # Previous notebook versions
│   ├── birdsong_data_ai_notebook_2025.py
│   ├── birdsong_data_ai_notebook_2026.py
│   └── Birdsong_Data_AI_Notebook_2026.ipynb
├── Audio Datasets (5 species, 98 WAV files):
│   ├── bachmans_sparrow/              # 19 recordings
│   ├── northern_cardinal/             # 9 recordings
│   ├── song_sparrow/                  # 15 recordings
│   ├── swamp_sparrow/                 # 10 recordings
│   └── zebra_finch/                   # 45 recordings
└── Notebooks (6 sequential lessons):
    ├── Notebook_0_Python_Basics.ipynb
    ├── Notebook_1_Sound_as_Data.ipynb
    ├── Notebook_2_Birdsong_Explorer.ipynb
    ├── Notebook_3_Clustering.ipynb
    ├── Notebook_4_Web_Audio.ipynb
    └── Notebook_5_Your_Own_Data.ipynb
```

### 1.3 The 6-Notebook Curriculum

| # | Notebook | What It Teaches | Key Libraries |
|---|----------|----------------|---------------|
| 0 | Python Basics | Variables, lists, loops, functions, basic plots | matplotlib |
| 1 | Sound as Data | Waveforms, spectrograms, sample rates, audio manipulation | librosa, numpy |
| 2 | Birdsong Explorer | Visual pattern recognition across species, spectrogram montages | librosa, matplotlib |
| 3 | Clustering | MFCCs, autoencoders (PyTorch), 2D scatter plots, unsupervised ML | PyTorch, sklearn |
| 4 | Web Audio | HTTP downloads, fetching audio from xeno-canto.org, custom URLs | requests |
| 5 | Your Own Data | Google Drive integration, personal recordings, full ML pipeline | pydub, Google Colab |

### 1.4 Data Flow Architecture

```
Raw Audio (WAV files, 5 species)
        │
        ▼
Waveform Visualization (amplitude vs. time)
        │
        ▼
Spectrogram Generation (frequency vs. time, color = loudness)
        │
        ▼
MFCC Feature Extraction (13 coefficients → 26-dim vector per recording)
        │
        ▼
StandardScaler Preprocessing (normalize to mean=0, std=1)
        │
        ▼
Autoencoder Training (26D → 128 → 64 → 2D bottleneck → decode back)
        │
        ▼
2D Scatter Plot (color-coded by species, showing natural clusters)
```

### 1.5 Key Design Principles

1. **Zero installation** — everything runs in Google Colab
2. **Progressive complexity** — Python basics → data processing → visualization → ML
3. **Domain-first** — biology motivates every technical concept
4. **Dual representation** — always show data two ways (waveform + spectrogram, human + machine)
5. **Real data** — 98 field recordings, not toy examples
6. **Interactive experimentation** — hyperparameter tuning, mystery challenges
7. **Capstone project** — students bring their own recordings
8. **Professional landing page** — index.html with Colab links and documentation

### 1.6 Technology Stack

| Category | Birdsong |
|----------|----------|
| Data format | WAV audio files |
| Loading | librosa |
| Feature extraction | MFCCs (librosa.feature.mfcc) |
| Visualization | matplotlib, librosa.display |
| ML framework | PyTorch (autoencoder) |
| Preprocessing | sklearn (StandardScaler) |
| Web data | requests (HTTP downloads) |
| Platform | Google Colab + Google Drive |
| Landing page | Static HTML/CSS |

---

## PART 2: THE GEO/CLIMATE TRANSLATION — WHAT TO BUILD

### 2.1 Concept Mapping

The Geo/Climate project follows the **exact same pedagogical arc**, substituting climate/geospatial data for audio data:

| Birdsong Concept | Geo/Climate Equivalent |
|-----------------|----------------------|
| Audio recordings (WAV) | Climate time series (CSV/JSON from NOAA API) |
| Waveform (amplitude vs. time) | Temperature time series (temp vs. date) |
| Spectrogram (frequency vs. time) | Heatmap (day-of-year vs. year, color = temp) |
| Species (cardinal, sparrow...) | Stations/Cities (Miami, New York, Chicago...) |
| MFCC features | Climate features (monthly means, std, range, trend slope) |
| Autoencoder clustering | Autoencoder clustering of station climate profiles |
| xeno-canto.org | NOAA API (ncei.noaa.gov) |
| Personal recordings | Student-chosen stations or local weather data |
| librosa | pandas, xarray, cartopy |
| Listening to audio | Looking at maps |

### 2.2 Proposed Directory Structure

```
FAU Geo/
├── README.md                              # Project documentation
├── index.html                             # Web landing page with Colab links
├── images/                                # Tutorial screenshots
│   ├── noaa_api_1_homepage.png
│   ├── noaa_api_2_search.png
│   └── noaa_api_3_results.png
├── archive/                               # Previous versions
│   └── Full_FAU_Geoscience_AI.ipynb       # Current prototype (move here)
├── datasets/                              # Pre-fetched station data (CSV)
│   ├── miami_USW00012839.csv
│   ├── newyork_USW00094728.csv
│   ├── chicago_USW00094846.csv
│   ├── losangeles_USW00023174.csv
│   └── seattle_USW00024233.csv
└── Notebooks (6 sequential lessons):
    ├── Notebook_0_Python_Basics.ipynb
    ├── Notebook_1_Climate_as_Data.ipynb
    ├── Notebook_2_Climate_Explorer.ipynb
    ├── Notebook_3_Clustering.ipynb
    ├── Notebook_4_Satellite_and_Reanalysis.ipynb
    └── Notebook_5_Your_Own_Data.ipynb
```

### 2.3 The 6-Notebook Curriculum

---

#### Notebook 0: A Geoscientist's Field Notebook in Python
**Parallel to:** Birdsong Notebook 0 (Python Basics)

**Content:**
- Variables: `station = "Miami Intl Airport"`, `latitude = 25.793`, `elevation_m = 3`
- Lists: `stations = ["Miami", "New York", "Chicago", "LA", "Seattle"]`
- Dictionaries: `{"station": "Miami", "lat": 25.793, "lon": -80.290, "climate": "tropical"}`
- Loops: iterate over stations, print climate zones
- Functions: `celsius_to_fahrenheit()`, `compute_anomaly()`
- Plotting: bar chart of annual average temps across 5 cities
- Comparison: simple sine wave (seasonal cycle) vs. real messy temperature data

**Key parallel to Birdsong:** Just as Notebook 0 uses bird field observations as the context for Python basics, this uses weather station metadata and simple climate facts.

**Libraries:** matplotlib, numpy

**Outputs:**
- Bar chart: average annual temp by city
- Line plot: idealized seasonal cycle (sine wave) vs. real daily temps
- Print statements with station metadata

---

#### Notebook 1: Climate as Data
**Parallel to:** Birdsong Notebook 1 (Sound as Data)

**Content:**
- **What is a climate time series?** Daily temperature = a list of numbers, one per day
- **Fetch data from NOAA API** — requests.get() with parameters (station, date range, data types)
- **Load into pandas** — DataFrame with DATE, TMAX, TMIN, TAVG columns
- **Data cleaning** — pd.to_numeric, fillna with (TMAX+TMIN)/2, datetime conversion
- **Daily temperature plot** — the "waveform" equivalent (amplitude = temperature, x = date)
- **Seasonal decomposition** — separate trend + seasonal + residual components
- **Annual averaging** — groupby year, compute mean
- **The climate "spectrogram"** — seaborn heatmap (year x day-of-year, color = temperature)
- **Data manipulation** — compute anomalies (deviation from long-term mean), rolling means

**Key parallel to Birdsong:** Notebook 1 teaches "sound = a list of numbers sampled over time." This teaches "climate = a list of measurements sampled over time." The heatmap is the spectrogram equivalent — it reveals patterns invisible in a simple time series, just as a spectrogram reveals frequency structure invisible in a waveform.

**Libraries:** requests, pandas, matplotlib, seaborn, numpy

**Outputs:**
- Daily temperature time series (the "waveform")
- Year × day-of-year heatmap (the "spectrogram")
- Annual average trend line with regression
- Anomaly plot (deviation from mean)

---

#### Notebook 2: Climate Explorer
**Parallel to:** Birdsong Notebook 2 (Birdsong Explorer)

**Content:**
- **Load all 5 station datasets** (pre-fetched CSVs for reliability)
- **Individual station profiles** — daily plot + heatmap + monthly climatology for each city
- **Station montage** — grid of heatmaps (5 cities side by side), like spectrogram montages
- **Comparative plots** — overlay annual averages for all cities on one chart
- **Temperature range analysis** — TMAX - TMIN per station (continental vs. maritime climates)
- **Mystery station challenge** — show unlabeled heatmaps, students guess the city
- **Geographic context** — folium map with all 5 stations, popup with mean temp

**Key parallel to Birdsong:** Notebook 2 builds visual pattern recognition. Students learn to "read" a heatmap and identify climate signatures (tropical = uniformly warm, continental = wide seasonal swing) just as birdsong students learn to identify species from spectrogram shapes.

**Libraries:** pandas, matplotlib, seaborn, folium, numpy

**Outputs:**
- 5-station heatmap montage
- Overlay line chart of annual averages
- Interactive folium map
- Mystery station identification challenge

---

#### Notebook 3: Clustering Climate Stations with Machine Learning
**Parallel to:** Birdsong Notebook 3 (Clustering)

**Content:**
- **Feature extraction** — for each station, compute:
  - 12 monthly mean temperatures (the climate "MFCCs")
  - 12 monthly standard deviations
  - Annual range (max monthly mean - min monthly mean)
  - Linear trend slope (warming/cooling rate)
  - Total: ~26 features per station (matching Birdsong's 26 MFCC features)
- **Expand the dataset** — fetch 20-30 stations across the US (diverse climates)
- **StandardScaler preprocessing** — normalize all features
- **Autoencoder architecture** (identical to Birdsong):
  ```
  Encoder: 26 → 128 → 64 → 2 (bottleneck)
  Decoder: 2 → 64 → 128 → 26
  Loss: MSELoss, Optimizer: Adam, lr=0.0001, epochs=5000
  ```
- **Training dynamics** — loss curve, convergence discussion
- **2D scatter plot** — color by climate zone or region
- **Interpretation** — which stations cluster together? Does geography predict clustering?
- **Hyperparameter tuning** — students change lr, epochs, bottleneck size

**Key parallel to Birdsong:** This is the ML heart of both curricula. The Birdsong project compresses 26 MFCC features to 2D; this compresses 26 climate features to 2D. Both use the same autoencoder architecture, same loss function, same training loop. Students see that the *method* is domain-agnostic — the same neural network architecture works for audio and climate.

**Libraries:** PyTorch, sklearn, pandas, matplotlib, numpy

**Outputs:**
- Loss curve during training
- 2D scatter plot of 20-30 US stations, color-coded by climate zone
- Discussion: tight clusters = similar climates, overlap = transitional zones

---

#### Notebook 4: Satellite Imagery and Reanalysis Data
**Parallel to:** Birdsong Notebook 4 (Web Audio)

**Content:**
- **GOES satellite imagery** — display live/recent GEOCOLOR, AirMass, IR loops via NESDIS URLs
- **Multiple products** — GEOCOLOR, lightning (GLM), AirMass, Sandwich, Fire Temperature
- **NCEP/NCAR Reanalysis** — load gridded temperature data via xarray from NOAA PSL
- **Cartopy mapping** — contour plots of temperature fields over the US
- **Animated monthly maps** — matplotlib.animation for 12-month temperature cycle
- **Custom region selection** — students choose lat/lon bounds, time period
- **Comparison** — station point data vs. gridded reanalysis (sparse vs. dense)

**Key parallel to Birdsong:** Notebook 4 teaches students to fetch real data from the web. Birdsong fetches audio from xeno-canto; Geo fetches satellite imagery from NESDIS and gridded reanalysis from NOAA PSL. Both expand beyond the curated dataset to real-world, messy, live data sources.

**Libraries:** requests, xarray, cartopy, matplotlib, matplotlib.animation, IPython.display

**Outputs:**
- GOES satellite GIF loops (multiple products)
- Cartopy contour map of reanalysis temperature
- 12-frame animated monthly temperature map
- Student-selected custom region map

---

#### Notebook 5: Analyze Your Own Data
**Parallel to:** Birdsong Notebook 5 (Your Own Data)

**Content:**
- **Choose your own station(s)** — find NOAA station IDs via Climate Data Online
- **Google Drive integration** — save fetched data to Drive, or upload local CSV
- **Full pipeline on student data:**
  1. Fetch from NOAA API (or load CSV)
  2. Clean and explore (daily plots, heatmaps)
  3. Compute climate features (monthly means, stds, trends)
  4. Run autoencoder clustering with curated + student stations
  5. See where their station falls in the 2D embedding
- **Bonus: multi-variable analysis** — add precipitation, wind, or humidity
- **Bonus: compare decades** — 1980s climate profile vs. 2010s for same station

**Key parallel to Birdsong:** The capstone. Birdsong students record their own audio and cluster it with the curated dataset. Geo students pick their own weather station and see how it clusters with the 20-30 curated stations. Same personal investment, same "bring your own data" empowerment.

**Libraries:** requests, pandas, PyTorch, sklearn, matplotlib, Google Colab (drive.mount)

**Outputs:**
- Student-chosen station visualizations
- Updated 2D scatter plot including student's station
- Written reflection on where their station clusters and why

---

### 2.4 What Already Exists vs. What Needs to Be Built

The current `Full_FAU_Geoscience_AI.ipynb` is a **prototype** that covers material from several of the proposed notebooks but needs to be decomposed and expanded:

| Current Notebook Content | Maps To | Status |
|-------------------------|---------|--------|
| Import libraries, fetch NOAA data | Notebook 1 | Exists, needs polish |
| Load into pandas, clean data | Notebook 1 | Exists, needs polish |
| Daily temperature plot (Miami 2000) | Notebook 1 | Exists |
| Annual averages, regression line | Notebook 1 | Exists |
| Animated year-by-year build | Notebook 1 | Exists |
| Monthly climatology | Notebook 2 | Exists (single station) |
| Heatmap (year × day-of-year) | Notebook 1/2 | Exists (single station) |
| Folium map | Notebook 2 | Exists (basic) |
| Miami vs. NY comparison | Notebook 2 | Exists (2 stations only) |
| 5-year rolling mean | Notebook 1/2 | Exists |
| Cartopy station map | Notebook 2/4 | Exists |
| Synthetic temp contour map | Notebook 4 | Exists (fake data) |
| NCEP/NCAR Reanalysis map | Notebook 4 | Exists |
| Animated monthly reanalysis | Notebook 4 | Exists |
| GOES satellite GIF display | Notebook 4 | Exists |
| Python basics (Notebook 0) | Notebook 0 | **MISSING** |
| Multi-station exploration/montage | Notebook 2 | **MISSING** |
| Mystery station challenge | Notebook 2 | **MISSING** |
| MFCC-equivalent climate features | Notebook 3 | **MISSING** |
| Autoencoder + clustering | Notebook 3 | **MISSING** |
| Hyperparameter tuning | Notebook 3 | **MISSING** |
| Google Drive integration | Notebook 5 | **MISSING** |
| Student's own station pipeline | Notebook 5 | **MISSING** |
| Pre-fetched CSV datasets | datasets/ | **MISSING** |
| index.html landing page | Root | **MISSING** |
| README.md | Root | **MISSING** |
| Tutorial screenshots | images/ | **MISSING** |

### 2.5 Datasets to Prepare

**5 Core Stations (diverse US climates):**

| City | Station ID | Climate Type | Approx. Annual Avg |
|------|-----------|-------------|-------------------|
| Miami, FL | USW00012839 | Tropical/Subtropical | ~25.5°C |
| New York, NY | USW00094728 | Humid Continental | ~13.0°C |
| Chicago, IL | USW00094846 | Continental | ~10.5°C |
| Los Angeles, CA | USW00023174 | Mediterranean | ~18.0°C |
| Seattle, WA | USW00024233 | Marine/Oceanic | ~11.5°C |

**Extended set (for Notebook 3 clustering, 20-30 stations):**
Add stations from: Phoenix, Denver, Anchorage, Honolulu, Minneapolis, Atlanta, Dallas, San Francisco, Boston, New Orleans, etc. — to cover all major US climate zones.

### 2.6 Landing Page (index.html)

Model directly on the Birdsong `index.html`:
- Earth-tone / blue-green color scheme (ocean/atmosphere colors)
- 6 notebook cards with "Open in Colab" buttons
- Getting Started section (3 steps)
- Dataset table (5 stations with metadata)
- Collaborator info and FAU branding
- Responsive design, custom fonts

### 2.7 Implementation Priority

**Phase 1 — Core Curriculum (match Birdsong parity):**
1. Notebook 0: Python Basics (geoscience context)
2. Notebook 1: Climate as Data (refactor from existing prototype)
3. Notebook 2: Climate Explorer (expand from existing prototype)
4. Notebook 3: Clustering (new — this is the ML heart)
5. Pre-fetch and save 5 core station CSVs

**Phase 2 — Advanced & Web:**
6. Notebook 4: Satellite & Reanalysis (refactor from existing prototype)
7. Notebook 5: Your Own Data (new — capstone)
8. Fetch extended station set for clustering

**Phase 3 — Polish & Deploy:**
9. README.md
10. index.html landing page
11. Tutorial screenshots
12. Move current prototype to archive/
13. Push to GitHub, verify Colab links

---

## PART 3: KEY TECHNICAL DETAILS

### 3.1 NOAA API Pattern (reuse throughout)

```python
import requests
import pandas as pd

url = "https://www.ncei.noaa.gov/access/services/data/v1"
params = {
    "dataset": "daily-summaries",
    "stations": "USW00012839",        # Station ID
    "startDate": "2000-01-01",
    "endDate": "2024-12-31",
    "dataTypes": "TMAX,TMIN,TAVG",
    "units": "metric",
    "format": "json"
}
r = requests.get(url, params=params, timeout=60)
df = pd.DataFrame(r.json())
```

### 3.2 Climate Feature Vector (parallel to MFCCs)

```python
def extract_climate_features(df):
    """Extract 26 features from a station's daily data (parallels 26 MFCC features)."""
    df["month"] = df["DATE"].dt.month
    monthly = df.groupby("month")["TAVG"]

    monthly_means = monthly.mean().values          # 12 features
    monthly_stds = monthly.std().values            # 12 features
    annual_range = monthly_means.max() - monthly_means.min()  # 1 feature
    trend_slope = np.polyfit(range(len(df)), df["TAVG"].values, 1)[0]  # 1 feature

    return np.concatenate([monthly_means, monthly_stds, [annual_range, trend_slope]])
    # Total: 26 features
```

### 3.3 Autoencoder Architecture (identical to Birdsong)

```python
import torch
import torch.nn as nn

class ClimateAutoencoder(nn.Module):
    def __init__(self, input_dim=26, bottleneck=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Training: MSELoss, Adam optimizer, lr=0.0001, 5000 epochs
```

### 3.4 Heatmap as Climate "Spectrogram"

```python
import seaborn as sns

df["dayofyear"] = df["DATE"].dt.dayofyear
pivot = df.pivot_table(index="year", columns="dayofyear", values="TAVG")

plt.figure(figsize=(15, 6))
sns.heatmap(pivot, cmap="coolwarm", cbar_kws={'label': '°C'})
plt.title("Daily Avg Temperature Heatmap")
plt.xlabel("Day of Year")
plt.ylabel("Year")
plt.show()
```

### 3.5 Cartopy Map Template

```python
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-130, -65, 20, 50])
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.STATES, linestyle=":")
ax.add_feature(cfeature.BORDERS)
# Add station points, contours, etc.
```

---

## PART 4: SIDE-BY-SIDE COMPARISON

| Dimension | Birdsong | Geo/Climate |
|-----------|----------|-------------|
| **Domain** | Bioacoustics | Climate science |
| **Raw data** | WAV audio files (98 recordings) | NOAA daily summaries (5+ stations, 25 years) |
| **"Waveform"** | Amplitude vs. time | Temperature vs. date |
| **"Spectrogram"** | Frequency vs. time (STFT) | Day-of-year vs. year heatmap |
| **Categories** | 5 bird species | 5 climate zones / cities |
| **Feature extraction** | 13 MFCCs → 26 features | 12 monthly means + 12 stds + range + trend → 26 features |
| **ML model** | PyTorch autoencoder (26→2D) | PyTorch autoencoder (26→2D) — identical |
| **Result** | Species cluster in 2D | Climate zones cluster in 2D |
| **Web data source** | xeno-canto.org | NOAA API, NESDIS satellite, NOAA PSL reanalysis |
| **Extra visualization** | Audio playback | Interactive maps (folium, cartopy) |
| **Capstone** | Record your own birds | Pick your own weather station |
| **Platform** | Google Colab | Google Colab |

---

## PART 5: SUMMARY

The FAU Birdsong project is a **6-notebook progressive curriculum** that takes students from zero Python experience to training neural networks on real scientific data. Its genius is in the pedagogical arc: basics → data representation → visual exploration → machine learning → real-world data → personal project.

The Geo/Climate project can follow this arc **exactly**, substituting:
- **Audio recordings** → **climate time series**
- **Spectrograms** → **heatmaps and maps**
- **Species identification** → **climate zone classification**
- **MFCCs** → **monthly climate statistics**
- **Same autoencoder** → **same autoencoder**

The existing `Full_FAU_Geoscience_AI.ipynb` prototype already contains building blocks for Notebooks 1, 2, and 4. The primary work remaining is:
1. **Notebook 0** (Python basics in geo context)
2. **Notebook 3** (ML clustering — the technical centerpiece)
3. **Notebook 5** (capstone with student data)
4. Decomposing and polishing existing content into separate notebooks
5. Building the landing page, README, and datasets

The result will be a **parallel curriculum** where the same ML techniques work on completely different scientific domains — reinforcing that data science methods are transferable across disciplines.
