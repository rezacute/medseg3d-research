# Dataset Guide

This document details the datasets used in the **QRC-EV** project, including their content, sources, and how to access them.

## Primary Datasets (EV Charging Sessions)

These datasets contain the core charging session data used for training and evaluating the forecasting models.

| Dataset | Description | Access Method | Status in Downloader |
| :--- | :--- | :--- | :--- |
| **ACN-Data (Caltech)** | Over 30,000 EV charging sessions from Caltech's Adaptive Charging Network (2018–present). High-resolution power data. | **API**: Requires a free API token. | **Supported** (requires `--acn-token` arg). |
| **UrbanEV (Shenzhen)** | Public charging station data from Shenzhen, China (Sep 2022–Feb 2023). Includes 24,798 piles across 1,682 stations. | **Direct Download**: GitHub repository. | **Supported** (Automatic). |
| **Palo Alto Open Data** | Historical EV charging station usage from Palo Alto (Jul 2011–Dec 2020). Contains ~260,000 sessions. | **Direct Download**: Junar Open Data Portal. | **Supported** (Automatic). |

### 1. ACN-Data (Caltech)
- **Source**: [https://ev.caltech.edu/dataset](https://ev.caltech.edu/dataset)
- **Content**: Session ID, start/end time, energy delivered, user ID, station ID.
- **Usage**:
  1. Register for an API token at the source URL.
  2. Run: `python scripts/download_data.py --datasets acn --acn-token YOUR_TOKEN`

### 2. UrbanEV (Shenzhen)
- **Source**: [https://github.com/IntelligentSystemsLab/UrbanEV](https://github.com/IntelligentSystemsLab/UrbanEV)
- **Content**:
  - `volume-11kW.csv`: Charging volume.
  - `occupancy.csv`: Station occupancy rates.
  - `e_price.csv` & `s_price.csv`: Electricity and service prices.
  - `inf.csv`: Infrastructure metadata.
- **Usage**: Run `python scripts/download_data.py --datasets urban`

### 3. Palo Alto Open Data
- **Source**: [https://data.paloalto.gov/dataviews/257812/ELECT-VEHIC-CHARG-STATI-83602/](https://data.paloalto.gov/dataviews/257812/ELECT-VEHIC-CHARG-STATI-83602/)
- **Content**: Charging station usage history, including plug-in time, energy consumed, and fee.
- **Usage**: Run `python scripts/download_data.py --datasets paloalto`

---

## Exogenous Data (Contextual Features)

These datasets provide external context such as EV adoption rates and grid pricing, which can be encoded into the quantum reservoir to improve forecasting accuracy.

| Dataset | Description | Access Method | Status in Downloader |
| :--- | :--- | :--- | :--- |
| **Argonne Monthly EV Sales** | US national monthly sales for light-duty electric vehicles. | **Direct Download**: AFDC Mirror. | **Supported** (Automatic). |
| **IEA Global EV Data** | Global historical data on EV stock, sales, and charging infrastructure. | **Manual**: IEA website. | **Skipped** (Instructions provided). |
| **AFDC EV Registrations** | Annual count of vehicle registrations by state and fuel type. | **Manual**: AFDC Data Tool. | **Skipped** (Instructions provided). |
| **CAISO Day-Ahead LMP** | Locational Marginal Pricing for California ISO (wholesale electricity price). | **API/Manual**: OASIS API. | **Skipped** (Instructions provided). |
| **ERCOT Settlement Prices** | Settlement point prices for Texas (ERCOT market). | **Manual**: ERCOT MIS. | **Skipped** (Instructions provided). |

### 4. Argonne Monthly EV Sales
- **Source**: [https://www.anl.gov/esia/light-duty-electric-drive-vehicles-monthly-sales-updates](https://www.anl.gov/esia/light-duty-electric-drive-vehicles-monthly-sales-updates) (Mirrored on AFDC)
- **Content**: Monthly sales figures by make and model.
- **Usage**: Run `python scripts/download_data.py --datasets argonne`

### 5. IEA Global EV Data Explorer
- **Source**: [https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer](https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer)
- **Action**: Download the "Global EV Data Explorer" Excel file manually and place it in `data/raw/`.

### 6. AFDC EV Registrations
- **Source**: [https://afdc.energy.gov/data/10962](https://afdc.energy.gov/data/10962)
- **Action**: Download the dataset (CSV/Excel) manually and place it in `data/raw/`.

### 7. CAISO / ERCOT Pricing
- **Sources**:
  - CAISO: [http://oasis.caiso.com/](http://oasis.caiso.com/)
  - ERCOT: [https://www.ercot.com/mktinfo/prices](https://www.ercot.com/mktinfo/prices)
- **Action**: These require complex API queries or navigating daily zip archives. For specific experiments, download the relevant historical price data for your target timeframe manually.
