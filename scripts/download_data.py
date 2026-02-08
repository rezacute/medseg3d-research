
import os
import argparse
import requests
from pathlib import Path
from datetime import datetime, timedelta
import zipfile
import io

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

class DatasetDownloader:
    def __init__(self, output_dir, acn_token=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.acn_token = acn_token

    def _download_file(self, url, filename, subfolder=None):
        target_path = self.output_dir
        if subfolder:
            target_path = target_path / subfolder
            target_path.mkdir(parents=True, exist_ok=True)
        
        target_file = target_path / filename
        print(f"Downloading {url} to {target_file}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(target_file, 'wb') as f:
            for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB'):
                f.write(data)
        print(f"Saved: {target_file}")
        return target_file

    def download_acn_data(self):
        """
        Downloads ACN-Data from Caltech.
        Note: Requires an API token.
        """
        if not self.acn_token:
            print("Skipping ACN-Data: No API token provided. Register at https://ev.caltech.edu/register")
            return

        print("Downloading ACN-Data (Caltech)...")
        # Example API call for last 30 days
        base_url = "https://api.ev.caltech.edu/api/v1/sessions/caltech"
        headers = {"Authorization": f"Bearer {self.acn_token}"}
        
        # We can paginate or get a specific range. For now, we'll just demonstrate the request.
        # This is a sample since ACN usually requires paging.
        response = requests.get(base_url, headers=headers)
        if response.status_code == 200:
            target_file = self.output_dir / "acn_caltech_sessions.json"
            with open(target_file, 'w') as f:
                f.write(response.text)
            print(f"Saved ACN sessions to {target_file}")
        else:
            print(f"Failed to download ACN-Data: {response.status_code} {response.text}")

    def download_urban_ev(self):
        """
        Downloads UrbanEV dataset from GitHub.
        """
        print("Downloading UrbanEV (Shenzhen)...")
        base_url = "https://raw.githubusercontent.com/IntelligentSystemsLab/UrbanEV/main/data/"
        files = [
            "adj.csv", "distance.csv", "duration.csv", "e_price.csv", "inf.csv",
            "inf_raw.csv", "occupancy.csv", "poi.csv", "s_price.csv",
            "volume-11kW.csv", "volume.csv", "weather_airport.csv",
            "weather_central.csv", "weather_header.txt"
        ]
        
        for f in files:
            self._download_file(base_url + f, f, subfolder="urban_ev")

    def download_palo_alto(self):
        """
        Downloads Palo Alto Open Data EV sessions.
        """
        print("Downloading Palo Alto sessions...")
        # Confirmed working URL from Junar portal
        url = "https://data.paloalto.gov/rest/datastreams/279760/data.csv"
        self._download_file(url, "palo_alto_ev_sessions.csv")

    def download_iea(self):
        """
        IEA Global EV Data Explorer requires manual download or scraping which is fragile.
        """
        print("Skipping IEA Global EV Data (Requires manual download from https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer)")

    def download_argonne(self):
        """
        Downloads Argonne Monthly EV Sales via AFDC mirror.
        """
        print("Downloading Argonne Monthly EV Sales (via AFDC mirror)...")
        # Confirmed working URL for PEV sales by model (requires auth token in query param)
        url = "https://afdc.energy.gov/files/u/data/data_source/10567/10567_pev_sales_2-28-20.xlsx?3e66cfe942"
        self._download_file(url, "argonne_ev_sales.xlsx")

    def download_afdc_registrations(self):
        """
        AFDC EV Registrations data requires navigating the data tool.
        """
        print("Skipping AFDC Registrations (Requires manual download from https://afdc.energy.gov/data/10962)")

    def download_caiso(self, days=7):
        """
        CAISO LMP data requires complex API queries or ZIP handling not suitable for this simple script.
        """
        print("Skipping CAISO Day-Ahead LMP (Requires OASIS API or manual download from http://oasis.caiso.com/)")

    def download_ercot(self):
        """
        ERCOT data requires navigating a daily file structure.
        """
        print("Skipping ERCOT Settlement Prices (Requires manual download from https://www.ercot.com/mp/data-products/data-product-details?id=NP6-788-ER)")

def main():
    parser = argparse.ArgumentParser(description="Download EV datasets for QRC-EV framework.")
    parser.add_argument("--all", action="store_true", help="Download all available datasets")
    parser.add_argument("--output-dir", type=str, default="data/raw/", help="Directory to save downloaded files")
    parser.add_argument("--acn-token", type=str, help="ACN-Data API token")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to download (acn, urban, paloalto, iea, argonne, afdc, caiso)")

    args = parser.parse_args()
    downloader = DatasetDownloader(args.output_dir, acn_token=args.acn_token)

    if args.all:
        datasets = ["acn", "urban", "paloalto", "argonne", "iea", "afdc", "caiso", "ercot"]
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = []
        print("No datasets specified. Use --all or --datasets [names]")

    for ds in datasets:
        try:
            if ds == "acn": downloader.download_acn_data()
            elif ds == "urban": downloader.download_urban_ev()
            elif ds == "paloalto": downloader.download_palo_alto()
            elif ds == "iea": downloader.download_iea()
            elif ds == "argonne": downloader.download_argonne()
            elif ds == "afdc": downloader.download_afdc_registrations()
            elif ds == "caiso": downloader.download_caiso()
            elif ds == "ercot": downloader.download_ercot()
            else: print(f"Unknown dataset: {ds}")
        except Exception as e:
            print(f"Error downloading {ds}: {e}")

if __name__ == "__main__":
    main()
