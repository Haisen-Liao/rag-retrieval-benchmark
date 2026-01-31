import os
import argparse
from beir import util

def download_beir_dataset(dataset: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading BEIR dataset: {dataset}")
    util.download_and_unzip(
        url=f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip",
        out_dir=out_dir
    )
    print("Download finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="BEIR dataset name, e.g. scifact, trec-covid")
    parser.add_argument("--out_dir", type=str, default="data/beir_raw")
    args = parser.parse_args()

    download_beir_dataset(args.dataset, args.out_dir)
