from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import pandas as pd

from robot_vlp.config import EXTERNAL_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    output_path: Path = EXTERNAL_DATA_DIR / "vlp_dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Pull VLP dataset from Github")
    df = pd.read_csv(
        "https://raw.githubusercontent.com/tyrel-glass/Public-VLP-Dataset/main/Public_VLP_Dataset.csv",
        index_col="index",
    )
    df.to_csv(output_path)

    logger.success("VLP dataset saved to: " + str(output_path))
    # -----------------------------------------


if __name__ == "__main__":
    app()
