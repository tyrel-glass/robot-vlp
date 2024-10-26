from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from typing_extensions import Annotated
from typing import Optional, List

from robot_vlp.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Annotated[str, typer.Option()],
    
    search_keys: Annotated[List[str], typer.Option()],
    overlap: Annotated[float, typer.Option()] = 0.5,
    
    # output_path: str ,

    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")

    print(input_path)
    print(type(search_keys))
    print(search_keys)
    print('overlap: ', overlap)
   

    # -----------------------------------------


if __name__ == "__main__":
    app()
