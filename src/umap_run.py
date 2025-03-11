import argparse
import logging
import os
import sys
import time
import warnings

import yaml

from dim_reduction import compute_umap
from parameters import IOParameters, UMAPParameters
from utils.data_utils import load_data, save_results
from utils.tiled_utils import TiledDataset

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout,  # Force all logs to stdout
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    # Load parameters
    with open(args.yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    # Validate parameters
    io_parameters = IOParameters(**parameters["io_parameters"])
    model_parameters = UMAPParameters(**parameters["model_parameters"])
    logger.info(f"Parameters loaded: {model_parameters}")

    # Load images from given data_uris
    if io_parameters.uid_retrieve != "":
        data_uri = f"{io_parameters.results_tiled_uri}/{io_parameters.uid_retrieve}"
    elif io_parameters.data_type == "file":
        data_uri = None
    else:
        data_uri = io_parameters.root_uri

    tiled_dataset = TiledDataset(
        data_uri,
        io_parameters.results_tiled_uri,
        read_tiled_key=io_parameters.data_tiled_api_key,
        write_tiled_key=io_parameters.results_tiled_api_key,
    )
    stacked_images = load_data(io_parameters, tiled_dataset, logger)

    start_time = time.time()

    # Run UMAP
    if io_parameters.save_model_path is not None:
        os.makedirs(io_parameters.save_model_path, exist_ok=True)
        save_model_path = (
            f"{io_parameters.save_model_path}/{io_parameters.uid_save}.joblib"
        )
    else:
        save_model_path = None
    latent_vectors = compute_umap(
        stacked_images,
        n_components=model_parameters.n_components,
        min_dist=model_parameters.min_dist,
        n_neighbors=model_parameters.n_neighbors,
        load_model_path=io_parameters.load_model_path,
        save_model_path=save_model_path,
    )

    save_results(latent_vectors, io_parameters, tiled_dataset, parameters)

    logger.info("UMAP done!")
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Execution time: {execution_time} seconds")
