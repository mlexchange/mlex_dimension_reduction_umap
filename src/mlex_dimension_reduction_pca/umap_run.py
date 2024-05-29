import umap.umap_ as umap
import argparse
import pathlib
import numpy as np
import pandas as pd
import time
import yaml
from tiled.client import from_uri
from mlex_dimension_reduction_pca.utils import load_images_from_directory

""" Compute UMAP
    Input: 1d data (N, M) or 2d data (N, H, W)
    Output: latent vectors of shape (N, 2) or (N, 3)
"""
def computeUMAP(data, 
                n_components=2, 
                min_dist=0.1, 
                n_neighbors=15, 
                random_state=42,
                standarize=False):
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    #data = StandardScaler().fit_transform(data) 

    umap_model = umap.UMAP(n_components=n_components, 
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            random_state=random_state)

    umap_result = umap_model.fit_transform(data)
    return umap_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    # Open the YAML file for all parameters
    with open(args.yaml_path, "r") as file:
        # Load parameters
        parameters = yaml.safe_load(file)

    # Validate and load I/O related parameters
    io_parameters = parameters["io_parameters"]
    # Check input and output dir are provided
    assert io_parameters[
        "output_dir"
    ], "Output dir (dir to save the computed latent vactors) not provided for training."

    # Validate model parameters:
    model_parameters = parameters["model_parameters"]
    print("model_parameters")
    print(model_parameters)

    # output directory
    output_dir = pathlib.Path(
        io_parameters["output_dir"] + "/" + io_parameters["uid_save"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images from given data_uris
    stacked_images = None

    uid_retrieve = io_parameters["uid_retrieve"]
    if uid_retrieve is not None:
        # Get feature vectors from autoencoder
        stacked_images = pd.read_parquet(
            f"data/mlexchange_store/{uid_retrieve}/f_vectors.parquet"
        ).values

    else:
        data_uris = io_parameters["data_uris"]

        for uri in data_uris:
            if "data/example_shapes/Demoshapes.npz" in uri:  # example dataset
                images = np.load(uri)["arr_0"]
            elif (
                "data/example_latentrepresentation/f_vectors.parquet" in uri
            ):  # example dataset
                df = pd.read_parquet(uri)
                images = df.values

            else:
                # FM, file system or tiled
                if io_parameters["data_type"] == "file":
                    images = load_images_from_directory(
                        io_parameters["root_uri"] + "/" + uri
                    )
                else:  # tiled
                    tiled_client = from_uri(
                        io_parameters["root_uri"],
                        api_key=io_parameters["data_tiled_api_key"],
                    )
                    images = tiled_client[uri][:]
                    if len(images.shape) == 2:
                        images = images[np.newaxis, :, :]

            if stacked_images is None:
                stacked_images = images
            else:
                stacked_images = np.concatenate((stacked_images, images), axis=0)


    start_time = time.time()    
    
    # Run UMAP
    latent_vectors = computeUMAP(images, 
                                 n_components=model_parameters['n_components'],
                                 min_dist=model_parameters['min_dist'],
                                 n_neighbors=model_parameters['n_neighbors'])

    # Save latent vectors
    output_name = 'latent_vectors.npy'
    save_path = str(output_dir) + '/' + output_name
    print(save_path)
    np.save(str(output_dir) + '/' + output_name, latent_vectors)

    print("UMAP done, latent vector saved.")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")