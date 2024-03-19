import umap.umap_ as umap
import argparse
import pathlib
import numpy as np
import json
import pandas as pd
import time
import yaml

from utils import UMAPParameters, load_images_from_directory

""" Compute UMAP
    Input: 1d data (N, M) or 2d data (N, H, W)
    Output: latent vectors of shape (N, 2) or (N, 3)
"""
def computeUMAP(data, 
                n_components=2, 
                min_dist=0.1, 
                n_neighbors=15, 
                random_state=42):
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
    assert io_parameters["images_dir"], "Input dir (image filepath) not provided for training."
    assert io_parameters["output_dir"], "Output dir (dir to save the computed latent vactors) not provided for training."

    # Validate model parameters:
    model_parameters = parameters["model_parameters"]
    print("model_parameters")
    print(model_parameters)

    images_dir = io_parameters["images_dir"]
    output_dir = pathlib.Path(io_parameters["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    images = None
    if images_dir == "data/example_shapes/Demoshapes.npz": # example dataset
        images = np.load(images_dir)['arr_0']
    elif images_dir == "data/example_latentrepresentation/f_vectors.parquet": # example dataset
        df = pd.read_parquet(images_dir)
        images = df.values
    elif images_dir.split('.')[-1] == 'parquet': # data clinic
        df = pd.read_parquet(images_dir)
        images = df.values
    else: # user uploaded zip file
        images = load_images_from_directory(images_dir)
    print(images.shape)
    start_time = time.time()    
    
    # Run UMAP
    latent_vectors = computeUMAP(images, 
                                 n_components=model_parameters['n_components'],
                                 min_dist=model_parameters['min_dist'],
                                 n_neighbors=model_parameters['n_neighbors'])

    # Save latent vectors
    output_name = 'latent_vectors.npy'
    np.save(str(output_dir) + "/" + output_name, latent_vectors)

    print("UMAP done, latent vector saved.")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")