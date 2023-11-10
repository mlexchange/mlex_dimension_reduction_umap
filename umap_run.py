import umap.umap_ as umap
import argparse
import pathlib
import numpy as np
import json
import pandas as pd

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

    parser.add_argument('image_dir', help='image filepath')
    parser.add_argument('output_dir', help='dir to save the computed latent vactors')
    parser.add_argument('parameters', help='dictionary that contains model parameters')
    
    args = parser.parse_args()

    images_dir = args.image_dir
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    images = None
    if images_dir == "data/example_shapes/Demoshapes.npz":
        images = np.load(images_dir)['arr_0']
    if images_dir == "data/example_latentrepresentation/f_vectors.parquet":
        df = pd.read_parquet(images_dir)
        images = df.values
    else: # user uploaded zip file
        images = load_images_from_directory(images_dir)
    print(images.shape)

    # Load dimension reduction parameter
    if args.parameters is not None:
        parameters = UMAPParameters(**json.loads(args.parameters))
    
    print(f'UMAP parameters: n_components={parameters.n_components}, min_dist={parameters.min_dist}, n_neighbors={parameters.n_neighbors}.')


    # Run UMAP
    latent_vectors = computeUMAP(images, 
                                 n_components=parameters.n_components,
                                 min_dist=parameters.min_dist,
                                 n_neighbors=parameters.n_neighbors)

    
    # Save latent vectors
    output_name = f'umap_{parameters.n_components}d_{parameters.n_neighbors}_{parameters.min_dist}.npy'
    np.save(str(output_dir) + "/" + output_name, latent_vectors)

    print("UMAP done, latent vector saved.")