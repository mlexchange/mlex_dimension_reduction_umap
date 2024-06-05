import umap
from joblib import dump, load


def compute_umap(
    data,
    n_components=2,
    min_dist=0.1,
    n_neighbors=15,
    random_state=42,
    load_model_path=None,
    save_model_path=None,
):
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)

    if load_model_path:
        umap_model = load(load_model_path)
        umap_result = umap_model.transform(data)
    else:
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )

        umap_model.fit(data)
        umap_result = umap_model.embedding_

        if save_model_path:
            dump(umap_model, save_model_path)

    return umap_result
