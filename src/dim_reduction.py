import umap.umap_ as umap


def compute_umap(
    data,
    n_components=2,
    min_dist=0.1,
    n_neighbors=15,
    random_state=42,
):
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)

    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )

    umap_result = umap_model.fit_transform(data)

    return umap_result
