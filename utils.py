from pydantic import BaseModel, Field

class PCAParameters(BaseModel):
    n_components: int = Field(description='number of components to keep')

class UMAPParameters(BaseModel):
    n_components: int = Field(description='number of components to keep')
    min_dist: float = Field(description='min distance between points')
    n_neighbors: int = Field(description='number of nearest neighbors')
    

