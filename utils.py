from pydantic import BaseModel, Field
import os
from PIL import Image
import numpy as np

class PCAParameters(BaseModel):
    n_components: int = Field(description='number of components to keep')

class UMAPParameters(BaseModel):
    n_components: int = Field(description='number of components to keep')
    min_dist: float = Field(description='min distance between points')
    n_neighbors: int = Field(description='number of nearest neighbors')
    
def load_images_from_directory(directory_path):
    image_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".png"):
            file_path = os.path.join(directory_path, filename)
            try:
                img = Image.open(file_path)
                img_array = np.array(img)
                image_data.append(img_array)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    image_data = np.array(image_data)
    return image_data