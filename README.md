# litho_identification

This repository includes the Master's thesis code of Nataly Chacon Buitrago, focused on lithology identification of point clouds from geological outcrops. 

![](/gaspe.png "Point cloud of the turbidites of the Cloridorme formation in the Gaspe Peninsula in Canada")

`Feature.py` calculates the features of 3D outcrop point clouds. The point cloud features are useful for classification and segmentation tasks.
In this case, the features obtained were used for the lithological classification task of 3D point cloud outcrops of turbidites. 
`Lithology_identification.ipynb` classifies the lithology of 3D outcrops using as input the features for the training and testing sets calculated using `Feature.py`.

Lithology classification is the building block for outcrop studies. Outcrop studies are an essential tool for characterizing and understanding 
depositional environments; they also provide important information on the size, geometry, and potential connectivity of geobodies.

