# ORCNN & CNN Regression
In this method we use a CNN regression network to estimates the diameter (mm) from the registered XYZ image, using the visible and the amodal masks from ORCNN. <br/> <br/> Please follow this procedure: <br/>
1. Annotate the dataset, see [ANNOTATE.md](ANNOTATE.md)
2. Train ORCNN, see [ORCNN_Train_and_Evaluate_AmodalVisibleMasks.ipynb](ORCNN_Train_and_Evaluate_AmodalVisibleMasks.ipynb)
3. Prepare an image dataset for CNN regression training, see [Create_XYZA_images_for_regression_training.ipynb](Create_XYZA_images_for_regression_training.ipynb)
4. Train the CNN regression network, see [Train_diameter_regression_AmodalVisibleMasks.ipynb](Train_diameter_regression_AmodalVisibleMasks.ipynb)
5. Estimate the diameters with the trained ORCNN and CNN regression network, see [Diameter_regression_AmodalVisibleMasks.ipynb](Diameter_regression_AmodalVisibleMasks.ipynb)
