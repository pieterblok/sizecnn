# Mask R-CNN + Filtering
In this method we only use the visible masks from Mask R-CNN. From the contour of the visible masks, an enclosing circle is fitted. Then, a pixel-to-mm conversion factor is calculated from the filtered XYZ-coordinates of the visible mask. The real-world diameter is estimated by multiplying the diameter of the enclosing circle (in pixels) with the pixel-to-mm factor. <br/> <br/>
Please follow this procedure: <br/>
   1. Train Mask R-CNN, see [MRCNN_Train_and_Evaluate_VisibleMasks.ipynb](MRCNN_Train_and_Evaluate_VisibleMasks.ipynb)
   2. Estimate the diameter from the visible masks, see [Diameter_estimation_VisibleMasks.ipynb](Diameter_estimation_VisibleMasks.ipynb)
