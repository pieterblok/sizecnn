## To Do list

- [x] Delete the invisible mask head branch 
- [x] Train the amodal and modal broccoli dataset (using ORCNN)
- [x] Get the intermediate tensors of the model
- [x] Train specific parts of the network (modal-mask head branch) and freeze the other parts
- [x] Train the network with custom tensors (RGBRGB modal-mask head, while freezing the other parts of the network)
- [x] Train Mask R-CNN on the float32 XYZ images of broccoli
- [x] Train the modal-mask of ORCNN with the RGB-XYZ tensors from the two corresponding backbones (using the intermediate tensor procedure)
- [ ] Train the modal-mask of ORCNN with only the XYZ tensors from the XYZ-backbone (using the intermediate tensor procedure)
- [ ] Alter the dataloader so that it can properly load a 6-channel image (RGBXYZ)
- [ ] Store the resize choice so that both the RGB and the XYZ are transformed in the same way (see build_transform_gen in detection_utils.py)
- [ ] Realize a proper loading of the bitmasks in detection_utils.py (function annotations_to_instances)
- [ ] Optimize the normalization procedure of the XYZ images (using sklearn.preprocessing.StandardScaler)
- [ ] Use point-cloud deep-learning to filter the 3D outliers from the XYZ-mask (using a Chamfer distance loss function)
- [ ] Deep-learning regression / point-cloud regression to yield the final output (X,Y,Z coordinates of the grasping point and the diameter of the broccoli head) 

