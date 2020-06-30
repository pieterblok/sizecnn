## To Do list

- [x] Delete the invisible mask head branch 
- [x] Train the amodal and modal broccoli dataset (using ORCNN)
- [x] Get the intermediate tensors of the model
- [x] Train specific parts of the network (modal-mask head branch) and freeze the other parts
- [x] Train the network with custom tensors (RGBRGB modal-mask head, while freezing the other parts of the network)
- [x] Train Mask R-CNN on the float32 XYZ images of broccoli
- [ ] Optimize the normalization procedure of the XYZ images (using sklearn.preprocessing.StandardScaler)
- [ ] Use point-cloud deep-learning to filter the 3D outliers from the XYZ-mask (using a Chamfer distance loss function)
- [ ] Train the modal-mask of ORCNN with the RGB-XYZ tensors from the two corresponding backbones (using the intermediate tensor procedure)
- [ ] Deep-learning regression / point-cloud regression to yield the final output (X,Y,Z coordinates of the grasping point and the diameter of the broccoli head) 

