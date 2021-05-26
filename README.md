# ROS package for segmentation

This ros package is for semantic segmentation.
it has different models (TransLab, Trans2Seg) 
## downlaod and setup
- Download the git repository using git inside your workspace src file
```bash
cd ~/catkin_ws/src
git clone https://github.com/GhadeerElmkaiel/transformers_semantic_ros.git
```
- Install requirements
```bash
cd transformers_segmentation_ros/
pip install -r requirments.txt
```
  
- Setup the translab code
```bash
cd src/
python3 setup.py develop --user
```
  
- Build the package
```bash
cd ../../..
catkin build transformers_segmentation_ros
source devel/setup.bash
```

## Pretrained Models:
Before using the package it is necessary to download a pretrained model for the neural network you want to use (Different models for different neural network structures).  
Multiple pretrained models can be found on [google drive](https://drive.google.com/drive/folders/1gHPFC8PWQWz_J8XjXGWbDqAZxDOiN9kR?usp=sharing)

- Download the model to the *demo* folder and rename the it to **trans2seg_model.pth**
- **Note:** the name of the model can be change in the launch file but by default the used name is **trans2seg_model.pth**
 

## run
to run the segmentation, it is recommended to use the launch file
```bash
rosrun transformers_semantic_ros transformers_semantic_node
```

## TODO
- [x] Add the ability of using multiple models
- [x] Add the ability of using Synchronized Depth image
- [ ] Add sepatrate function for inference so it does inference for all topics at once
- [ ] Change the .py files to .pyx
- [X] complete the launch file