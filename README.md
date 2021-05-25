<!-- # ROS package for glass and floor segmentation

## downlaod and setup
- Download the git repository using git inside your workspace src file
```bash
cd ~/catkin_ws/src
git clone https://github.com/GhadeerElmkaiel/glass_segmentation_ros.git
```
- Install requirements
```bash
cd glass_segmentation_ros/
pip install -r requirments.txt
```
  
- Setup the translab code
```bash
cd src/
python3 setup.py develop --user
```

## Pretrained Models:
Before using the package it is necessary to download a pretrained model.  
Multiple pretrained models can be found on [google drive](https://drive.google.com/drive/folders/1IltRzX39q-Sx5tYN61suQrAWXyzJGGMD?usp=sharing)

- Download the model to the *demo* folder and rename the it to **model.pth**
 

## run
to run the segmentation, currently you need to run the node using rosrun
```bash
rosrun glass_semantic_ros glass_semantic_node
```

## TODO
- [x] Add the ability of using multiple models
- [ ] Add sepatrate function for inference so it does inference for all topics at once
- [ ] Change the .py files to .pyx
- [ ] complete the launch file -->