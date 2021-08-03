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
pip3 install -r requirements.txt
```
  
- Setup the translab code
```bash
cd src/
python3 setup.py develop --user
```
  
- Build the package
```bash
cd ../../..
catkin build transformers_semantic_ros
source devel/setup.bash
```

## Pretrained Models:
Before using the package it is necessary to download a pretrained model for the neural network you want to use (Different models for different neural network structures).  
Multiple pretrained models can be found on [google drive](https://drive.google.com/drive/folders/1gHPFC8PWQWz_J8XjXGWbDqAZxDOiN9kR?usp=sharing)
*Note: The code is configured for the model **Sber2400_50_All_classes***

- Create a folder called **demo** inside the */src* folder
- Download the model to the *demo* folder and rename the it to **trans2seg_model.pth**
- **Note:** the name of the model can be change in the launch file but by default the used name is **trans2seg_model.pth**
 

## run
to run the segmentation, it is recommended to use the launch file
```bash
roslaunch transformers_semantic_ros transformers_segmentation.launch
```

## Add new models
In general it should be easy to add new models to the package.
The argument **model_name** should be changed to the new model when running the code *(the default is Trans2Seg)*
in the code also the loading of the model should be added
```python
if self.MODEL_NAME== "TransLab" or self.MODEL_NAME=="Trans2Seg":				# If model is TransLab or Trans2Seg
            self.model = get_segmentation_model().to(self.device)
            rospyLogInfoWrapper("Loaded the model!")
        else:
            # here add new model
            # replace "else" by "elif self.MODEL_NAME== new_model_name"
            # then add the code for loading the model 
            pass
```
Also in the callbacks the new models forward pass should be added:
```python
        with torch.no_grad():
            # pass
            if self.MODEL_NAME == "TransLab":
                output, output_boundary = self.model.evaluate(tensor)
                result = output.argmax(1)[0].data.cpu().numpy().astype('uint8')*127
                result = cv2.resize(result, size_, interpolation=cv2.INTER_NEAREST)
                encode = "mono8"
            elif self.MODEL_NAME == "Trans2Seg":
                output = self.model(tensor)
                mask = torch.argmax(output[0], 1)[0].cpu().data.numpy()
             
                result = get_color_pallete(mask, cfg.DATASET.NAME)
                result = result.convert("RGB")
                result = result.resize(size_, Img.NEAREST)
                result = np.array(result)
                encode = "rgb8"
            # elif self.MODEL_NAME == "new_model_name":
            #    code for forward pass through the new model 
```
**Note: The requirments for the new models should also be installed in the same environment (this repository currently has only the files for Trans2Seg and TransLab neural networks)**


## TODO
- [x] Add the ability of using multiple models
- [x] Add the ability of using Synchronized Depth image
- [ ] Add sepatrate function for inference so it does inference for all topics at once
- [ ] Change the .py files to .pyx
- [X] complete the launch file