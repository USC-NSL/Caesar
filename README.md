# Caesar: Cross-Camera Complex Activity Detection

## Requirements
- Python 3.X
- OpenCV 3.X
- TensorFlow 1.12+
- Install other py packages using ```pip install requirements.txt```

## Structure
The workflow is shown in the image below. You will run one of the scripts to turn the device into a mobile/tracker/action/web node. 

![Workflow of Caesar Demo](data/workflow.png)

- ```mobile/``` src code for object detection
- ```tracker/``` src code for tracking and ReID
- ```server/``` src code for action detection
- ```web/``` src code for web interface 
- ```network/``` src code for I/O and RPC
- ```config/``` config file for the system setup  
- ```main_xxx.py``` the main script to run the node 
- ```checkpoints/``` folder that contains the model files 
- ```data/``` a test video is here 
- ```misc/``` scripts for debugging/testing

## Run the Pipeline
1. **Prepare models** (download the models to the ```checkpoints/``` folder)
- Object Detection (Option-1: [MobileNet-V2](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz), Option-2: [YOLOv2](https://pjreddie.com/darknet/yolov2/)). 
- ReID ([DeepSort](https://drive.google.com/open?id=1m2ebLHB2JThZC8vWGDYEKGsevLssSkjo))
- Action Detection ([ACAM](https://drive.google.com/open?id=138gfVxWs_8LhHiVO03tKpmYBzIaTgD70)). For ACAM-related setup, please look at [its repo](https://github.com/oulutan/ACAM_Demo) and make sure your environment can run its demo code.

2. **Check following files**
- Double check the key-value mapping in ```config/label_mapping.txt``` to reflect your object detector's output (key is the class id, value is the object label). The default one is for SSD-MobileNetV2.
- Modify the ```config/camera_topology.txt``` to indicate the camera connectivity (see the inline comments in the file for detail)
- Modify the ```config/act_def.txt``` to define your complex activity using the syntax (see the inline comments in the file for detail)
- Modify the parameters in ```main_xx.py``` files (look for lines with ```### CONFIG``` comments) to be same as your machines' setup. 

3. **Run the node**
- On a mobile device: Run ```python main_mobile.py``` (need GPU). If you are using Nvidia TX2 or above, please set the board to be the best-performance mode to get the max FPS. Here is a [tutorial for TX2](https://www.jetsonhacks.com/2017/03/25/nvpmodel-nvidia-jetson-tx2-development-kit/). 
- On a server: Run one of ```python main_tracker.py``` / ```python main_act.py``` / ```python main_web.py``` (the tracker and act need GPU).

4. **About the runtime**
- The node will save logs into a ```debug.log``` text file in the home dir 
- For GPU-enabled nodes, start their script first and wait for a while to make sure it is ready before starting the up-stream nodes. 
- If you want to exist, press Ctrl-C to exit, some node may need Ctrl-C second time to fully end. 

5. **Currently Supported Vocabulary**
- *Cross-tube actions*: 'close', 'near', 'far', 'approach', 'leave', 'cross'
- *Single-tube actions*: 'start', 'end', 'move', 'stop', 'use_phone', 'carry', 'use_computer', 'give', 'talk', 'sit', 'with_bike', 'with_bag'

## Debugging Step by Step
- First, turn on the ```SAVE_DATA``` in the main script so it could save its intermediate results to an npy file under then ```res/[main_script_name]``` folder
- Then, you can modify the config in ```main_gt.py```. This script will read the raw videos and the intermediate data (specified by the parameters), and render the intermediate data to the frames (e.g. detections, track ids, actions) so you can see them. Moreover, it can server as a node and upload the data to the next running node. 


**Example:** 
- Run ```main_mobile.py``` only for several videos and get the data saved. Then you can config the data source as the mobile's output in ```main_gt.py```. Moreover, you should make the tracker's IP as the server address in ```main_gt.py```. 
- Then, you can run ```main_tracker.py``` which starts waiting for incoming data, and at same time you run ```main_gt.py``` to serve as a mobile node that uploads video and detection results. 
- When finish all uploading, you can see the tracker's npy file in its own result folder. Put this file's path in ```main_gt.py``` and change the sending address to ```main_act.py```. 
- After doing this, you can run ```main_gt.py``` again but as a tracker node, and use it to input to ```main_act.py``` for action results. You can repeat the process for debugging ```main_web.py``` in the last step. 

**If you have any issues, please leave your questions in "issues"**
