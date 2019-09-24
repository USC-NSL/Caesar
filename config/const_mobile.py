# True if you want to save the module's results to an npy file 
SAVE_DATA = True

# True if you want to run the pipeline online and upload the data to next hop 
UPLOAD_DATA = False

# The uploading FPS, this number should be lower if following nodes are overloaded
UPLOAD_FPS = 20

# How many frames you will process. Set to negative if process all frames  
MAX_FRAME_ID = -1

# The number of cached frames (and metadata) 
QUEUE_SIZE = 64

# Mapping from object detection id to object name. This is model-specific
OBJ_LABEL_FILE = 'config/label_mapping.txt'

# Object detection threshold 
OBJ_THRES = 0.25

# Path to the video file. If set to integer (like 0), will process webcam live feed 
VIDEO_PATH = 'data/v1.avi' # '0'

# Name of the video source, MUST start with alpha character 
CLIENT_NAME = 'v1'

# Address of the next hop (i.e. the tracker node) 
SERVER_ADDR = 'localhost:50051'

# Which model you want to use for object detection 
OBJ_MODEL = 'mobilenet'    # Other choices: 'yolo', 'mrcnn'

# Path to the DNN model file 
OBJ_MODEL_PATH = 'checkpoints/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'   # mobilenet
# OBJ_MODEL_PATH = 'checkpoints/visdrone_model_0360000.pth'      # For 'mrcnn'                            # mrcnn

# The batch size for obj detector (recommend: 16 for mobilenet, 1 for others)
OBJ_BATCH_SIZE = 16

