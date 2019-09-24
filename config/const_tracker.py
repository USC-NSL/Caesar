# True if you want to save the module's results to an npy file 
SAVE_DATA = True

# True if you want to run the pipeline online and upload the data to next hop 
UPLOAD_DATA = False

# The number of cached frames (and metadata) 
QUEUE_SIZE = 128

# Object that will be tracked 
TRACK_LABELS = ['person', 'car']

# Object that will be detected as "traveling" with a tube  
ATTACH_LABELS = ['bike', 'bag']

# address for receiving incoming traffic
LOCAL_ADDR = 'localhost'

# port for receiving incoming traffic
LOCAL_PORT = 50051

# Address of the next hop (i.e. the action node) 
SERVER_ADDR = 'localhost:50052'

### CONFIG: camera topology config file 
TOPO_PATH = 'config/camera_topology.txt'

# The shape of the input image (width, height), Make sure it matches your video 
IMG_SHAPE = (640, 480)

# Path to the ReID DNN model
TRACK_MODEL_PATH = 'checkpoints/deepsort/mars-small128.pb'