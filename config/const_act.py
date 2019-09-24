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
LOCAL_PORT = 50052

# Address of the next hop (i.e. the action node) 
SERVER_ADDR = 'localhost:50053'

# Path to the action detection NN model 
NN_ACT_MODEL_PATH = 'checkpoints/model_ckpt_soft_attn_pooled_cosine_drop_ava-130'

# How many tubes will be batch-processed 
NN_BATCH = 4

# Max number of frames in a tube chunk to be processed by DNN
TUBE_SIZE = 32