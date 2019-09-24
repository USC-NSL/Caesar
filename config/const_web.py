# True if you want to save the module's results to an npy file 
SAVE_DATA = True

# The number of cached frames (and metadata) 
QUEUE_SIZE = 128

# address for receiving incoming traffic
LOCAL_ADDR = 'localhost'

# port for receiving incoming traffic
LOCAL_PORT = 50053

# All mobile sources' names (e.g. if you have two devices v1 and v2, you should put both here)
CLIENT_NAMES = ['v1', 'v2']

# True if you want to view the track id in the visualization
SHOW_TRACK = True 

# True if you want to save the rendered results into video files 
SAVE_VIDEO = False 

# FPS that we will use to save the result frames to video files  
VIDEO_WRITE_FPS = 20

# The shape of the input image (width, height), Make sure it matches your video 
IMG_SHAPE = (640, 480)

# url for webpage, change to the machine's IP if you want the webpage to be publicly accissible
WEB_ADDR = 'localhost:50088' 

# The display FPS on the webpage. This should be similar as mobile's processing FPS
WEB_DISPLAY_FPS = 15

