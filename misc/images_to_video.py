import os 
import sys 

import cv2


FPS = 30 


image_folder = sys.argv[1]
video_name = sys.argv[2]

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort() 

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(
    video_name, 
    cv2.VideoWriter_fourcc(*'XVID'), 
    FPS, 
    (width, height),
)

for i, image in enumerate(images):
    if i % 100 == 0:
        print("processed frame %d" % i)
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

print('done')