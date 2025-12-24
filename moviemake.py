from moviepy import *

import os
from os import listdir
from os.path import isfile, join

mypath = "./video"
files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
files = sorted(files, key=lambda x: os.path.getmtime(x))
print(files)

clip = ImageSequenceClip(files, fps = 30)
clip.write_videofile("video.mp4", fps = 30)
