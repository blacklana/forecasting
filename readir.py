import cv2
import numpy as np
import os
import pandas as pd


loc = "./daily/"

dir = os.listdir(loc)

files = []


def check(a):
    for d in dir:
        x = d.split(".")
        files.append(x[0])

    print(files)
    my_set = set(files)  # Convert list to set
    return a in my_set


print(check("RIVAN"))
