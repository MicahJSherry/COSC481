import matplotlib.pyplot as plt
from scipy.io import wavfile

import os

import numpy as np

import openl3




data_dir = "../spark22"

sat_dirs = os.listdir(data_dir)

count =0
model = openl3.models.load_image_embedding_model(input_repr="mel256", content_type="music",
                                                 embedding_size=512)
for sat in sat_dirs:
    png_files = os.listdir(f"{data_dir}/{sat}")
    for file in png_files:
        #print(f"{data_dir}/{genre}/{file}")

        try:
            
            filepath = f"{data_dir}/{sat}/{file}"
            print(f"{count}:{filepath}")
            openl3.process_image_file(filepath, output_dir='../embedings', model=model)
        except:
            count += 1