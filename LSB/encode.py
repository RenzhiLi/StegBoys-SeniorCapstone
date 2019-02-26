from LSBSteg import LSBSteg
import cv2
import os, os.path
import csv

path = "Input/"
opath = "Output/"
i = 0

with open('random_text.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for image_path, text in zip(os.listdir(path), csv_reader):
        if(image_path.endswith(".png")):
            inputpath = os.path.join(path, image_path)
            steg = LSBSteg(cv2.imread(inputpath))
            img_encoded = steg.encode_text(text[0])
            outputpath = os.path.join(opath, "out" + str(i) +".png")
            cv2.imwrite(outputpath, img_encoded)

            i += 1
