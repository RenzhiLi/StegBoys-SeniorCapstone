from LSBSteg import LSBSteg
import cv2
import os, os.path
import csv
import random

path = "./test_input/"
opath = "./test_output/"
i = 0
NUM = 300
num = random.randint(100,300)


with open('random_text.txt') as csv_file:
    texts=[]
    csv_reader = csv.reader(csv_file, delimiter=',')
    for t in csv_reader:
        texts.append(t)
    text_len = len(texts)
    for image_path in os.listdir(path):
        if(image_path.endswith(".png")):
            text = ''
            for _ in range(num):
                rand = random.randint(0,text_len-1)
                text += texts[rand][0] + ' '
            inputpath = os.path.join(path, image_path)
            steg = LSBSteg(cv2.imread(inputpath))
            if i == 0:
                print('starting to embedding message...')
                print('input picture: ' +str(inputpath))
                print('message embedding in this message:' + text)
            img_encoded = steg.encode_text(text)
            outputpath = os.path.join(opath, "out" + str(i) +".png")
            cv2.imwrite(outputpath, img_encoded)

            i += 1
