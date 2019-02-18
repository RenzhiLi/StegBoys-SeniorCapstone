import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
filextends = ('.jpg','.jpeg','.exif','.gif','.bmp','.png','.bpg',
	'.tif')
def load_pics(directory : 'images directory') -> "a loaded pic list":
	im = []
	for root,dirnames,filenames in os.walk(directory):
		if root == directory:
			for filename in filenames:
				if filename.lower().endswith(filextends):
					#re.search(r'\.jpg',filename)
					filepath = os.path.join(root, filename)
					im.append(Image.open(filepath))
	return im

def preprocess(data : 'loaded pic list (from load_pics())',
				**kwargs : 'resize:tuple;color_mode:str;output_format:str'
				) ->'numpy array':
	''' kwargs:
	resize: regularize image size, recieves tuple, leave blank for no regularize
	color_mode: regularize image color format, indicated by a string
	leave blank for no regularize
	1 (1-bit pixels, black and white, stored with one pixel per byte)
	L (8-bit pixels, black and white)
	P (8-bit pixels, mapped to any other mode using a color palette)
	RGB (3x8-bit pixels, true color)
	RGBA (4x8-bit pixels, true color with transparency mask)
	CMYK (4x8-bit pixels, color separation)
	YCbCr (3x8-bit pixels, color video format)
	LAB (3x8-bit pixels, the L*a*b color space)
	HSV (3x8-bit pixels, Hue, Saturation, Value color space)
	I (32-bit signed integer pixels)
	F (32-bit floating point pixels) (from Pillow documentation)
	output_format: indicates the desired output format , str
	'2d' for a 2-dimentional numpy array
	leave blank for the original format (3/4-dimentional numpy array)
	(if the image are not regularized to the same size, the output will be a list)
	'''
	output = []
	try:
		resize = kwargs['resize']
	except:
		resize = None
	try:
		color_mode = kwargs['color_mode']
	except:
		color_mode = None
	try:
		output_format = kwargs['output_format']
	except:
		output_format = None
	try:
		int_process_RGB = kwargs['int_process_RGB']
	except:
		int_process_RGB = False
	for i in data:
		if not(resize is None):
			i = i.resize(resize)
		if not(color_mode is None):
			i = i.convert(mode=color_mode)
		i = np.array(i)
		if int_process_RGB:
			a, b, c = i[:,:,0], i[:,:,1], i[:,:,2]
			#R*256^2+G*256+B
			i = a*65536 + b*256 + c
		if not(output_format is None):
			if output_format == '2d':
				i = i.reshape((i.size))
		output.append(i)
	if not(resize is None or color_mode is None):
		output = np.array(output)
	return output

if __name__ == '__main__':
	# example
	im = load_pics('./sample_images')
	#da = preprocess(im,resize=(128,128),color_mode='RGB',output_format='2d')
	da = preprocess(im,resize=(64,64),color_mode='RGB',output_format='2d',int_process_RGB=True)
	pd.DataFrame(da).to_csv('sample_images.csv',index=False,header=False)
	print(da)
	print(da.shape)






