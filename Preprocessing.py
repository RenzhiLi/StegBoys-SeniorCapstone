import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
filextends = ('.jpg','.jpeg','.exif','.gif','.bmp','.png','.bpg',
	'.tif')
def load_pics(directory : 'images directory') -> "a loaded pic list":
	im , file_paths = [], []
	if os.path.isdir(directory):
		for root,dirnames,filenames in os.walk(directory):
			if root == directory:
				for filename in filenames:
					if filename.lower().endswith(filextends):
						#re.search(r'\.jpg',filename)
						filepath = os.path.join(root, filename)
						file_paths.append(filepath)
						im.append(Image.open(filepath))
	elif os.path.isfile(directory):
		im.append(Image.open(directory))
	return im, file_paths

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

def exe_cmdline():
	# command line args: imput dir, output csv file name, resize shape
	im , file_path = load_pics(sys.argv[1])
	data = preprocess(im,resize=eval(sys.argv[3]),color_mode='RGB',output_format='2d',int_process_RGB=True)
	data = np.concatenate((np.array(file_path).reshape(-1,1),data),axis=1)
	pd.DataFrame(data).to_csv(sys.argv[2],index=False,header=False)

def main():
	# example
	# da = preprocess(im,resize=(128,128),color_mode='RGB',output_format='2d')
	im , file_path = load_pics('./sample_images')
	data = preprocess(im,resize=(64,64),color_mode='RGB',output_format='2d',int_process_RGB=True)
	data = np.concatenate((np.array(file_path).reshape(-1,1),data),axis=1)
	pd.DataFrame(data).to_csv('sample_images.csv',index=False,header=False)
	print(data)
	print(data.shape)

def built_training_set(directory,targetfile,resize,tag):
	im , file_paths, count = [], [], 0
	for root,dirnames,filenames in os.walk(directory):
		if root == directory:
			for filename in filenames:
				if filename.lower().endswith(filextends):
					filepath = os.path.join(root, filename)
					file_paths.append(filepath)
					im.append(Image.open(filepath))
					if count > 100:
						data = preprocess(im,resize=resize,color_mode='RGB',output_format='2d',int_process_RGB=True)
						data = np.concatenate((np.array(file_paths).reshape(-1,1),data),axis=1)
						im , file_paths, count = [], [], 0
						if tag == 0:
							data = np.concatenate((data,np.zeros(data.shape[0]).reshape(-1,1)),axis=1)
						elif tag == 1:
							data = np.concatenate((data,np.ones(data.shape[0]).reshape(-1,1)),axis=1)
						pd.DataFrame(data).to_csv(targetfile,index=False,header=False,mode='a')
					count=count+1
	if count != 0:
		data = preprocess(im,resize=resize,color_mode='RGB',output_format='2d',int_process_RGB=True)
		data = np.concatenate((np.array(file_paths).reshape(-1,1),data),axis=1)
		if tag == 0:
			data = np.concatenate((data,np.zeros(data.shape[0]).reshape(-1,1)),axis=1)
		elif tag == 1:
			data = np.concatenate((data,np.ones(data.shape[0]).reshape(-1,1)),axis=1)
		pd.DataFrame(data).to_csv(targetfile,index=False,header=False,mode='a')
	return data

if __name__ == '__main__':
	exe_cmdline()
