import cv2
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004, mosaicing_CFA_Bayer
import imageio
import math

def ReadExpoTimes(fileName):
	return np.power(2, np.loadtxt(fileName))

def ReadImages(fileNames):
	imgs = []
	for imgStr in fileNames:
		img = cv2.imread(imgStr, -1)

		# equivalent to im2single from Matlab
		img = img / 2**16
		img = np.float32(img)

		img.clip(0, 1)

		imgs.append(img)
	return np.array(imgs)

def Demosaic(imgs):
	demosaicedArray = []
	for img in imgs:
		img = mosaicing_CFA_Bayer(img)
		img = demosaicing_CFA_Bayer_Malvar2004(img)
		demosaicedArray.append(img)
	return demosaicedArray

def ApplyGammaCorrection(imgs):
	return np.power(imgs,1/GAMMA)

def ResizeImgs(imgs, width, height):
	resizedImgs = []
	for img in imgs:
		cv2.resize(img, resizedImg, size(width,height))
		resizedImgs.append(resizedImg)
	return resizedImgs

def WriteImages(imgs, folderNames): # Possibly add extension argument
	for index, img in enumerate(imgs):
		try:
			#cv2.imwrite(folderNames[index], img)
			cv2.imwrite('Test/EXTRA/001/wroteThatOnMyOwn.tif', img)
			break
		except:
			print("Unexpected error while writing imgs :", sys.exc_info()[0])
			raise
	return

def ReadTrainingData(fileNames):
	imgs = ReadImages(fileNames)
	label = ReadLabel(fileNames[0])
	return imgs, label

def ReadLabel(fileName):
	label = imageio.imread(fileName[:fileName.rfind("/")+1] + 'HDRImg.hdr', 'hdr')
	label = label[:, :, [2, 1, 0]]
	return label

def select_subset(patches):
	maxTh = 0.8
	minTh = 0.2

	thresh = 0.5 * patches.shape[0] * patches.shape[1] * 3

	badInds = np.logical_or(patches > maxTh, patches < minTh)

	indices = badInds.sum(0).sum(0).sum(0) > thresh
	return np.where(indices == 1)[0]

def get_num_patches(width, height, patch_size, stride):
	num_patches_x = math.floor((width-patch_size)/stride)+1;
	num_patches_y = math.floor((height-patch_size)/stride)+1;
	return num_patches_y * num_patches_x;
