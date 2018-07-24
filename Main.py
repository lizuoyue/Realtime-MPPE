import numpy as np
import os, sys
if os.path.exists('../../Python-Lib/'):
	sys.path.insert(1, '../../Python-Lib')
import tensorflow as tf
import math, time
from Model import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2, json

def find_peaks(heatmap, th, gaussian = False):
	if gaussian:
		from scipy.ndimage.filters import gaussian_filter
		heatmap = gaussian_filter(heatmap, sigma = 3)
	map_l = np.zeros(heatmap.shape)
	map_l[1:, :] = heatmap[:-1, :]
	map_r = np.zeros(heatmap.shape)
	map_r[:-1, :] = heatmap[1:, :]
	map_u = np.zeros(heatmap.shape)
	map_u[:, 1:] = heatmap[:, :-1]
	map_d = np.zeros(heatmap.shape)
	map_d[:, :-1] = heatmap[:, 1:]

	peaks_binary = np.logical_and.reduce(
		(heatmap >= map_l, heatmap >= map_r, heatmap >= map_u, heatmap >= map_d, heatmap > th)
	)
	peaks = [item for item in zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])]
	return peaks

img_t = tf.placeholder(tf.float32, [None, None, None, 3])
pred_t = Predict(img_t)

d = {'kernel:0': '_0', 'bias:0': '_1'}
assign_op = []
for v in tf.global_variables():
	parts = v.name.split('/')
	if len(parts) == 3:
		name = parts[1] + '_' + parts[0] + d[parts[2]] + '.npy'
	else:
		name = parts[0] + d[parts[1]] + '.npy'
	weights = np.load('weights/' + name)
	if len(weights.shape) > 1:
		assign_op.append(v.assign(weights.transpose([2, 3, 1, 0])))
	else:
		assign_op.append(v.assign(weights))

box_size = 368
shapes = np.array([0.5, 1, 1.5, 2])

def predict_heatmap(img):
	size = max(img.shape[0], img.shape[1])
	multipliers = (box_size * shapes) / size
	res = np.zeros(img.shape[:2] + (19, ))
	with tf.Session() as sess:
		for item in assign_op:
			item.op.run()
		for m in multipliers:
			input_img = cv2.resize(img, (0, 0), fx = m, fy = m, interpolation = cv2.INTER_CUBIC)[np.newaxis, ...]
			l1, l2 = sess.run(pred_t, feed_dict = {img_t: input_img / 255.0 - 0.5})
			res += cv2.resize(l2[0, ...], (img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
		res /= len(multipliers)
	return res

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return super(NumpyEncoder, self).default(obj)

result = {}
files = ['data/000000000000.jpg']#glob.glob('/disks/data4/zyli/coco2017data/train2017/*')
for seq, file in enumerate(files):
	img_id = file.split('/')[-1].replace('.jpg', '')
	print(seq, img_id)
	img = np.array(Image.open(file), np.float32)
	res = predict_heatmap(img)
	res_single = []
	for i in range(18):
		res_single.append(find_peaks(res[..., i], res[..., i].max() * 0.4))
	result[img_id] = res_single
	if seq % 1000 == 999:
		with open('heatmap_result.json', 'w') as fp:
			fp.write(json.dumps(result, cls = NumpyEncoder))
			fp.close()


