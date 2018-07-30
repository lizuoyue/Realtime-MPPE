import json
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import random
import h5py, time

def generate_comb(n, m):
	assert(n >= m)
	res = []
	if m == 0:
		res = [[0] * n]
	else:
		if n - 1 >= m:
			res.extend([[0] + item for item in generate_comb(n - 1, m)])
		res.extend([[1] + item for item in generate_comb(n - 1, m - 1)])
	return res

# comb = generate_comb(6, 3)
# for i, item in enumerate(comb):
# 	print(i, item)
# quit()

img_json = json.load(open('person_keypoints_val2017.json'))
hm_json = json.load(open('heatmap_val2017.json'))
image_ids = set(list([item['image_id'] for item in img_json['annotations']]))
random.shuffle(img_json['images'])

for img_info in img_json['images']:
	key = img_info['file_name'].replace('.jpg', '')
	h, w = img_info['height'], img_info['width']
	# img = io.imread(img_info['coco_url'])
	xx, yy = np.meshgrid(np.arange(w), np.arange(h))
	hm = np.zeros((18, h, w))
	t = time.time()
	a = sum(len(item) for item in hm_json[key])
	for i, part in enumerate(hm_json[key]):		
		for x, y, v in part:
			if v < 500:
				continue
			hm[i] = np.maximum(hm[i], v / 6000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 100))		
	print((time.time() - t) / 1)
	# h5f = h5py.File('%s.h5' % key, 'w')
	# h5f.create_dataset('hm', data = hm, compression = 'gzip', compression_opts = 9)
	# h5f.close()
	# plt.imshow(img)
	# plt.imshow(hm, alpha = 0.5)
	# plt.show()
	# quit()
	continue
	last = io.imread('heatmap_val2017/%s.png' % key)
	last[0, 0] = 1
	plt.imshow(img)
	plt.imshow(last, alpha = 0.5)
	plt.show()



