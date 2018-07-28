import json
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

img_json = json.load(open('person_keypoints_val2017.json'))
hm_json = json.load(open('heatmap_val2017.json'))

for img_info in img_json['images']:
	img = io.imread(img_info['coco_url'])
	key = img_info['file_name'].replace('.jpg', '')
	h, w, _ = img.shape
	xx, yy = np.meshgrid(np.arange(w), np.arange(h))
	for part in hm_json[key]:
		print(part)
	print('=============')
	# for x, y in part:
	# 	hm = np.maximum(0, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 100))
	# 	plt.imshow(img)
	# 	plt.imshow(hm, alpha = 0.5)
	# 	plt.show()
