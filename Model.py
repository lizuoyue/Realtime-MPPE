import os, sys
if os.path.exists('../../Python-Lib/'):
	sys.path.insert(1, '../../Python-Lib')
import tensorflow as tf

def VGG19(scope, img, reuse = None):
	with tf.variable_scope(scope, reuse = reuse):
		conv1_1 = tf.layers.conv2d       (inputs = img    , filters =  64, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv1_1') # 256
		conv1_2 = tf.layers.conv2d       (inputs = conv1_1, filters =  64, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv1_2') # 256
		pool1   = tf.layers.max_pooling2d(inputs = conv1_2, pool_size = 2, strides = 2)																	 # 128
		conv2_1 = tf.layers.conv2d       (inputs = pool1  , filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv2_1') # 128
		conv2_2 = tf.layers.conv2d       (inputs = conv2_1, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv2_2') # 128
		pool2   = tf.layers.max_pooling2d(inputs = conv2_2, pool_size = 2, strides = 2)																	 #  64
		conv3_1 = tf.layers.conv2d       (inputs = pool2  , filters = 256, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv3_1') #  64
		conv3_2 = tf.layers.conv2d       (inputs = conv3_1, filters = 256, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv3_2') #  64
		conv3_3 = tf.layers.conv2d       (inputs = conv3_2, filters = 256, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv3_3') #  64
		conv3_4 = tf.layers.conv2d       (inputs = conv3_3, filters = 256, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv3_4') #  64
		pool3   = tf.layers.max_pooling2d(inputs = conv3_4, pool_size = 2, strides = 2)																	 #  32
		conv4_1 = tf.layers.conv2d       (inputs = pool3  , filters = 512, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv4_1') #  32
		conv4_2 = tf.layers.conv2d       (inputs = conv4_1, filters = 512, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv4_2') #  32
		conv4_3 = tf.layers.conv2d       (inputs = conv4_2, filters = 256, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv4_3') #  32
		conv4_4 = tf.layers.conv2d       (inputs = conv4_3, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv4_4') #  32
		return conv4_4

def FirstStageBranch(scope, feature, num, reuse = None):
	with tf.variable_scope(scope, reuse = reuse):
		conv5_1 = tf.layers.conv2d       (inputs = feature, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv5_1')
		conv5_2 = tf.layers.conv2d       (inputs = conv5_1, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv5_2')
		conv5_3 = tf.layers.conv2d       (inputs = conv5_2, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv5_3')
		conv5_4 = tf.layers.conv2d       (inputs = conv5_3, filters = 512, kernel_size = 1, padding = 'same', activation = tf.nn.relu, name = 'conv5_4')
		conv5_5 = tf.layers.conv2d       (inputs = conv5_4, filters = num, kernel_size = 1, padding = 'same', activation = None      , name = 'conv5_5')
		return conv5_5

def StageBranch(scope, feature, num, reuse = None):
	with tf.variable_scope(scope, reuse = reuse):
		mconv1  = tf.layers.conv2d       (inputs = feature, filters = 128, kernel_size = 7, padding = 'same', activation = tf.nn.relu, name = 'Mconv1' )
		mconv2  = tf.layers.conv2d       (inputs = mconv1 , filters = 128, kernel_size = 7, padding = 'same', activation = tf.nn.relu, name = 'Mconv2' )
		mconv3  = tf.layers.conv2d       (inputs = mconv2 , filters = 128, kernel_size = 7, padding = 'same', activation = tf.nn.relu, name = 'Mconv3' )
		mconv4  = tf.layers.conv2d       (inputs = mconv3 , filters = 128, kernel_size = 7, padding = 'same', activation = tf.nn.relu, name = 'Mconv4' )
		mconv5  = tf.layers.conv2d       (inputs = mconv4 , filters = 128, kernel_size = 7, padding = 'same', activation = tf.nn.relu, name = 'Mconv5' )
		mconv6  = tf.layers.conv2d       (inputs = mconv5 , filters = 128, kernel_size = 1, padding = 'same', activation = tf.nn.relu, name = 'Mconv6' )
		mconv7  = tf.layers.conv2d       (inputs = mconv6 , filters = num, kernel_size = 1, padding = 'same', activation = None      , name = 'Mconv7' )
		return mconv7

def Predict(img):
	num_stages = 6
	feature = VGG19('', img)
	l1 = [FirstStageBranch('CPM_L1', feature, 38)]
	l2 = [FirstStageBranch('CPM_L2', feature, 19)]
	for i in range(2, num_stages + 1):
		stage_input = tf.concat([l1[-1], l2[-1], feature], axis = -1)
		l1.append(StageBranch('stage%d_L1' % i, stage_input, 38))
		l2.append(StageBranch('stage%d_L2' % i, stage_input, 19))
	else:
		return l1[-1], l2[-1]


