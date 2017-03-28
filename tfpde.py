import tensorflow as tf 
import numpy as np 

import PIL
from PIL import Image as im2
from io import BytesIO
from IPython.display import clear_output, Image, display

def DisplayArray(a, fmt = 'jpeg', rng = [0,1]):
	a = (a-rng[0])/float(rng[1]-rng[0])*255
	a = np.uint8(np.clip(a,0,255))
	f = BytesIO
	im2.fromarray(a).save(f,fmt)
	clear_output(wait=True)
	display(Image(data=f.getvalue()))

sess = tf.Session()

#turn 2D array into conv kern
#adding sums together the shape, get shape x 1 x 1
#i.e. 1 input channel 1 output channel 
def make_kernel(a):
	a = np.asarray(a)
	a = a.reshape(list(a.shape)+[1,1])
	return tf.constant(a,dtype=1)