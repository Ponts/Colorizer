import fnmatch
import threading
import os
import numpy as np
import tensorflow as tf
import random



# Override this class for specific datasets
class DataLoader():
	def __init__(self, shape, batchSize, prefetchSize = None):
		if prefetchSize == None:
			self.prefetchSize = batchSize
		else:
			self.prefetchSize = prefetchSize
		self.shape = shape
		self.batchSize = batchSize	


	def loadData(self):
		pass

	def deQue(self):
		data = self.dataset.batch(self.batchSize)
		data = self.dataset.prefetch(self.prefetchSize)
		return data