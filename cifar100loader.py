import fnmatch
import threading
import os
import numpy as np
import tensorflow as tf
import random
import loader
import tensorflow_datasets as tfds

class Cifar100Loader(loader.DataLoader):
	def __init__(self, bufferSize, batchSize):
		super().__init__(np.shape((32,32,3)), batchSize*2)
		self.batchSize = batchSize
		self.bufferSize = bufferSize
		self.loadData()

		self.deQue()
		

	def loadData(self):
		self.dataset = tfds.load(name="cifar100", split=tfds.Split.TRAIN)
		self.dataset.shuffle(self.bufferSize).batch(self.batchSize)

	def deQue(self):
		super().deQue()

if __name__=="__main__":
	loader = Cifar100Loader(100, 100)