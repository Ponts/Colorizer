import fnmatch
import threading
import os
import numpy as np
import tensorflow as tf
import random
import loader

class GANTrainer():
	def __init__(self, dataLoader, generator, discriminator, noiseSize=100, noiseVar=0.1):
		self.dataLoader = dataLoader
		self.G = generator
		self.D = discriminator
		self.noiseSize = noiseSize
		self.gOptimizer = tf.keras.optimizers.Adam(1e-4)
		self.dOptimizer = tf.keras.optimizers.Adam(1e-4)


	def loss(self, real, fake):
		dLoss = -tf.reduce_mean(real) + tf.reduce_mean(fake)
		gLoss = -tf.reduce_mean(fake)
		return gLoss, dLoss


	@tf.function
	def trainStep(self, real):
		noise = tf.random.normal([tf.shape(real)[0], self.noiseSize])

		with tf.GradientTape() as gTape, tf.GradientTape() as dTape:
			fakes = self.G.forward(noise, training=True)

			fakeLogits = self.D.forward(fake)
			realLogits = self.D.forward(realLogits)

			dLoss, gLoss = self.loss(realLogits, fakeLogits)
		genGrads = gTape.gradient(gLoss, self.G.trainable_variables)
		discGrads = dTape.gradient(dLoss, self.D.trainable_variables)

		self.gOptimizer.apply_gradients(zip(genGrads, self.G.trainable_variables))
		self.dOptimizer.apply_gradients(zip(discGrads, self.D.trainable_variables))


	def train(self, epochs):
		for epoch in range(epochs):
			for batch in self.dataLoader.dataset:
				#print(tf.shape(batch['image']))
				self.trainStep(batch['image'])
				return
