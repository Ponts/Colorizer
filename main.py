import tensorflow as tf
import cifar100loader
import ganTrainer
import model

if __name__=="__main__":
	loader = cifar100loader.Cifar100Loader(100,100)
	d = model.Model((32,32,3),1)
	g = model.Model((100),(32,32,3))
	trainer = ganTrainer.GANTrainer(loader, g, d)

	trainer.train(1)
