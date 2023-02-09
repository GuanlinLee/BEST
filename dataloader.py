import torchvision
import torch
from torch.utils.data.dataset import Dataset



class Data:
	def __init__(self, dataset, path):
		self.dataset = dataset
		self.path = path
	def data_loader(self, train_trans, test_trans, batch_size):
		if self.dataset == 'cifar10':
			trainset = torchvision.datasets.CIFAR10(root=self.path, train=True, download=True, transform=train_trans)
			testset = torchvision.datasets.CIFAR10(root=self.path, train=False, download=True, transform=test_trans)
		elif self.dataset == 'cifar100':
			trainset = torchvision.datasets.CIFAR100(root=self.path, train=True, download=True, transform=train_trans)
			testset = torchvision.datasets.CIFAR100(root=self.path, train=False, download=True, transform=test_trans)
		elif self.dataset == 'svhn':
			trainset = torchvision.datasets.SVHN(root=self.path, split='train', download=True, transform=train_trans)
			testset = torchvision.datasets.SVHN(root=self.path, split='test', download=True, transform=test_trans)
		else:
			ValueError('Unsupported dataset')
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
													 shuffle=True, num_workers=0)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
										 shuffle=False, num_workers=0)

		return trainloader, testloader


