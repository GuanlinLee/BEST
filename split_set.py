import dataloader
import numpy as np
import argparse
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', type=str,
					help='which dataset used to train')
parser.add_argument('--num', default=5000, type=int,
					help='num of data to split')
parser.add_argument('--class_num', default=10, type=int,
					help='num of classes')
args = parser.parse_args()

if args.num % args.class_num != 0:
	raise ValueError('Cannot handle unbalanced data')
transform_test=transforms.Compose([torchvision.transforms.Resize((32,32)),
								   transforms.ToTensor(),
								   ])
DATA = dataloader.Data(args.dataset, './data/')
trainloader, testloader = DATA.data_loader(transform_test, transform_test, 1)

trainset = []
trainset_y = []
testset = []
testset_y = []

count_dict = {}
for i in range(args.class_num):
	count_dict[str(i)] = 0
for x, y in testloader:
	if count_dict[str(y.item())] < (args.num // args.class_num):
		trainset.append(x.detach().cpu().numpy())
		trainset_y.append(y.detach().cpu().numpy())
		count_dict[str(y.item())] += 1
	else:
		testset.append(x.detach().cpu().numpy())
		testset_y.append(y.detach().cpu().numpy())

trainset = np.concatenate(trainset).reshape((args.num, 3, 32, 32))
trainset_y = np.concatenate(trainset_y).reshape((args.num,))

testset = np.concatenate(testset).reshape((-1, 3, 32, 32))
testset_y = np.concatenate(testset_y).reshape((-1,))

np.save('./data/'+args.dataset+'_train_x_num%d.npy'%args.num, trainset)
np.save('./data/'+args.dataset+'_train_y_num%d.npy'%args.num, trainset_y)

np.save('./data/'+args.dataset+'_test_x_num%d.npy'%args.num, testset)
np.save('./data/'+args.dataset+'_test_y_num%d.npy'%args.num, testset_y)
