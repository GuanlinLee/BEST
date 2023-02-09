import torchvision
import torchvision.transforms as transforms
import torch
import torch.utils.data
import resnet
from torch import nn
from tqdm.auto import tqdm
import time
from datetime import timedelta
import torchattacks
import os
import numpy as np
import argparse
import dataloader
import wrn
import vgg
import mobilenetv2
import attacks
from torch.utils.data import TensorDataset, DataLoader
from logging import getLogger
import logging
import random
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms.functional as TF
class LogFormatter:
	def __init__(self):
		self.start_time = time.time()

	def format(self, record):
		elapsed_seconds = round(record.created - self.start_time)

		prefix = "%s - %s - %s" % (
			record.levelname,
			time.strftime("%x %X"),
			timedelta(seconds=elapsed_seconds),
		)
		message = record.getMessage()
		message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
		return "%s - %s" % (prefix, message) if message else ""
def create_logger(filepath, rank):
	# create log formatter
	log_formatter = LogFormatter()

	# create file handler and set level to debug
	if filepath is not None:
		if rank > 0:
			filepath = "%s-%i" % (filepath, rank)
		file_handler = logging.FileHandler(filepath, "a")
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(log_formatter)

	# create console handler and set level to info
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(log_formatter)

	# create logger and set level to debug
	logger = logging.getLogger()
	logger.handlers = []
	logger.setLevel(logging.DEBUG)
	logger.propagate = False
	if filepath is not None:
		logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	# reset logger elapsed time
	def reset_time():
		log_formatter.start_time = time.time()

	logger.reset_time = reset_time

	return logger
def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
					help='model architecture')
parser.add_argument('-ae', '--ext', metavar='ext_ARCH', default='resnet',
					help='model architecture to extract')
parser.add_argument('--dataset', default='cifar10', type=str,
					help='which dataset used to train')
parser.add_argument('--num', default=5000, type=int,
					help='num of data to split')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='wd')
parser.add_argument('--seed', default=0, type=int,
					help='seed for initializing training. ')
parser.add_argument('--train_num', default=0, type=int,
					help='num of trainset can be use')
parser.add_argument('--class_num', default=10, type=int,
					help='num of classes')
parser.add_argument('--logits', action='store_true',
					help='return logits', default=False)
parser.add_argument('--save1', default='M0.pkl', type=str,
					help='victim model save name')
parser.add_argument('--save', default='M0M1.pkl', type=str,
					help='extracted model save name')
parser.add_argument('--exp', default='exp_test_grad', type=str,
					help='exp name')

parser.add_argument('--method', default='BEST', type=str,
					help='extract method to use')

parser.add_argument('--step', default=10, type=int,
					help='how many steps PGD to generate data')
parser.add_argument('--aug', default=1, type=int,
					help='whether use data augmentation')
parser.add_argument('--eps', default=8, type=int,
					help='eps for UE')
parser.add_argument('--gamma', type=float, default=0.1,
					help='LR is multiplied by gamma on schedule.')
parser.add_argument('--beta',type=float, default=0.1)

args = parser.parse_args()
logger = getLogger()
if not os.path.exists('./extraction/'+args.dataset+'/' +args.save1+'/'+ args.arch +'_to_'+ args.ext+'/'+args.exp):
	os.makedirs('./extraction/'+args.dataset+'/' +args.save1+'/'+ args.arch +'_to_'+ args.ext+'/'+args.exp)
logger = create_logger(
	os.path.join('./extraction/'+args.dataset+'/' +args.save1+'/'+ args.arch +'_to_'+ args.ext+'/'+args.exp + '/', args.exp + ".log"), rank=0
)
logger.info("============ Initialized logger ============")
logger.info(
	"\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
)
args.save = './extraction/'+args.dataset+'/' +args.save1+'/'+ args.arch +'_to_'+ args.ext+'/'+args.exp + '/' +  args.save
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
setup_seed(args.seed)
wd=args.wd
learning_rate=args.lr
epochs=args.epochs
batch_size=args.batch_size
torch.backends.cudnn.benchmark = True
args.class_num = 10 if args.dataset == 'cifar10' else 100
if args.arch == 'resnet':
	n_t = resnet.resnet18(args.dataset).cuda()
	checkpoint = torch.load('./models/'+args.dataset+'/'+args.save1)
	n_t.load_state_dict(checkpoint['state_dict'])
	n_t.eval()
elif args.arch == 'wrn':
	n_t = wrn.WideResNet(num_classes= 10 if args.dataset == 'cifar10' else 100).cuda()
	checkpoint = torch.load('./models/'+args.dataset+'/'+args.save1)
	n_t.load_state_dict(checkpoint['state_dict'])
	n_t.eval()

for param in n_t.parameters():
	param.requires_grad = False

if args.ext == 'resnet':
	n = resnet.resnet18('tiny').cuda()
	checkpoint = torch.load('./pretrained/tiny/resnet_pretrained/resnet.pkl')
	n.load_state_dict(checkpoint['state_dict'])
	n.fc = nn.Linear(512, 10 if args.dataset == 'cifar10' else 100).cuda()
elif args.ext == 'wrn':
	n = wrn.WideResNet(num_classes=200).cuda()
	checkpoint = torch.load('./pretrained/tiny/wrn_pretrained/wrn.pkl')
	n.load_state_dict(checkpoint['state_dict'])
	n.fc = nn.Linear(640, 10 if args.dataset == 'cifar10' else 100).cuda()
elif args.ext == 'vgg':
	n = vgg.vgg19_bn(200).cuda()
	checkpoint = torch.load('./pretrained/tiny/vgg_pretrained/vgg.pkl')
	n.load_state_dict(checkpoint['state_dict'])
	n.classifier = nn.Sequential(
		nn.Linear(512, 512),
		nn.ReLU(True),
		nn.Dropout(),
		nn.Linear(512, 512),
		nn.ReLU(True),
		nn.Dropout(),
		nn.Linear(512, 10 if args.dataset == 'cifar10' else 100),
	).cuda()
elif args.ext == 'mobilenet':
	n = mobilenetv2.mobilenetv2(200).cuda()
	checkpoint = torch.load('./pretrained/tiny/mobilenet_pretrained/mobilenet.pkl')
	n.load_state_dict(checkpoint['state_dict'])
	n.fc = nn.Linear(1280, 10 if args.dataset == 'cifar10' else 100).cuda()

if args.train_num % args.class_num != 0:
	raise ValueError('Cannot handle unbalanced data')
if args.train_num != 0:
	transform_test=transforms.Compose([torchvision.transforms.Resize((32,32)),
									   transforms.ToTensor(),
									   ])
	DATA = dataloader.Data(args.dataset, './data/')
	trainloader, testloader = DATA.data_loader(transform_test, transform_test, 1)
	trainset = []
	trainset_y = []
	count_dict = {}
	for i in range(args.class_num):
		count_dict[str(i)] = 0
	for x, y in trainloader:
		if count_dict[str(y.item())] < (args.train_num // args.class_num):
			trainset.append(x.detach().cpu().numpy())
			trainset_y.append(y.detach().cpu().numpy())
			count_dict[str(y.item())] += 1
	trainset = np.concatenate(trainset).reshape((args.train_num, 3, 32, 32))
	trainset_y = np.concatenate(trainset_y).reshape((args.train_num,))

def data_aug(image):
	image = TF.center_crop(image, [int(32.0 * random.uniform(0.95, 1.0)), int(32.0 * random.uniform(0.95, 1.0))])
	image = TF.resize(image, [32, 32])
	noise = torch.randn_like(image).cuda() * 0.001
	image = torch.clamp(image + noise, 0.0, 1.0)
	if random.uniform(0, 1) > 0.5:
		image = TF.vflip(image)
	if random.uniform(0, 1) > 0.5:
		image = TF.hflip(image)
	angles=[-15, 0, 15]
	angle = random.choice(angles)
	image = TF.rotate(image, angle)
	return image


ext_y = np.load('./data/'+args.dataset+'_train_y_num%d.npy'%args.num)
ext_x = np.load('./data/'+args.dataset+'_train_x_num%d.npy'%args.num)
test_y = np.load('./data/'+args.dataset+'_test_y_num%d.npy'%(5000))
test_x = np.load('./data/'+args.dataset+'_test_x_num%d.npy'%(5000))

if args.train_num != 0:
	ext_x = np.concatenate([ext_x, trainset]).reshape((args.num+args.train_num, 3, 32, 32))
	ext_y = np.concatenate([ext_y, trainset_y]).reshape((args.num+args.train_num, ))
ext_dataset = TensorDataset(torch.from_numpy(ext_x),torch.from_numpy(ext_y))
ext_dataloader = DataLoader(ext_dataset, batch_size=batch_size,
						   shuffle=True, num_workers=0)

test_dataset = TensorDataset(torch.from_numpy(test_x),torch.from_numpy(test_y))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
						   shuffle=True, num_workers=0)

Loss = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(n.parameters() ,momentum=args.momentum,
							lr=learning_rate,weight_decay=wd)#+ optimize_parameters
milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
scheduler = MultiStepLR(optimizer,milestones=milestones,gamma=args.gamma)
train_clean_acc = []
train_adv_acc = []
test_clean_acc = []
test_adv_acc = []
best_eval_acc = 0.0

for epoch in range(epochs):
	loadertrain = tqdm(ext_dataloader, desc='{} E{:03d}'.format('train', epoch), ncols=0, total=len(ext_dataloader))
	epoch_loss = 0.0
	clean_acc = 0.0
	adv_acc = 0.0
	total=0.0
	for (x_train, y_train) in loadertrain:
		x_train, y_train = x_train.cuda(), y_train.cuda()
		if args.aug == 1:
			x_train = data_aug(x_train)
		n.train()
		if args.method == 'BEST':
			y, y_adv, loss = attacks.BEST(x_train, n, n_t, Loss, args)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		epoch_loss += loss.data.item()
		_, predicted = torch.max(y.data, 1)
		_, predictedadv = torch.max(y_adv.data, 1)
		total += y_train.size(0)
		clean_acc  += predicted.eq(y_train.data).cuda().sum()
		adv_acc += predictedadv.eq(y_train.data).cuda().sum()
		fmt = '{:.4f}'.format
		loadertrain.set_postfix(loss=fmt(loss.data.item()),
								clean_acc=fmt(clean_acc.item() / total * 100),
								adv_acc=fmt(adv_acc.item() / total * 100))
	train_clean_acc.append(clean_acc.item() / total * 100)
	train_adv_acc.append(adv_acc.item() / total * 100)
	scheduler.step()
	if (epoch) % 1 == 0:
		Loss_test = nn.CrossEntropyLoss().cuda()
		test_loss_cl = 0.0
		test_loss_adv = 0.0
		correct_cl = 0.0
		correct_adv = 0.0
		total = 0.0
		pgd_eval = torchattacks.PGD(n, eps=8.0/255.0, steps=20)
		n.eval()
		loadertest = tqdm(test_dataloader, desc='{} E{:03d}'.format('test', epoch), ncols=0)
		with torch.enable_grad():
			for x_test, y_test in loadertest:
				x_test, y_test = x_test.cuda(), y_test.cuda()
				x_adv = pgd_eval(x_test, y_test)
				n.eval()
				y_pre = n(x_test)
				y_adv = n(x_adv)
				loss_cl = Loss_test(y_pre, y_test)
				loss_adv = Loss_test(y_adv, y_test)
				test_loss_cl += loss_cl.data.item()
				test_loss_adv += loss_adv.data.item()
				_, predicted = torch.max(y_pre.data, 1)
				_, predicted_adv = torch.max(y_adv.data, 1)
				total += y_test.size(0)
				correct_cl += predicted.eq(y_test.data).cuda().sum()
				correct_adv += predicted_adv.eq(y_test.data).cuda().sum()
				fmt = '{:.4f}'.format
				loadertest.set_postfix(loss_cl=fmt(loss_cl.data.item()),
									   loss_adv=fmt(loss_adv.data.item()),
									   acc_cl=fmt(correct_cl.item() / total * 100),
									   acc_adv=fmt(correct_adv.item() / total * 100))
		test_clean_acc.append(correct_cl.item() / total * 100)
		test_adv_acc.append(correct_adv.item() / total * 100)
		if correct_adv.item() / total * 100 > best_eval_acc:
			best_eval_acc = correct_adv.item() / total * 100
			checkpoint = {
				'state_dict': n.state_dict(),
				'epoch': epoch
			}
			torch.save(checkpoint, args.save)
checkpoint = {
	'state_dict': n.state_dict(),
	'epoch': epoch
	}
torch.save(checkpoint, args.save+'last.pkl')

np.save(args.save+'_train_acc_cl.npy', train_clean_acc)
np.save(args.save+'_train_acc_adv.npy', train_adv_acc)
np.save(args.save+'_test_acc_cl.npy', test_clean_acc)
np.save(args.save+'_test_acc_adv.npy', test_adv_acc)
