from torch.nn import functional as F
import torch.nn
from torch.autograd import Variable

def BEST(x, n, n_t, Loss, args):
	y = n(x)
	n.eval()
	pes_label = torch.full((x.size(0), args.class_num), 1/args.class_num).cuda()
	# generate uncertain example
	x_adv = x.detach() + torch.zeros_like(x).uniform_(-args.eps/255.0, args.eps/255.0)
	for _ in range(args.step):
		x_adv.requires_grad_()
		with torch.enable_grad():
			loss = - F.kl_div(F.log_softmax(n(x_adv), dim=1),
							  F.softmax(pes_label, dim=1), size_average=False)
		grad = torch.autograd.grad(loss, [x_adv])[0]
		x_adv = x_adv.detach() + 2./255. * torch.sign(grad.detach())
		x_adv = torch.min(torch.max(x_adv, x - args.eps/255.0), x + args.eps/255.0)
		x_adv = torch.clamp(x_adv, 0.0, 1.0)
	n.train()
	x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
	y_adv = n(x_adv)
	y_t_adv = n_t(x_adv).detach()
	y_t_adv = y_t_adv.topk(1, 1)[1].reshape(-1)
	loss = Loss(y_adv, y_t_adv)
	return y, y_adv, loss





