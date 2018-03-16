# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:22:44 2018
@author: Spandan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os, time, sys,itertools,pickle,imageio
import matplotlib.pyplot as plt
import statistics as st
#%%
class generator(nn.Module):
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.main = nn.Sequential(nn.ConvTranspose2d(100, d*8, 4, 1, 0),
                                  nn.BatchNorm2d(d*8),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
                                  nn.BatchNorm2d(d*4),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(d*4 , d*2, 4, 2, 1),
                                  nn.BatchNorm2d(d*2),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(d*2, d, 4, 2, 1),
                                  nn.BatchNorm2d(d),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(d, 3, 4, 2, 1),
                                  nn.Tanh())
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,input_dat):
        x = self.main(input_dat)
        return x
#gen = generator()
#gen = gen.cuda()
#print(gen)
#%%
class discriminator(nn.Module):
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(3, d, 4, 2, 1),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(d, d*2, 4, 2, 1),
                                  nn.BatchNorm2d(d*2),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(d*2, d*4, 4, 2, 1),
                                  nn.BatchNorm2d(d*4),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(d*4, d*8, 4, 2, 1),
                                  nn.BatchNorm2d(d*8),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(d*8, 1, 4, 1, 0),
                                  nn.Sigmoid())

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,input_dat):
        x = self.main(input_dat)
        return x
#disc = discriminator()
#disc = disc.cuda()
#print(disc)
#%%
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
#%%
fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise
fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)
#%%
def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
#%%
def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
#%%
if __name__ == '__main__':
	batch_size = 4096
	lr = 0.0002
	train_epoch = 100
	g_per_epoch = 1
	prev_G_loss = 9999999.00
	incG = 0

	#Load Data
	img_size = 64
	isCrop = True
	if isCrop:
		transform = transforms.Compose([
			transforms.Scale(108),
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
		])
	else:
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
		])
	'''
	transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])
	'''
	 #%%
	data_dir = 'D:\\Spandan\\cele'

	dset = datasets.ImageFolder(data_dir, transform)
	train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True,num_workers = 6)
	temp = plt.imread(train_loader.dataset.imgs[0][0])
	if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
		sys.stderr.write('Uncompatible image size...USe 64x64')
		sys.exit(1)
	#%%
	# network
	count = 0
	G = generator(128)
	D = discriminator(128)
	G.weight_init(mean=0.0, std=0.02)
	count = count +1
	print(count)
	D.weight_init(mean=0.0, std=0.02)

	G.cuda()
	D.cuda()

	# Binary Cross Entropy loss
	BCE_loss = nn.BCELoss()

	# Adam optimizer
	G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
	D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
	#%%
	print('Generator :')
	print(G)
	print('Discriminator :')
	print(D)
	#%%
	# results save folder
	save_path = 'D:\\Spandan\\cel3\\'
	if not os.path.isdir(save_path + 'CelebA_DCGAN_results'):
		print('Making results folder')
		os.mkdir(save_path+'CelebA_DCGAN_results')
	if not os.path.isdir(save_path+'CelebA_DCGAN_results\\Random_results'):
		print('Making random results folder')
		os.mkdir(save_path+'CelebA_DCGAN_results\\Random_results')
	if not os.path.isdir(save_path+'CelebA_DCGAN_results\\Fixed_results'):
		print('Making fixed folder')
		os.mkdir(save_path+'CelebA_DCGAN_results\\Fixed_results')
	#%%
	a = save_path + 'CelebA_DCGAN_results\\'
	train_hist = {}
	train_hist['D_losses'] = []
	train_hist['G_losses'] = []
	train_hist['per_epoch_ptimes'] = []
	train_hist['total_ptime'] = []

	print('Training start!')
	start_time = time.time()
	for epoch in range(train_epoch):
		D_losses = []
		G_losses = []

		# learning rate decay
		if (epoch%20) == 0 and (epoch !=0):
			G_optimizer.param_groups[0]['lr'] /= 5
			D_optimizer.param_groups[0]['lr'] /= 5
			print("learning rate change!")


		num_iter = 0

		epoch_start_time = time.time()
		for x_, _ in train_loader:
			# train discriminator D
			D.zero_grad()
			if isCrop:
				x_ = x_[:, :, 22:86, 22:86]

			mini_batch = x_.size()[0]

			y_real_ = torch.ones(mini_batch)
			y_fake_ = torch.zeros(mini_batch)

			x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
			D_result = D(x_).squeeze()
			D_real_loss = BCE_loss(D_result, y_real_)

			z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
			z_ = Variable(z_.cuda())
			G_result = G(z_)

			D_result = D(G_result).squeeze()
			D_fake_loss = BCE_loss(D_result, y_fake_)
			D_fake_score = D_result.data.mean()

			D_train_loss = D_real_loss + D_fake_loss

			D_train_loss.backward()
			D_optimizer.step()

			D_losses.append(D_train_loss.data[0])

			# train generator G
			#for _ in range(g_per_epoch):
			G.zero_grad()
			z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
			z_ = Variable(z_.cuda())

			G_result = G(z_)
			D_result = D(G_result).squeeze()
			G_train_loss = BCE_loss(D_result, y_real_)
			G_train_loss.backward()
			G_optimizer.step()

			G_losses.append(G_train_loss.data[0])

			num_iter += 1
			print('Running Epoch : [{}/{}]:: Discrim. Loss: {}, Gen Loss: {}'.format((epoch + 1), train_epoch, st.mean(D_losses),
																 st.mean(G_losses)), end = "\r")
			#sys.stdout.write("\033[F")
		epoch_end_time = time.time()
		per_epoch_ptime = (epoch_end_time - epoch_start_time)/60
		print('Epoch_Summary : [{}/{}] - time: {:.2f} min, Discrim. Loss: {}, Gen Loss: {}'.format((epoch + 1), train_epoch, per_epoch_ptime, st.mean(D_losses),
																  st.mean(G_losses)))
		print('-'*100)
		'''if epoch == 0:
			prev_G_loss = st.mean(G_losses)
		elif prev_G_loss < st.mean(G_losses):
			incG = incG +1
		if incG > 1:
			g_per_epoch = g_per_epoch + 1
			incG = 0
		'''

		#time.sleep(5)
		p = a + 'Random_results\\CelebA_DCGAN_' + str(epoch + 1) + '.png'
		fixed_p = a + 'Fixed_results\\CelebA_DCGAN_' + str(epoch + 1) + '.png'
		show_result((epoch+1), save=True, path=p, isFix=False)
		show_result((epoch+1), save=True, path=fixed_p, isFix=True)
		train_hist['D_losses'].append(st.mean(D_losses))
		train_hist['G_losses'].append(st.mean(G_losses))
		train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

	end_time = time.time()
	total_ptime = end_time - start_time
	train_hist['total_ptime'].append(total_ptime)

	#print("Avg per epoch ptime: {}, total {} epochs ptime: {}" .format(torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
	print("Training finished!... save training results")
	torch.save(G.state_dict(), a + "generator_param.pkl")
	torch.save(D.state_dict(), a + "discriminator_param.pkl")
	with open( a + 'train_hist.pkl', 'wb') as f:
		pickle.dump(train_hist, f)
	#%%
	show_train_hist(train_hist, save=True, path=a + 'CelebA_DCGAN_train_hist.png')
	images = []
	for e in range(train_epoch):
		img_name = a + 'Fixed_results\\CelebA_DCGAN_' + str(e + 1) + '.png'
		images.append(imageio.imread(img_name))
	imageio.mimsave(a + 'generation_animation.gif', images, fps=5)
