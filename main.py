import torch
from torch import nn, optim, autograd
import numpy as np
import visdom
import random
from matplotlib import pyplot as plt


h_dim = 400
batchsize = 512
viz = visdom.Visdom()

class Generator(nn.Module):

    def __init__(self):
        super(Generator,self).__init__()

        self.net = nn.Sequential(
            # z: [batch,2] => [batch,2] 2 is random setted
            nn.Linear(2,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
        )
    def forward(self,z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
            #sigmoid compress x to [0,1]
        )
    def forward(self,x):
        output = self.net(x)
        return output.view(-1)


def data_generator():
    '''
    8-gaussian mixture models
    :return:
    '''
    scale = 2.
    centers = [
        (1,0),
        (-1,0),
        (0,1),
        (0,-1),
        (1./np.sqrt(2),1./np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x,y in centers]

    #数据生成迭代器
    while True:
        dataset = []

        for i in range(batchsize):

            point = np.random.randn(2)*0.02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)

        dataset = np.array(dataset).astype(np.float32)
        dataset /= 1.414
        yield dataset

def genarate_image(D,G,xr,epoch):
    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS,N_POINTS,2),dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1,2))

    with torch.no_grad():
        points = torch.Tensor(points).cuda()
        disc_map = D(points).cpu().numpy()
        x = y = np.linspace(-RANGE, RANGE, N_POINTS)
        cs = plt.contour(x,y,disc_map.reshape((len(x),len(y))).transpose())
        plt.clabel(cs,inline=1,fontsize=10)

    with torch.no_grad():
        z = torch.randn(batchsize,2).cuda()
        samples = G(z).cpu().numpy()
        xr = next(data_generator())
    plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    viz.matplot(plt,win='contour',opts=dict(title='p(x):%d'%epoch))


def grandient_penalty(D,xr,xf):
    t = torch.rand(batchsize,1).cuda()
    t = t.expand_as(xr)
    mid = t * xf +(1-t) * xr
    mid.requires_grad_()

    pred = D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True,
                          only_inputs=True)[0]

    gp = torch.pow(grads.norm(2, dim=1)-1, 2).mean()
    return gp


def main():

    torch.manual_seed(23)
    np.random.seed(23)

    data_iter = data_generator()
    x = next(data_iter)
    #[batchsize,2]表示x1,x2

    G = Generator().cuda()
    D = Discriminator().cuda()
    #print(G)
    #print(D)
    optim_G = optim.Adam(G.parameters(),lr=5e-4,betas=(0.5,0.9))
    optim_D = optim.Adam(D.parameters(),lr=5e-4,betas=(0.5,0.9))

    loss = viz.line([[0.0, 0.0]], [0.], win='loss', opts=dict(title='loss', legend=['D', 'G']))

    for epoch in range(50000):

        #1.train Discriminator firstly
        for i in range(5):
            #1.1 train on real data
            xr = next(data_iter)
            xr = torch.from_numpy(xr).cuda()
            #[b,z]=>[b,1]
            predr = D(xr)
            #max predr
            lossr = -predr.mean()

            #1.2 train on fake data
            #[b,]
            z = torch.randn(batchsize,2).cuda()
            xf = G(z).detach()  #tf.stop_gradient
            pref = D(xf)
            lossf = pref.mean()

            #1.3 gradient penalty
            gp = grandient_penalty(D,xr,xf.detach())

            #1.4 aggregate all
            loss_D = lossr + lossf + 0.1 * gp

            #1.5 optimize
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()


        #2.train Generator secondly
        z = torch.randn(batchsize,2).cuda()
        xf = G(z)
        pref = D(xf)
        loss_G = -pref.mean()

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        step = 0
        if epoch % 100 == 0:
            step += 1
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win=loss, update='append')
            print(loss_D.item(), loss_G.item(),epoch)
            genarate_image(D,G,xr,epoch)


if __name__ == '__main__':
    main()

