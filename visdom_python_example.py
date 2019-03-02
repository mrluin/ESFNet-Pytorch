from visdom import Visdom
import torch
import torchvision.transforms.functional as TF
from configs.config import MyConfiguration
from data.dataset import MyDataset
from PIL import Image
import numpy as np

viz = Visdom()
"""
# Visdom Arguments only for needed
server : host name of visdom server 'http://localhost'
port : the port for visdom server 8097
base_url : the base visdom server url /
env : default environment to plot to when no env is provided . main
raise_exceptions: Raise exceptions upon failure rather than printing the . True
log_to_filename : log all plotting and updating events, used for later . None

"""

'''
""" for image visualization, creating a dataloader do transformation except normalization"""
    image visualization need image dtype is torch.float32
    
    
image = Image.open('0.tif')
# though .to_tensor scale value in [0,1] interval but it works
image = TF.to_tensor(image)
# title for window title and caption for image, search for title not for the caption
viz.image(image, opts = dict(title='data!', caption='data'))
# images receive BCHW for a batchsize
#viz.images()
'''
'''
config = MyConfiguration()
dataset = MyDataset(config=config, subset='train')
data, target = next(iter(dataset))

viz.image(data, opts=dict(title='data', caption='data.1'))
print(data.dtype)
print(target.dtype)
target = target.float()
viz.image(target, opts=dict(title='target', caption='target.1'))
'''

#textwindow = viz.text('hello world!')
#updatetextwindow = viz.text('hello world! more text should be here')

#viz.text('and here it is', win=updatetextwindow, append=True)

'''
# text window with callbacks
txt = 'This is a write demo notepad. Type below. Delete clear text: <br>'
callback_text_window = viz.text(txt)
'''
'''
# matplotlib demo:
import matplotlib.pyplot as plt
plt.plot([1, 23, 2, 4])
plt.ylabel('some numbers')
viz.matplot(plt)
'''
'''
# what about save_path ? what does the viz.save means ?
viz.save(envs=['main'])
'''

'''
# vis.scatter plots 散点图
Y = np.random.rand(100)
old_scatter = viz.scatter(
    X=np.random.rand(100, 2),
    Y=(Y[Y > 0] + 1.5).astype(int),
    opts=dict(
        legend=['Didnt', 'Update'],
        xtickmin=-50,
        xtickmax=50,
        xtickstep=0.5,
        ytickmin=-50,
        ytickmax=50,
        ytickstep=0.5,
        markersymbol='cross-thin-open',
    ),
)

viz.update_window_opts(
    win=old_scatter,
    opts=dict(
        legend=['Apples', 'Pears'],
        xtickmin=0,
        xtickmax=1,
        xtickstep=0.5,
        ytickmin=0,
        ytickmax=1,
        ytickstep=0.5,
        markersymbol='cross-thin-open',
    ),
)
'''
'''
# line plot
viz.line(Y = np.random.rand(10), opts=dict(showlegend=True))
'''

'''
# bar plot
viz.bar(X=np.random.rand(20))
viz.bar(X=np.abs(np.random.rand(5,3)),
        opts=dict(stacked=True, legend=['Facebook', 'Google', 'Twitter'], rownames=['2012','2013','2014','2015','2016']))
viz.bar(X=np.random.rand(20, 3),
        opts=dict(stacked=False, legend=['The Netherlands', 'France', 'United States']))
'''
'''
#histogram
viz.histogram(X = np.random.rand(10000), opts=dict(numbins=20))
'''
'''
# heatmap 接的是一个矩阵 
viz.heatmap(
    X=np.outer(np.arange(1, 6), np.arange(1, 11)),
    opts=dict(
        columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
        colormap='Electric',
    )
)
'''
'''
# contour
x = np.tile(np.arange(1, 101), (100, 1))
y = x.transpose()
X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
viz.contour(X=X, opts=dict(colormap='Viridis'))

# surface
viz.surf(X=X, opts=dict(colormap='Hot'))
'''
'''
# line plots
viz.line(Y=np.random.rand(10), opts=dict(showlegend=True))

Y = np.linspace(-5, 5, 100)
viz.line(
    Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),
    X=np.column_stack((Y, Y)),
    opts=dict(markers=False),
)
'''
'''
# line updates
win = viz.line(
    X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
    Y=np.column_stack((np.linspace(5, 10, 10),
                       np.linspace(5, 10, 10) + 5)),
)
viz.line(
    X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
    Y=np.column_stack((np.linspace(5, 10, 10),
                       np.linspace(5, 10, 10) + 5)),
    win=win,
    update='append'
)
viz.line(
    X=np.arange(21, 30),
    Y=np.arange(1, 10),
    win=win,
    name='2',
    update='append'
)
viz.line(
    X=np.arange(1, 10),
    Y=np.arange(11, 20),
    win=win,
    name='delete this',
    update='append'
)
viz.line(
    X=np.arange(1, 10),
    Y=np.arange(11, 20),
    win=win,
    name='4',
    update='insert'
)
viz.line(X=None, Y=None, win=win, name='delete this', update='remove')

# np.linspace(from, to , size)  np.arange(from, to)

win = viz.line(
    X=np.column_stack((
        np.arange(0, 10),
        np.arange(0, 10),
        np.arange(0, 10),
    )),
    Y=np.column_stack((
        np.linspace(5, 10, 10),
        np.linspace(5, 10, 10) + 5,
        np.linspace(5, 10, 10) + 10,
    )),
    opts={
        'dash': np.array(['solid', 'dash', 'dashdot']),
        'linecolor': np.array([
            [0, 191, 255],
            [0, 191, 255],
            [255, 0, 0],
        ]),
        'title': 'Different line dash types'
    }
)

viz.line(
    X=np.arange(0, 10),
    Y=np.linspace(5, 10, 10) + 15,
    win=win,
    name='4',
    update='insert',
    opts={
        'linecolor': np.array([
            [255, 0, 0],
        ]),
        'dash': np.array(['dot']),
    }
)

Y = np.linspace(0, 4, 200)
win = viz.line(
    Y=np.column_stack((np.sqrt(Y), np.sqrt(Y) + 2)),
    X=np.column_stack((Y, Y)),
    opts=dict(
        fillarea=True,
        showlegend=False,
        width=800,
        height=800,
        xlabel='Time',
        ylabel='Volume',
        ytype='log',
        title='Stacked area plot',
        marginleft=30,
        marginright=30,
        marginbottom=80,
        margintop=30,
    ),
)
'''
'''
import math
# stemplot
Y = np.linspace(0, 2 * math.pi, 70)
X = np.column_stack((np.sin(Y), np.cos(Y)))
viz.stem(
    X=X,
    Y=Y,
    opts=dict(legend=['Sine', 'Cosine'])
)
'''
'''
# quiver plot
X = np.arange(0, 2.1, .2)
Y = np.arange(0, 2.1, .2)
X = np.broadcast_to(np.expand_dims(X, axis=1), (len(X), len(X)))
Y = np.broadcast_to(np.expand_dims(Y, axis=0), (len(Y), len(Y)))
U = np.multiply(np.cos(X), Y)
V = np.multiply(np.sin(X), Y)
viz.quiver(
    X=U,
    Y=V,
    opts=dict(normalize=0.9),
)
'''
'''
# mesh plot
x = [0, 0, 1, 1, 0, 0, 1, 1]
y = [0, 1, 1, 0, 0, 1, 1, 0]
z = [0, 0, 0, 0, 1, 1, 1, 1]
X = np.c_[x, y, z]
i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
Y = np.c_[i, j, k]
viz.mesh(X=X, Y=Y, opts=dict(opacity=0.5))
'''

loss_window = viz.line(
    X=torch.stack((torch.zeros(1), torch.zeros(1)),dim=1),
    Y=torch.stack((torch.zeros(1), torch.zeros(1)),dim=1),
    opts = dict(title='title',
                #width=1000,
                #height=500,
                showlegend=True,
                legend=['training_loss', 'valid_loss'],
                xtype= 'linear',
                #xlabel='xlabel',
                xtick = True,
                #xtickmin= -10,
                #xtickmax= 10,
                xtickvals = 10,
                xticklabels ='xticklabels',
                xtickstep=1,
                ytype='linear',
                #ylabel='ylabel',
                ytick=True,
                #ytickmin=-10,
                #ytickmax=10,
                #ytickvals=5,
                yticklabels='yticklabels',
                #ytickstep=1,
                #marginleft=20,
                #marginright=30,
                #margintop=40,
                #marginbottom=50)
                )
)



# 未知属性 xtickvals xticklabels



'''
# loss and metrics curve in training phase

loss_window = viz.line(
    Y =torch.zeros((1)).cpu(),
    X = torch.zeros((1)).cpu(),
    opts = dict(xlabel='epoch', ylabel='loss',title='loss', legend=['loss'])
)

'''
# x : epoch torch.ones(1)*epoch y:loss or other metrics
for epoch in range(100):
    train_loss = torch.randn(1)
    valid_loss = torch.randn(1)

    viz.line(X=torch.stack((torch.ones(1)*epoch, torch.ones(1)*epoch),1),
             Y=torch.stack((torch.Tensor(train_loss), torch.Tensor(valid_loss)),1),
             win=loss_window, update='append')

'''
Y = np.linspace(-5,5,100)
viz.line(
    Y = np.column_stack((Y*Y, np.sqrt(Y+5))),
    X = np.column_stack((Y,Y)),
    opts = dict(markers=False)
)
'''

# 每个epoch的 training loss and validation loss
