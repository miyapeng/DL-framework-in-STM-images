import matplotlib.pyplot as plt
import numpy as np

def get_list():
    a1 = [[0.1,0.2,0.3,0.4,0.5],[0.3,0.4,0.5,0.6,0.7]]
    l1 = [[0.9,0.8,0.7,0.6,0.5],[0.5,0.4,0.3,0.2,0.1]]
    return a1,l1

def get_name():
    acc_name = ["train_acc", "val_acc"]
    loss_name = ["train_loss", "val_loss"]
    return acc_name, loss_name

def get_color():
    color = ['r','b']
    return color


def plot_figure(a1,idx,color,name,xlabel,ylabel,title,args):
    num = len(a1)
    save_dir = args.fig_dir + "/{}.png".format(idx)
    acc_list = []
    for i in range(num):
        max=np.max(a1[i])
        min = np.min(a1[i])
        acc_list.append(max)
        acc_list.append(min)
    Y_max = np.max(acc_list)
    Y_min = np.min(acc_list)
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis([0,(len(a1[0])-1), Y_min,Y_max])
    for i in range(num):
        plt.plot(a1[i],color=color[i],linewidth=2,label=name[i])
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(save_dir)
    plt.close()

def plot_acc(a1, idx, args):
    color = get_color()
    acc_name, loss_name = get_name()
    plot_figure(a1,idx,color,acc_name,"epoch","acc","{} acc_cuver".format(args.model_name),args)

def plot_loss(l1,idx,args):
    color = get_color()
    acc_name, loss_name = get_name()
    plot_figure(l1, idx, color, loss_name, "epoch", "loss", "{} loss_cuver".format(args.model_name), args)


