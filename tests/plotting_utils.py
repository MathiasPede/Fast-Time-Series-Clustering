import numpy as np
from matplotlib import pyplot as plt


def histogram_plot(distribution):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    numBins = 50
    ax.hist(distribution, numBins, color='green', alpha=0.8)
    plt.show()


def bar_plot(data, xas, names, yname='', xname='', title='', yerr=None, yscale='log', colors=('blue','orange', 'green', 'red')):
    plt.figure()
    var_amount = len(data)
    x_length = len(xas)
    ind = np.arange(x_length)
    width = (0.7 / var_amount)
    for i in range(var_amount):
        plt.bar(ind + i*width, data[i], width, yerr=yerr[i], label=names[i], color=colors[i])
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.yscale(yscale)
    plt.grid(True)
    plt.title(title)
    plt.xticks(ind + width / 2, xas)
    plt.legend(loc='best')
    plt.show()


def scatter_plot(y_data, x_data, yname='', xname='', title='', yscale='log', marker="+", xrange=None, yrange=None, regression=False):
    plt.figure()
    fig3, ax3 = plt.subplots()
    ax3.plot(x_data, y_data, color='blue', marker=marker, ls="")
    if regression:
        ax3.plot(np.unique(x_data), np.poly1d(np.polyfit(x_data, y_data, 1))(np.unique(x_data)), color="g")
    if yrange is None:
        ax3.autoscale(enable=True, axis="y", tight=False)
    else:
        plt.ylim(yrange)
    if xrange is not None:
        plt.xlim(xrange)
    plt.title(title)
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.yscale(yscale)
    plt.grid(True)
    plt.show()


def multiple_scatter_plot(y_datas, x_datas, labels, xname='', yname='', title='', colors=('blue', 'red', 'green', 'orange'), yscale='log', regression=False, marker='+'):
    plt.figure()
    fig3, ax3 = plt.subplots()
    for i in range(len(y_datas)):
        x = x_datas[i]
        y = y_datas[i]
        ax3.plot(x, y, color=colors[i], marker=marker, ls="", label=labels[i])
        if regression:
            ax3.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color=colors[i])
    ax3.autoscale(enable=True, axis="y", tight=False)
    ax3.legend()
    plt.title(title)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.yscale(yscale)
    plt.grid(True)
    plt.show()


def multiple_plot(y_datas, x_datas, labels, xname='', yname='', title='', forms=('b-.', 'r-.', 'g-.', 'm-.', "y-."), yscale='log'):
    plt.figure()
    fig3, ax3 = plt.subplots()
    for i in range(len(y_datas)):
        plt.plot(x_datas[i], y_datas[i], forms[i], label=labels[i])
    ax3.autoscale(enable=True, axis="y", tight=False)
    ax3.legend()
    plt.title(title)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.yscale(yscale)
    plt.grid(True)
    plt.show()


def text_scatter_plot(y_data, x_data, yname='', xname='', title='', yscale='log', marker="+", xrange=None, yrange=None, names=None, regression=False, best_line=False):
    plt.figure()
    fig3, ax3 = plt.subplots()
    ax3.plot(x_data, y_data, color='blue', marker=marker, ls="")
    for i in range(len(y_data)):
        plt.text(x_data[i]+0.002, y_data[i], names[i])
    if regression:
        ax3.plot(np.unique(x_data), np.poly1d(np.polyfit(x_data, y_data, 1))(np.unique(x_data)), color="g")
    if best_line:
        xs = [0.0,1.0]
        ys = [0.0,1.0]
        ax3.plot(np.unique(xs), np.poly1d(np.polyfit(xs, ys, 1))(np.unique(xs)), color="r")
    if yrange is None:
        ax3.autoscale(enable=True, axis="y", tight=False)
    else:
        plt.ylim(yrange)
    if xrange is not None:
        plt.xlim(xrange)
    plt.title(title)
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.yscale(yscale)
    plt.grid(True)
    plt.show()