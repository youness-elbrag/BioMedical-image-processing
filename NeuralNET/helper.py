import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing as mp
import torch
from time import time


def plot(train_plot, epochs, test_plot, typePlot=False):
    fig, (ax, ax1) = plt.subplots(1, 2)
    fig.suptitle('Vertically stacked subplots')
    ax.plot(train_plot, epochs, label="training")
    ax.legend()
    ax1.plot(test_plot, epochs, label="testing")
    ax1.legend()
    if typePlot == True:
        image_ = make_grid(image[:12], nrow=12)
        label_ = label[:12].numpy()
        plt.xlabel(label_)
        plt.imshow(np.transpose(image_.numpy(), (1, 2, 0)), cmap="gray")
        plt.show()


def TimeCounter_Process(function):
    def warp_func(*args, **kwargs):
        start_time = time()
        result = function(*args, **kwargs)
        end_time = time() 
        print(f'Function {function.__name__!r} executed in {(end_time-start_time ):.4f}s')
        return result
    return warp_func 



@TimeCounter_Process
def Proceesing_Parallel_Training(*tasks, Parallel_Training_GPU=False):
    Processors = []
    Cpu_Counts = mp.cpu_count()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if Parallel_Training_GPU == True:
        for task in tasks:
            Pool_task_ = mp.Process(target=task)
            Pool_task_.start()
            Processors.append(Pool_task_)
        for Process in Processors:
            Process.join()
    else:
        Pool_task_1 = mp.Process(target=tasks)
        Pool_task_1.start()
        Pool_task_1.join()
  