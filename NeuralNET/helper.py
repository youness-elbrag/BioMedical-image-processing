import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing as mp
import torch
import time


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
    start_time = time.time()
    function
    end_time = time.time()
    return f" Time spent for multiprocessing Tasks training : {end_time - start_time}"


@TimeCounter_Process
def Proceesing_Parallel_Training(*tasks, Parallel_Training_GPU=False):
    Processors = []
    Cpu_Counts = multiprocessing.cpu_count()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.share_memory().to(device)
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
 