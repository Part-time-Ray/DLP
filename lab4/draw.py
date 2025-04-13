class kl_annealing():
    def __init__(self, type, epoch, cycle, ratio):
        self.iteration = 0
        self.type = type
        if self.type == "Cyclical":
            self.frange_cycle_linear(epoch, start=0.0, stop=1.0, n_cycle=cycle, ratio=ratio)
        elif self.type == "Monotonic":
            self.frange_cycle_linear(epoch, start=0.0, stop=1.0, n_cycle=epoch, ratio=ratio)
        else:
            self.beta = [1.0] * epoch
        
    def update(self):
        self.iteration += 1
    
    def get_beta(self):
        if self.self.type == "None": return 1.0
        return round(self.beta[self.iteration], 2)

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        self.beta = []
        cycle_num = n_iter // n_cycle
        for i in range(cycle_num):
            for j in range(n_cycle+1):
                val = start + (stop - start) * ratio * (j)
                self.beta.append(min(stop, val))
        self.beta = self.beta[:n_iter]

def teacher_forcing_ratio(epoch=100, start=1, end=0, step=0.1, cooldown=10):
    return [start] * cooldown + [max(0, start - step * i) for i in range(epoch - cooldown)]

import matplotlib.pyplot as plt
import numpy as np



def plot_kl_annealing(cyclical, monotonic, none):
    plt.figure(figsize=(10, 5))
    plt.plot(cyclical.beta, label='Cyclical')
    plt.plot(monotonic.beta, label='Monotonic')
    plt.plot(none.beta, label='None')
    plt.xlabel('Iteration')
    plt.ylabel('Beta Value')
    plt.title('KL Annealing')
    plt.legend()
    plt.grid()
    plt.savefig('kl_annealing.png')


def plot_teacher_forcing_ratio():
    ratio = teacher_forcing_ratio()
    
    plt.figure(figsize=(10, 5))
    plt.plot(ratio)
    plt.xlabel('Iteration')
    plt.ylabel('Teacher Forcing Ratio')
    plt.title('Teacher Forcing Ratio')
    plt.legend(['Teacher Forcing Ratio'])
    plt.grid()
    plt.savefig('teacher_forcing_ratio.png')

cyclical = kl_annealing("Cyclical", 100, 10, 0.1)
monotonic = kl_annealing("Monotonic", 100, 10, 0.025)
none = kl_annealing("None", 100, 10, 0.1)
plot_kl_annealing(cyclical, monotonic, none)
plot_teacher_forcing_ratio()
