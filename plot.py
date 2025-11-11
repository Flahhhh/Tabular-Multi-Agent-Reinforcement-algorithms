import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import json
import matplotlib
import numpy as np

plt.style.use('fivethirtyeight')
params = {
    'figure.figsize': (15, 8),
    'font.size': 24,
    'legend.fontsize': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
pylab.rcParams.update(params)
WINDOW = 50

def plot_data(logs):
    fig, axs = plt.subplots(2, 2, figsize=(20,30), sharey=False, sharex=True)

    axs[0,0].plot(logs[0], 'blue', linewidth=1, alpha=0.2)
    axs[0,1].plot(logs[1], 'blue', linewidth=1, alpha=0.2)
    axs[1,0].plot(logs[2], 'blue', linewidth=1, alpha=0.2)
    axs[1,1].plot(logs[3], 'blue', linewidth=1, alpha=0.2)

    axs[0,0].set_title('Reward(Agent 1)')
    axs[0,1].set_title('Reward(Agent 2)')
    axs[1,0].set_title('Loss(Agent 1)')
    axs[1,1].set_title('Loss(Agent 2)')

    axs[0,0].set_xlabel(f'Episodes')
    axs[0,0].set_ylabel(f'Reward')

    axs[0,1].set_xlabel(f'Episodes')
    axs[0,1].set_ylabel(f'Reward')

    axs[1,0].set_xlabel(f'Episodes')
    axs[1,0].set_ylabel(f'Loss')

    axs[1,1].set_xlabel(f'Episodes')
    axs[1,1].set_ylabel(f'Loss')

    plt.savefig('plots/PHC-Matrix.png', format='png')

def sliding_mean_data(d):
    return np.convolve(d, np.ones(WINDOW) / WINDOW, 'valid')
def open_logs_json(path):
    with open(path) as json_data:
        d = json.load(json_data)
        rewards = d["rewards"]
        losses = d["loss"]

        json_data.close()

        rewards = list(zip(*rewards))
        losses = list(zip(*losses))

    return sliding_mean_data(rewards[0]), sliding_mean_data(rewards[1]), \
            sliding_mean_data(losses[0]), sliding_mean_data(losses[1])

if __name__ == "__main__":
    path = r"logs/PHC_2025-11-11 06:59/logs.json"
    logs = open_logs_json(path)

    plot_data(logs)
