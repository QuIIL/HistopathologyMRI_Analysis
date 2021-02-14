import numpy as np
import pylab as plt


def log_schedule(t, T, K=100):
    """
    Training Signal Annealing log-schedule
    :param t: training step
    :param T: total training step
    :param K: number of categories (for classification problem, preset to 100 for other tasks)
    :return:  training signal releasing threshold
    """
    return (1 - np.exp(-t/T * 5)) * (1 - 1/K) + 1/K


def linear_schedule(t, T, K=100):
    """
    Training Signal Annealing log-schedule
    :param t: training step
    :param T: total training step
    :param K: number of categories (for classification problem, preset to 100 for other tasks)
    :return:  training signal releasing threshold
    """
    return t/T * (1 - 1/K) + 1/K


def exp_schedule(t, T, K=100):
    """
    Training Signal Annealing log-schedule
    :param t: training step
    :param T: total training step
    :param K: number of categories (for classification problem, preset to 100 for other tasks)
    :return:  training signal releasing threshold
    """
    return np.exp((t/T - 1)*5) * (1 - 1/K) + 1/K


if __name__ == "__main__":
    _T = 1000
    _K = 100
    thr_log = np.array([log_schedule(t, _T, _K) for t in range(_T)])
    thr_lin = np.array([linear_schedule(t, _T, _K) for t in range(_T)])
    thr_exp = np.array([exp_schedule(t, _T, _K) for t in range(_T)])
    plt.plot(thr_lin, '-', color='dodgerblue')
    plt.plot(thr_log, '--', color='darkgrey')
    plt.plot(thr_exp, 'k-.')
    plt.legend(['linear-schedule', 'log-schedule', 'exp-schedule'])
    plt.ylabel('Training signal threshold')
    plt.xlabel('Training progress')
    plt.xticks(np.linspace(0, 1000, 6), labels=np.round(np.linspace(0, 1, 6), 1))
    plt.show()
