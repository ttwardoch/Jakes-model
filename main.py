import random
import matplotlib.pyplot as plt
import numpy as np
import cmath
from tqdm import tqdm

FREQUENCY = 30000000000  # 30 GHz
VELOCITY = 30  # m/s
c = 300000000
n_bins = 60
iterations = 100 # number of simulators
Jakes_iterations = 10 # number of angles

MAX_DOPPLER = VELOCITY/c*FREQUENCY
MIN_DOPPLER = -VELOCITY/c*FREQUENCY
DELTA = (MAX_DOPPLER - MIN_DOPPLER)/n_bins

bins = np.zeros(n_bins, dtype='complex_')
bins_labels = np.linspace(MIN_DOPPLER, MAX_DOPPLER, num=n_bins+1) + DELTA/2
bins_labels = bins_labels[:-1].copy()

def doppler_shift(angle, velocity, frequency):
    return velocity/c*frequency*np.cos(angle)

shifts = []
for _ in tqdm(range(iterations)):
    bins_Jakes = np.zeros(n_bins, dtype='complex_')
    for __ in range(Jakes_iterations):
        angle = random.random()*2*np.pi
        doppler = doppler_shift(angle, VELOCITY, FREQUENCY)
        shifts.append(doppler)
        bin_target = int((doppler-MIN_DOPPLER) // DELTA)
        signal = cmath.exp(1j * (random.random()*2*np.pi))
        bins_Jakes[bin_target] += signal
    bins_Jakes = abs(bins_Jakes)**2
    bins += bins_Jakes

bins = bins / np.sum(bins) / MAX_DOPPLER * n_bins / 2

theory_x = np.linspace(MIN_DOPPLER, MAX_DOPPLER, n_bins*10+1) + DELTA / 20
theory_x = theory_x[2:-1-2].copy()
C = 1/(np.pi * VELOCITY *FREQUENCY/c) #min(bins)
theory_y = [C/np.sqrt(1-(x/MAX_DOPPLER)**2) for x in theory_x]

# plot
#plt.bar(bins_labels, bins, width=DELTA, label="Estimate of expected value for Jakes' simulator")
plt.plot(theory_x, theory_y, c="r", label="Power spectrum from Clarke's model")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Expected power spectrum of Jakes' simulator")
plt.title("Clarke's model power spectrum")
#plt.legend()
plt.tight_layout()
plt.show()
