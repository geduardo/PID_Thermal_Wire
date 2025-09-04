import numpy as np
import matplotlib.pyplot as plt

# Datos en mL por temperatura (°C)
data_ml = {
    0: [1500,1450,1517,1480],
    50:  [1417, 1567, 1417,1500,1350],
    100: [1417, 1417, 1317, 1477],
    150: [1417, 1387, 1387],
    175:[1357,1307,1280],
    200: [1217, 1167, 1217],
    225: [1117, 1197, 1097],
    250: [917, 967, 917],
}

FACTOR = 0.312  # mL → MPa

temps = sorted(data_ml.keys())
means = [np.mean(np.array(data_ml[t]) * FACTOR) for t in temps]
stds  = [np.std (np.array(data_ml[t]) * FACTOR, ddof=1) for t in temps]

plt.errorbar(temps, means, yerr=stds, fmt='o-', capsize=5, label='σu')
plt.xlabel('Temperature (°C)',fontsize=16)
plt.ylabel('Ultimate strength σu (MPa)',fontsize=16)
plt.grid(True)
plt.legend()
plt.show()
