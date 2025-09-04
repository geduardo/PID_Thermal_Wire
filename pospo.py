import numpy as np
import matplotlib.pyplot as plt

# Datos en mL por tiempo (s)
data_ml = {
    0: [900/0.312],
    3:  [1517, 1717, 1617, 1717, 1817],
    4:  [1617, 1517, 1617, 1717],
    6:  [1617, 1617, 1580],
    8:  [1417, 1517, 1550],
    12: [1380, 1417, 1490, 1417, 1490, 1567],
    24: [1617, 1427],
    48: [1517, 1417, 1342]
}

# Conversión mL → MPa
FACTOR = 0.312

# Procesar datos
times  = sorted(data_ml.keys())
means  = [np.mean(np.array(data_ml[t]) * FACTOR) for t in times]
stds   = [np.std (np.array(data_ml[t]) * FACTOR, ddof=1) for t in times]

# Gráfica
plt.errorbar(times, means, yerr=stds, fmt='o-', capsize=5, label='σu')
plt.xlabel('300ºC preheating Time (s)',fontsize=16)
plt.ylabel('Ultimate strength σu (MPa)',fontsize=16)
plt.grid(True)
plt.legend()
plt.show()
