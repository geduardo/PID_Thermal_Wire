import numpy as np
import matplotlib.pyplot as plt

# Datos en mL
data_ml = {
    3:  [1517, 1717, 1617, 1717, 1817],
    6:  [1617, 1717, 1617],
    12: [1342, 1417, 1517, 1417, 1517, 1567],
    24: [1617, 1427],
    48: [1517, 1417]
}

# Conversión mL -> MPa (0.312 factor)
ml_to_mpa = 0.312

# Calcular medias y desviaciones
times = []
means = []
stds = []

for t, values in data_ml.items():
    values_mpa = np.array(values) * ml_to_mpa
    times.append(t)
    means.append(values_mpa.mean())
    stds.append(values_mpa.std(ddof=1))  # desviación estándar muestral

# Graficar
plt.errorbar(times, means, yerr=stds, fmt='o-', capsize=5, label='σu promedio')
plt.xlabel('Tiempo a 300°C (s)')
plt.ylabel('Tensión de rotura σu (MPa)')
plt.title('σu vs Tiempo a 300°C')
plt.grid(True)
plt.legend()
plt.show()
