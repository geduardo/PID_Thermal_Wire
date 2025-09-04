import pandas as pd
import matplotlib.pyplot as plt

csv_path = "elongaciones.csv"
df = pd.read_csv(csv_path)

# Distancia en píxeles
plt.figure()
plt.plot(df["t_sec"], df["d_px"])
plt.xlabel("Tiempo [s]")
plt.ylabel("Distancia [px]")
plt.title("Distancia vs tiempo")
plt.grid(True)
plt.show()

# (Opcional) Distancia en unidades si existe columna d_mm (o similar)
unit_col = next((c for c in df.columns if c.startswith("d_") and c not in ("d_px",)), None)
if unit_col:
    plt.figure()
    plt.plot(df["t_sec"], df[unit_col])
    plt.xlabel("Tiempo [s]")
    plt.ylabel(f"Distancia [{unit_col.split('_',1)[1]}]")
    plt.title("Distancia (unidades) vs tiempo")
    plt.grid(True)
    plt.show()

# Deformación epsilon
plt.figure()
plt.plot(df["t_sec"], df["epsilon"])
plt.xlabel("Tiempo [s]")
plt.ylabel("ε")
plt.title("Deformación vs tiempo")
plt.grid(True)
plt.show()

# (Opcional) Distancia y ε en ejes gemelos
plt.figure()
ax = plt.gca()
ax.plot(df["t_sec"], df["d_px"], label="d_px")
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Distancia [px]")
ax2 = ax.twinx()
ax2.plot(df["t_sec"], df["epsilon"], linestyle="--", label="epsilon")
ax2.set_ylabel("ε")
plt.title("Distancia y deformación")
ax.grid(True)
plt.show()
