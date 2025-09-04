import pandas as pd
import matplotlib.pyplot as plt

# Leer CSV
df = pd.read_csv("elongaciones.csv")

# Gráfico de L_px
plt.figure(figsize=(8,4))
plt.plot(df["idx"], df["L_px"], marker="o")
plt.xlabel("Medida (idx)")
plt.ylabel("Longitud [px]")
plt.title("Evolución de longitud")
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico de epsilon
plt.figure(figsize=(8,4))
plt.plot(df["idx"], df["epsilon"], marker="o", color="red")
plt.xlabel("Medida (idx)")
plt.ylabel("Elongación ε")
plt.title("Evolución de elongación")
plt.axhline(0, color="black", linestyle="--")
plt.grid(True)
plt.tight_layout()
plt.show()
