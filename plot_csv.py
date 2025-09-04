import sys, os, csv
import matplotlib.pyplot as plt

# ======================
# CONFIGURACIÓN DE ESTILO
# ======================
CONFIG = {
    "TITLE_FS": 18,     # tamaño de fuente del título
    "LABEL_FS": 22,     # tamaño de fuente de los ejes (xlabel/ylabel)
    "TICK_FS": 14,      # tamaño de los números de los ejes
    "LEGEND_FS": 12,    # tamaño de la leyenda
    "LINE_W": 2.0,      # grosor de línea
    "MARKER_SIZE": 5,   # tamaño de los marcadores
}

def read_csv(path):
    t, temp, sp, pwm, err = [], [], [], [], []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t.append(float(row.get('time_s', '')))  # required
            except Exception:
                continue
            temp_val = row.get('temp_c', '')
            sp_val   = row.get('setpoint_c', '')
            pwm_val  = row.get('pwm', '')
            err_val  = row.get('error_c', '')

            temp.append(float(temp_val) if temp_val not in (None, '',) else float('nan'))
            sp.append(float(sp_val) if sp_val not in (None, '',) else float('nan'))
            try:
                pwm.append(float(pwm_val))
            except Exception:
                pwm.append(float('nan'))
            try:
                err.append(float(err_val))
            except Exception:
                err.append(float('nan'))
    return t, temp, sp, pwm, err

def _has_any_number(arr):
    # True si hay algún valor no-NaN
    return any(x == x for x in arr)

def plot_file(path):
    t, temp, sp, pwm, err = read_csv(path)
    if not t:
        print('No data in CSV')
        return

    fig, ax1 = plt.subplots()

    ax1.plot(
        t, temp, 'b-', marker='o',
        linewidth=CONFIG["LINE_W"], markersize=CONFIG["MARKER_SIZE"],
        label='Temp (°C)'
    )

    if _has_any_number(sp):
        ax1.plot(
            t, sp, 'r--', marker='x',
            linewidth=CONFIG["LINE_W"], markersize=CONFIG["MARKER_SIZE"],
            label='Setpoint (°C)'
        )

    ax1.set_xlabel('Time (s)', fontsize=CONFIG["LABEL_FS"])
    ax1.set_ylabel('Temperature (°C)', fontsize=CONFIG["LABEL_FS"])
    ax1.tick_params(axis='both', which='major', labelsize=CONFIG["TICK_FS"])
    ax1.legend(loc='upper left', fontsize=CONFIG["LEGEND_FS"])

    if _has_any_number(pwm):
        ax2 = ax1.twinx()
        ax2.plot(
            t, pwm, 'g:', marker='s',
            linewidth=CONFIG["LINE_W"], markersize=CONFIG["MARKER_SIZE"],
            label='PWM (0-255)'
        )
        ax2.set_ylabel('PWM (bytes)', fontsize=CONFIG["LABEL_FS"])
        ax2.tick_params(axis='both', which='major', labelsize=CONFIG["TICK_FS"])
        ax2.set_ylim(0, 255)
        ax2.legend(loc='upper right', fontsize=CONFIG["LEGEND_FS"])

    plt.title(os.path.basename(path), fontsize=CONFIG["TITLE_FS"])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_csv.py path/to/log.csv')
        sys.exit(1)
    plot_file(sys.argv[1])
