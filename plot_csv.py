import sys, os, csv
import matplotlib.pyplot as plt


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
            sp_val = row.get('setpoint_c', '')
            pwm_val = row.get('pwm', '')
            err_val = row.get('error_c', '')

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


def plot_file(path):
    t, temp, sp, pwm, err = read_csv(path)
    if not t:
        print('No data in CSV')
        return

    fig, ax1 = plt.subplots()
    ax1.plot(t, temp, 'b-', marker='o', label='Temp (°C)')
    if any(x == x for x in sp):  # any non-NaN
        ax1.plot(t, sp, 'r--', marker='x', label='Setpoint (°C)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('°C')
    ax1.legend(loc='upper left')

    if any(x == x for x in pwm):
        ax2 = ax1.twinx()
        ax2.plot(t, pwm, 'g:', marker='s', label='PWM (0-255)')
        ax2.set_ylabel('PWM')
        ax2.set_ylim(0, 255)
        ax2.legend(loc='upper right')

    plt.title(os.path.basename(path))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_csv.py path/to/log.csv')
        sys.exit(1)
    plot_file(sys.argv[1])
