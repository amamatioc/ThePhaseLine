import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def phase_line_diagram(A, y1, y2, y3):
    y_vals = np.linspace(y1 - 5, y3 + 5, 1000)

    dy_dt = -A * (y_vals - y1) * (y_vals - y2) * (y_vals - y3) ** 2

    plt.figure(figsize=(10, 6))
    plt.plot(y_vals, dy_dt, label=r"$y' = -A(y - y_1)(y - y_2)(y - y_3)^2$", color="blue")
    plt.axhline(0, color="gray", linestyle="--")

    equilibria = [y1, y2, y3]
    plt.scatter(equilibria, [0, 0, 0], color="red", zorder=5)

    # Label stability type
    for y_eq in equilibria:
        if y_eq == y1:
            stability = "Source"
            color = "green"
            position = -0.5
        elif y_eq == y2:
            stability = "Sink"
            color = "purple"
            position = 0.5
        elif y_eq == y3:
            stability = "Node"
            color = "orange"
            position = -0.5

        plt.text(y_eq, position, f"{stability}", color=color, ha="center", va="bottom")

    plt.arrow(y1 - 1, -0.3, 0.8, 0, color="green", head_width=0.2, head_length=0.2)
    plt.arrow(y1 + 1, -0.3, -0.8, 0, color="green", head_width=0.2, head_length=0.2)

    plt.arrow(y2 - 1, 0.3, -0.8, 0, color="purple", head_width=0.2, head_length=0.2)
    plt.arrow(y2 + 1, 0.3, 0.8, 0, color="purple", head_width=0.2, head_length=0.2)

    plt.arrow(y3 - 1, -0.3, 0.8, 0, color="orange", head_width=0.2, head_length=0.2)
    plt.arrow(y3 + 1, -0.3, -0.8, 0, color="orange", head_width=0.2, head_length=0.2)

    plt.xlabel("y", fontsize=12)
    plt.ylabel("$y'$", fontsize=12)
    plt.title("Phase Line Diagram for $y' = -A(y - y_1)(y - y_2)(y - y_3)^2$", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


def phase_line_d():
    # Define y' = (y - 1)(y - 2)(y - 3)
    y_vals = np.linspace(0, 4, 100)
    dy_dt = (y_vals - 1) * (y_vals - 2) * (y_vals - 3)

    plt.figure(figsize=(10, 4))
    plt.plot(y_vals, dy_dt, label="$y' = (y - 1)(y - 2)(y - 3)$")
    plt.axhline(0, color="gray", linestyle="--")
    plt.scatter([1, 2, 3], [0, 0, 0], color="red")
    plt.text(1, 0.1, "Source", color="red")
    plt.text(2, -0.3, "Sink", color="red")
    plt.text(3, 0.1, "Source", color="red")
    plt.xlabel("y")
    plt.ylabel("$y'$")
    plt.title("Phase Line for $y' = (y - 1)(y - 2)(y - 3)$")
    plt.legend()
    plt.savefig("phase_line_d.png")
    plt.show()

def phase_line_e():
    # Define y' = - (y - 1)^(5/3) (y - 2)^2 (y - 3)
    y_vals = np.linspace(-1, 4, 100)  # Extend the range to include y < 0
    dy_dt = np.zeros_like(y_vals)

    for i, y in enumerate(y_vals):
        if y > 1:
            dy_dt[i] = - (y - 1) ** (5 / 3) * (y - 2) ** 2 * (y - 3)
        elif y > 2:
            dy_dt[i] = - (y - 1) ** (5 / 3) * (y - 2) ** 2 * (y - 3)
        elif y > 3:
            dy_dt[i] = - (y - 1) ** (5 / 3) * (y - 2) ** 2 * (y - 3)
    plt.figure(figsize=(10, 4))
    plt.plot(y_vals, dy_dt, label=r"$y' = - (y - 1)^{5/3} (y - 2)^2 (y - 3)$")
    plt.axhline(0, color="gray", linestyle="--")
    plt.scatter([1, 2, 3], [0, 0, 0], color="red")
    plt.text(1, 0.1, "Node", color="red")
    plt.text(2, -0.3, "Node", color="red")
    plt.text(3, 0.1, "Sink", color="red")
    plt.xlabel("y")
    plt.ylabel("$y'$")
    plt.title("Phase Line for $y' = - (y - 1)^{5/3} (y - 2)^2 (y - 3)$")
    plt.legend()
    plt.savefig("phase_line_e.png")
    plt.show()

def phase_line_f():
    # Define y' = y * sin(y)
    y_vals = np.linspace(-10, 10, 400)
    dy_dt = y_vals * np.sin(y_vals)

    plt.figure(figsize=(10, 4))
    plt.plot(y_vals, dy_dt, label="$y' = y \\sin(y)$")
    plt.axhline(0, color="gray", linestyle="--")

    # Finding equilibria visually at multiples of pi
    equilibria_y = np.arange(-3 * np.pi, 3 * np.pi + 1, np.pi)
    plt.scatter(equilibria_y, np.zeros_like(equilibria_y), color="red")

    # Label the nature of equilibria based on stability
    for y_eq in equilibria_y:
        # Check the behavior around each equilibrium point
        left_val = (y_eq - 0.1) * np.sin(y_eq - 0.1)
        right_val = (y_eq + 0.1) * np.sin(y_eq + 0.1)

        if left_val < 0 and right_val > 0:
            plt.text(y_eq, 0.5, "Source", color="red", ha="center")
        elif left_val > 0 and right_val < 0:
            plt.text(y_eq, -0.5, "Sink", color="blue", ha="center")
        else:
            plt.text(y_eq, 0.5, "Node", color="green", ha="center")

    plt.xlabel("y")
    plt.ylabel("$y'$")
    plt.title("Phase Line for $y' = y \\sin(y)$")
    plt.legend()
    plt.grid(True)
    plt.show()

def classify_equilibrium(dy_dt, index):
    slope_before = dy_dt[index - 1] if index > 0 else dy_dt[index]
    slope_after = dy_dt[index + 1] if index < len(dy_dt) - 1 else dy_dt[index]

    if slope_before < 0 and slope_after > 0:
        return "Source", "purple"
    elif slope_before > 0 and slope_after < 0:
        return "Sink", "green"
    else:
        return "Node", "orange"

def phase_line_g_perturbation(perturbation):
    # Define y' = y * sin(y) + perturbation
    y_vals = np.linspace(-10, 10, 400)
    dy_dt = y_vals * np.sin(y_vals) + perturbation

    plt.figure(figsize=(10, 4))
    plt.plot(y_vals, dy_dt, label=f"$y' = y \\sin(y) {'+' if perturbation > 0 else '-'} {abs(perturbation)}$")
    plt.axhline(0, color="gray", linestyle="--")

    # Identify and classify equilibrium points
    roots = y_vals[np.isclose(dy_dt, 0, atol=0.05)]
    for root in roots:
        index = np.searchsorted(y_vals, root)
        stability, color = classify_equilibrium(dy_dt, index)
        plt.scatter(root, 0, color="red")
        plt.text(root, 0.5 if perturbation > 0 else -0.5, stability, color=color, ha="center")

    plt.xlabel("y")
    plt.ylabel("$y'$")
    plt.title(f"Phase Line for $y' = y \\sin(y) {'+' if perturbation > 0 else '-'} {abs(perturbation)}$")
    plt.legend()
    plt.grid(True)
    plt.show()


def phase_line_g():
    # Define y' = y * sin(y) - 0.1
    y_vals = np.linspace(-10, 10, 1000)
    dy_dt = y_vals * np.sin(y_vals) - 0.1

    plt.figure(figsize=(12, 6))
    plt.plot(y_vals, dy_dt, label=r"$y' = y \sin(y) - 0.1$", color="blue")
    plt.axhline(0, color="gray", linestyle="--")

    # Identify and classify equilibrium points
    roots = y_vals[np.isclose(dy_dt, 0, atol=0.05)]
    displayed_middle_labels = set()
    for root in roots:
        index = np.searchsorted(y_vals, root)
        stability, color = classify_equilibrium(dy_dt, index)

        # Only display "Sink" and "Source" labels
        if stability in {"Source", "Sink"}:
            # Scatter plot the root
            plt.scatter(root, 0, color="red")

            # Label equilibrium points without overlapping in the middle
            if -1 < root < 1:  # Middle region
                if stability not in displayed_middle_labels:
                    plt.text(root, 1.0 if stability == "Source" else -1.0, stability,
                             color=color, fontsize=10, ha="center")
                    displayed_middle_labels.add(stability)
            else:  # Outside the middle region
                y_text_offset = 1.0 if stability == "Source" else -1.0
                plt.text(root, y_text_offset, stability, color=color, fontsize=10, ha="center")

    plt.xlabel("y", fontsize=12)
    plt.ylabel("$y'$", fontsize=12)
    plt.title("Phase Line for $y' = y \sin(y) - 0.1$", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()
def phase_line_h():
    K = 1500  # carrying capacity
    C = 3200  # scaling factor for population growth

    s_values = [0, 50, 100, 125, 150, 175, 200]
    def dydt(y, s):
        return -(y * (y - K)) / C - s
    def find_equilibria(s):
        eq_points = fsolve(dydt, [0, K], args=(s))
        return eq_points

    y_vals = np.linspace(0, 2000, 400)  # Population values for the phase line

    plt.figure(figsize=(12, 8))
    for s in s_values:
        dydt_vals = [dydt(y, s) for y in y_vals]

        equilibria = find_equilibria(s)

        plt.plot(y_vals, dydt_vals, label=f's = {s}')

        for eq in equilibria:
            if 0 <= eq <= 2000:
                plt.plot(eq, 0, 'ro' if s == 175 else 'bo')  # Marked with different colors for bifurcation point

    plt.axhline(0, color='black', lw=0.5)
    plt.xlabel('Population (y)')
    plt.ylabel("dy/dt (Population Change Rate)")
    plt.title("Phase Lines for Different Harvesting Rates (s)")
    plt.legend()
    plt.grid(True)
    plt.show()


A = 1
y1 = -2
y2 = 1
y3 = 3

phase_line_diagram(A, y1, y2, y3)
phase_line_d()
phase_line_e()
phase_line_f()
phase_line_g()
phase_line_g_perturbation(0.1)
phase_line_h()