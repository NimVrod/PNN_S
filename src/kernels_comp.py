import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

x = np.linspace(-4, 4, 500)
sigmas = [0.1, 0.3, 0.5, 1.0]
colors = ["#dc2626", "#ea580c", "#2563eb", "#16a34a"]
labels = [r"$\sigma = 0.1$", r"$\sigma = 0.3$", r"$\sigma = 0.5$", r"$\sigma = 1.0$"]

def epanechnikov(d, s):
    u = d / s
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)

def triangular(d, s):
    u = d / s
    return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0)

kernels = {
    "Gaussowska\n(Gaussian Kernel)":               lambda d, s: np.exp(-d**2 / (2*s**2)),
    "Laplasjańska\n(Laplacian Kernel)":             lambda d, s: np.exp(-np.abs(d) / s),
    "Cauchy'ego\n(Cauchy Kernel)":                  lambda d, s: 1 / (1 + d**2 / s**2),
    "Odwrotna multikwadratowa\n(Inverse Multiquadric Kernel)": lambda d, s: 1 / np.sqrt(d**2 + s**2),
    "Epanecznikowa\n(Epanechnikov Kernel)":          epanechnikov,
    "Trójkątna\n(Triangular Kernel)":               triangular,
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor("#f8fafc")

for ax, (title, fn) in zip(axes.flat, kernels.items()):
    ax.set_facecolor("#ffffff")
    for s, c, lbl in zip(sigmas, colors, labels):
        y = fn(x, s)
        ax.plot(x, y, color=c, linewidth=2.0, label=lbl)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8, color="#1e293b")
    ax.set_xlabel(r"$\|x - x_i\|$", fontsize=9.5, color="#475569")
    ax.set_ylabel(r"$K(x, x_i)$", fontsize=9.5, color="#475569")
    ax.legend(fontsize=8.5, framealpha=0.85, edgecolor="#cbd5e1")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="#cbd5e1")
    ax.tick_params(labelsize=8, colors="#475569")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e8f0")
    ax.set_xlim(-4, 4)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

fig.suptitle("Porównanie funkcji jądra dla różnych wartości " + r"$\sigma$",
             fontsize=13, fontweight="bold", color="#0f172a", y=1.01)

plt.tight_layout()
plt.show()
print("OK")