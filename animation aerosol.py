import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import rasterio

# ============================
# LOAD DEM
# ============================
dem_file = "C:/Users/COM/Desktop/week/elevation_data10km.tif"
with rasterio.open(dem_file) as src:
    DEM_data = src.read(1)
    transform = src.transform

nrows, ncols = DEM_data.shape
dem_extent = [0, ncols, 0, nrows]

# ============================
# HELPER FUNCTIONS
# ============================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def norm(x, xmin, xmax):
    return np.clip((x - xmin) / (xmax - xmin), 0, 1)

# ============================
# INITIAL PARTICLES
# ============================
np.random.seed(0)
n_particles = 1500
x = np.random.rand(n_particles)
y = np.random.rand(n_particles)
z0 = np.random.rand(n_particles) * 0.08
PM_base = 80.0

# ============================
# FIGURE SETUP
# ============================
fig = plt.figure(figsize=(16, 6))

# 3D scatter
ax1 = fig.add_subplot(131, projection="3d")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_zlim(0, 1)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Vertical Mixing")
scatter = ax1.scatter(x, y, z0, s=6, c="gray", alpha=0.8)
info = ax1.text2D(0.02, 0.96, "", transform=ax1.transAxes, fontsize=10,
                  bbox=dict(facecolor="white", alpha=0.85))

# DEM map
ax2 = fig.add_subplot(132)
im_dem = ax2.imshow(DEM_data, extent=dem_extent, origin='upper', cmap='terrain')
ax2.set_title("DEM Map")
plt.colorbar(im_dem, ax=ax2, label="Elevation (m)")

# PM2.5 map
ax3 = fig.add_subplot(133)
pm_map = np.zeros_like(DEM_data)
im_pm = ax3.imshow(pm_map, extent=dem_extent, origin='upper', cmap='Reds', vmin=0, vmax=150)
ax3.set_title("PM2.5 Map")
plt.colorbar(im_pm, ax=ax3, label="PM2.5 (µg/m³)")

# ============================
# SLIDERS
# ============================
plt.subplots_adjust(left=0.05, bottom=0.45)

def add_slider(y, label, vmin, vmax, init):
    ax_slider = plt.axes([0.2, y, 0.6, 0.025])
    return Slider(ax_slider, label, vmin, vmax, valinit=init)

s_T      = add_slider(0.40, "T (°C)",           20, 45, 32)
s_BLH    = add_slider(0.37, "BLH (m)",          200, 2000, 800)
s_RH     = add_slider(0.34, "RH (%)",           30, 100, 70)
s_precip = add_slider(0.31, "Precip (mm)",       0, 20, 0)
s_wind   = add_slider(0.28, "Wind (m/s)",        0, 12, 2)

# ============================
# UPDATE FUNCTION
# ============================
def update(frame):
    T      = s_T.val
    BLH    = s_BLH.val
    RH     = s_RH.val
    precip = s_precip.val
    wind   = s_wind.val

    # ------------------------
    # Vertical mixing factor
    # ------------------------
    mixing = (
        0.4 * sigmoid(T - 30) +
        0.3 * norm(BLH, 200, 2000) +
        0.1 * norm(wind, 0, 12)
    )
    mixing *= (1 - 0.4 * norm(RH, 40, 100))
    mixing = np.clip(mixing, 0, 1)

    # ------------------------
    # PM removal factor
    # ------------------------
    decay_rate = (
        0.9 * norm(wind, 0, 12) +
        0.6 * norm(precip, 0, 20)
    )
    decay_rate *= sigmoid(T - 30)
    keep_ratio = np.exp(-decay_rate)
    keep_ratio = np.clip(keep_ratio, 0.05, 1.0)
    n_keep = int(n_particles * keep_ratio)

    # ------------------------
    # UPDATE PARTICLES (3D)
    # ------------------------
    z = z0 + mixing * np.random.rand(n_particles) + DEM_data[nrows//2, ncols//2]/1000.0
    z = np.clip(z, 0, 1)
    scatter._offsets3d = (x[:n_keep], y[:n_keep], z[:n_keep])
    scatter.set_alpha(0.9 * keep_ratio)
    scatter.set_sizes(np.full(n_keep, 8 * keep_ratio))
    info.set_text(
        f"PM effect keep={keep_ratio:.2f}\n"
        f"T={T:.1f}°C | Wind={wind:.1f} m/s | Precip={precip:.1f} mm"
    )

    # ------------------------
    # UPDATE PM2.5 MAP
    # ------------------------
    # simple model: PM2.5 = base * (1 - mixing) * keep_ratio + DEM effect
    pm_map = PM_base * (1 - mixing) * keep_ratio + DEM_data/100


    pm_map = np.clip(pm_map, 0, 150)
    im_pm.set_data(pm_map)

    return scatter, im_pm

# ============================
# ANIMATION
# ============================
ani = FuncAnimation(fig, update, interval=120)
plt.show()

