import numpy as np
import matplotlib.pyplot as plt

# Set up plot
plt.ion()  # Turn on interactive mode for dynamic updates
fig, ax = plt.subplots()
(line,) = ax.plot([], [], "b.", markersize=0.7)
ax.axis([-35, 35, -35, 35])
ax.axis("off")
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()

# Set up parameters
range_bound = 50
Ptrans = 0.1
t = np.arange(1, 50000)  # time points
# k0 = -0.9
# k1 = 0.8
k0 = 0.9
k1 = 0.8
e0 = np.array([-10, 10])
e1 = np.array([10, -10])

# Pre-compute the rotation matrices for all possible angles to avoid recalculating
rotation_matrices = {
    (th1, th2): (
        k0
        * np.array(
            [
                [np.cos(np.deg2rad(th1)), np.sin(np.deg2rad(th1))],
                [-np.sin(np.deg2rad(th1)), np.cos(np.deg2rad(th1))],
            ]
        ),
        k1
        * np.array(
            [
                [np.cos(np.deg2rad(th2)), np.sin(np.deg2rad(th2))],
                [-np.sin(np.deg2rad(th2)), np.cos(np.deg2rad(th2))],
            ]
        ),
    )
    for th1 in range(-range_bound, range_bound)
    for th2 in range(-range_bound, range_bound)
}

# Simulate
for th2 in range(-range_bound, range_bound):
    for th1 in range(-range_bound, range_bound):
        # Generate state transitions for the whole time series at once
        random_values = np.random.rand(t.size)
        s = np.where(random_values < Ptrans, 1, 0)
        s = (
            np.cumsum(s) % 2
        )  # Cumulative sum mod 2 gives the correct switching behavior

        # Get precomputed rotation matrices
        A0, A1 = rotation_matrices[(th1, th2)]

        # Propagate
        x = np.zeros((2, t.size))
        x[:, 0] = e1
        for tp in range(1, t.size):
            # Dynamics
            if s[tp] == 1:
                x[:, tp] = A1 @ (x[:, tp - 1] - e1) + e1
            else:
                x[:, tp] = A0 @ (x[:, tp - 1] - e0) + e0

        # Update plot data
        line.set_data(x[0, :], x[1, :])
        fig.canvas.draw()
        fig.canvas.flush_events()

plt.ioff()  # Turn off interactive mode
plt.show()
