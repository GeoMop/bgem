"""
Various DFN plotting and visualisation functions.
"""


def plotly_fractures(fr_set, fr_points):
    """
    Plot generated fractures.
    :param fr_set: List[FractureShape]
    :param fr_set: List[np.array(n, 2)] local point coordinates on fractures
    :return:
    """
    import plotly.offline as pl
    import plotly.graph_objs as go
    # import plotly.graph_objects as go
    for ifr, (fr, points) in enumerate(zip(fr_set, fr_points)):
        n_side = 5
        boundary = np.empty((4, n_side, 3))
        corners = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])
        for s in range(4):
            start, end = corners[s, :], corners[(s + 1) % 4, :]
            boundary[s, :, :] = start[None, :] + (end - start)[None, :] * np.linspace(0, 1, n_side, endpoint=False)[:,
                                                                          None]
        boundary = boundary.reshape((-1, 3))
        boundary = fr.transform(boundary)
        points = fr.transform(points)

        fig = go.Figure(data=[
            go.Scatter3d(x=boundary[:, 0], y=boundary[:, 1], z=boundary[:, 2],
                         marker=dict(size=1, color='blue')),
            go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                         marker=dict(size=1.5, color='red'))
        ])
        fig.update_layout(
            scene=dict(
                # xaxis=dict(range=[-2, 2]),
                # yaxis=dict(range=[-2, 2]),
                # zaxis=dict(range=[-1, 1]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)

            ),
        )
        pl.plot(fig, filename='fractures.html')


def plot_fr_orientation(fractures):
    family_dict = collections.defaultdict(list)
    for fr in fractures:
        x, y, z = \
            fracture.FisherOrientation.rotate(np.array([0, 0, 1]), axis=fr.rotation_axis, angle=fr.rotation_angle)[0]
        family_dict[fr.region].append([
            to_polar(z, y, x),
            to_polar(z, x, -y),
            to_polar(y, x, z)
        ])

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
    for name, data in family_dict.items():
        # data shape = (N, 3, 2)
        data = np.array(data)
        for i, ax in enumerate(axes):
            phi = data[:, i, 0]
            r = data[:, i, 1]
            c = ax.scatter(phi, r, cmap='hsv', alpha=0.75, label=name)
    axes[0].set_title("X-view, Z-north")
    axes[1].set_title("Y-view, Z-north")
    axes[2].set_title("Z-view, Y-north")
    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
    fig.legend(loc=1)
    fig.savefig("fracture_orientation.pdf")
    plt.close(fig)
    # plt.show()
