def view_peaks(*, image, noise, objects,
               show=False,
               color='red',
               type='filled circle',
               width=800,
               plt=None):
    """
    view the image with peak positions overplotted
    """
    import biggles
    import images

    tim = image.copy()

    if plt is None:
        aim = images.asinh_scale(image, noise=noise)
        plt = images.view(aim, show=False)

    # the viewer transposes the image
    points = biggles.Points(
        objects['col'],
        objects['row'],
        color=color,
        type=type,
    )
    plt.add(points)

    if show:
        arat = image.shape[0]/image.shape[1]
        plt.show(width=width, height=width*arat)

    return plt
