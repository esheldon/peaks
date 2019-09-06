import numpy as np
import peaks


def test_finder_smoke(seed=8712, show=False):
    """
    make sure the thing runs
    """

    rng = np.random.RandomState(seed)

    dims = [50]*2
    nobj = 5

    sim = peaks.sim.Sim(rng=rng, dims=dims, nobj=nobj)

    image, object_data = sim.make_image()

    kernel_fwhm_pix = sim.psf_fwhm/sim.scale
    kernel = peaks.gauss_kernel(
        fwhm=kernel_fwhm_pix,
        dims=[25]*2,
    )

    finder = peaks.PeakFinder(
        image=image,
        kernel=kernel,
        noise=sim.noise,
    )
    finder.go()

    if show:
        peaks.vis.view_peaks(
            image,
            sim.noise,
            finder.objects,
        )


def test_finder(seed=31415, show=False):
    """
    make sure we recover the right number of objects
    in a grid test
    """

    rng = np.random.RandomState(seed)

    dims = [50]*2
    nobj = 9

    sim = peaks.sim.GridSim(
        rng=rng,
        dims=dims,
        nobj=nobj,
        fwhm_range=(0, 0),
        noise=0.0001,
    )

    image, object_data = sim.make_image()

    kernel_fwhm_pix = sim.psf_fwhm/sim.scale
    kernel = peaks.gauss_kernel(
        fwhm=kernel_fwhm_pix,
        dims=[25]*2,
    )

    finder = peaks.PeakFinder(
        image=image,
        kernel=kernel,
        noise=sim.noise,
    )
    finder.go()


    if show:
        peaks.vis.view_peaks(
            image=image,
            noise=sim.noise,
            objects=finder.objects,
            show=True,
        )

    nfound = finder.objects.size
    assert nfound == nobj, \
        'expected %d objects, got %d' % (nobj, nfound)
