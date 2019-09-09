import numpy as np
import peaks


def test_meas_smoke(seed=12, show=False):
    """
    make sure the thing runs
    """

    rng = np.random.RandomState(seed)

    dims = [100]*2
    nobj = 5

    sim = peaks.sim.GridSim(
        rng=rng,
        dims=dims,
        nobj=nobj,
    )

    image, object_data = sim.make_image()

    kernel_fwhm_pix = sim.psf_fwhm/sim.scale

    finder = peaks.PeakFinder(
        image=image,
        kernel_fwhm=kernel_fwhm_pix,
        noise=sim.noise,
    )
    finder.go()

    moms = peaks.measure.get_moments(
        objects=finder.objects,
        image=image,
        fwhm=2.5,
        scale=sim.scale,
        noise=sim.noise,
    )
    assert moms.size == finder.objects.size


def test_meas(seed=8712):
    """
    make sure the thing runs
    """

    rng = np.random.RandomState(seed)

    dims = [100]*2
    nobj = 1

    sim = peaks.sim.GridSim(
        rng=rng,
        dims=dims,
        nobj=nobj,
        noise=0.001,
        flux_range=(100, 100),
        fwhm_range=(0, 0),
    )

    image, object_data = sim.make_image()

    kernel_fwhm_pix = sim.psf_fwhm/sim.scale

    finder = peaks.PeakFinder(
        image=image,
        kernel_fwhm=kernel_fwhm_pix,
        noise=sim.noise,
    )
    finder.go()

    # large aperture should give fairly close flux
    moms = peaks.measure.get_moments(
        objects=finder.objects,
        image=image,
        fwhm=10.0,
        scale=sim.scale,
        noise=sim.noise,
    )

    w, = np.where(np.abs(moms['flux'] - 100.0) < 1)
    assert w.size == nobj, 'fluxes close for large aper'
    w, = np.where(np.abs(moms['T'] - 0.290136) < 0.01)
    assert w.size == nobj, 'T close for large aper'
