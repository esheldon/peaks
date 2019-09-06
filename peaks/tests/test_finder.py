import numpy as np
import peaks


def test_finder_smoke(seed=None, show=False):
    """
    make sure the thing runs
    """

    rng = np.random.RandomState(seed)

    dims = [50]*2
    nobj = 5

    sim = peaks.sim.Sim(rng=rng, dims=dims, nobj=nobj)

    im, object_data = sim.make_image()

    if show:
        import images
        images.view(im)
