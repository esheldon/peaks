import numpy as np
from .moments import fwhm_to_T


class Sim(dict):
    def __init__(self,
                 *,
                 rng,
                 dims,
                 nobj,
                 psf_fwhm=0.9,
                 scale=0.263,
                 border=0,
                 noise=1,
                 shape_noise=0.2,
                 fwhm_range=(0.0, 2.5),
                 flux_range=(10.0, 100.0)):

        self.rng = rng
        self.dims = dims
        self.scale = float(scale)
        self.border = int(border)
        self.nobj = int(nobj)
        self.psf_fwhm = float(psf_fwhm)
        self.shape_noise = float(shape_noise)
        self.fwhm_range = fwhm_range
        self.flux_range = flux_range
        self.noise = noise

        assert len(dims) == 2
        assert len(fwhm_range) == 2
        assert len(flux_range) == 2

        self._set_psf()
        self._set_jacobian()

    def make_image(self):
        """
        make an image
        """

        image = np.zeros(self.dims)

        objects = self.get_object_struct(n=self.nobj)

        for i in range(self.nobj):
            this_image, gm = self.get_object_image()

            image += this_image

            objects[i] = self._gm_to_data(gm)

        image += self.rng.normal(scale=self.noise, size=image.shape)
        return image, objects

    def _gm_to_data(self, gm):
        import ngmix
        o = self.get_object_struct()[0]

        row, col = gm.get_cen()
        e1, e2, T = gm.get_e1e2T()
        fwhm = ngmix.moments.T_to_fwhm(T)
        flux = gm.get_flux()

        o['row'] = row/self.scale
        o['col'] = col/self.scale
        o['e1'] = e1
        o['e2'] = e2
        o['T'] = T
        o['fwhm'] = fwhm
        o['flux'] = flux
        return o

    def get_object_image(self):
        """
        get image of a single object
        """

        gm = self.get_object()

        image = gm.make_image(self.dims, jacobian=self.jacobian)
        return image, gm

    def get_object(self):
        import ngmix

        row, col = self.get_position()

        g1, g2 = self.get_shape()

        fwhm = self.get_fwhm()
        T = ngmix.moments.fwhm_to_T(fwhm)

        flux = self.get_flux()

        pars = [
            row*self.scale,
            col*self.scale,
            g1,
            g2,
            T,
            flux,
        ]
        gm0 = ngmix.GMixModel(pars, 'gauss')
        return gm0.convolve(self.psf)

    def get_object_struct(self, n=1):
        dt = [
            ('row', 'f8'),
            ('col', 'f8'),
            ('e1', 'f8'),
            ('e2', 'f8'),
            ('fwhm', 'f8'),
            ('T', 'f8'),
            ('flux', 'f8'),
        ]
        return np.zeros(n, dtype=dt)

    def get_position(self):
        row = self.rng.uniform(
            low=0+self.border,
            high=self.dims[0]-self.border-1,
        )
        col = self.rng.uniform(
            low=0+self.border,
            high=self.dims[1]-self.border-1,
        )
        return row, col

    def get_shape(self):
        while True:
            g1, g2 = self.rng.normal(scale=self.shape_noise, size=2)
            g = np.sqrt(g1**2 + g2**2)
            if g < 1.0:
                break
        return g1, g2

    def get_fwhm(self):
        return self.rng.uniform(
            low=self.fwhm_range[0],
            high=self.fwhm_range[1],
        )

    def get_flux(self):
        return self.rng.uniform(
            low=self.flux_range[0],
            high=self.flux_range[1],
        )

    def _set_psf(self):
        """
        set the psf as a gaussian
        """
        import ngmix
        T = ngmix.moments.fwhm_to_T(self.psf_fwhm)
        pars = [0.0, 0.0, 0.0, 0.0, T, 1.0]
        self.psf = ngmix.GMixModel(pars, 'gauss')

    def _set_jacobian(self):
        """
        set the psf as a gaussian
        """
        import ngmix

        self.jacobian = ngmix.DiagonalJacobian(
            row=0,
            col=0,
            scale=self.scale,
        )


class GridSim(Sim):
    def make_image(self):
        self._setup_grid()
        return super(GridSim, self).make_image()

    def get_position(self):
        row = self._grid_rows[self._grid_counter]
        col = self._grid_cols[self._grid_counter]
        self._grid_counter += 1
        return row, col

    def _setup_grid(self):
        # size of each dimension
        ngrid = int(np.sqrt(self.nobj))

        ntot = ngrid**2
        if ntot < self.nobj:
            ngrid += 1

        row_spacing = int(self.dims[0]/(ngrid+1))
        col_spacing = int(self.dims[1]/(ngrid+1))

        self._grid_rows = np.zeros(ngrid**2)
        self._grid_cols = np.zeros(ngrid**2)

        itot = 0
        for irow in range(ngrid):
            row = (irow+1)*row_spacing
            for icol in range(ngrid):
                col = (icol+1)*col_spacing

                self._grid_rows[itot] = row
                self._grid_cols[itot] = col
                itot += 1

        self._grid_counter = 0


def gauss_kernel(*, fwhm, dims):
    """
    Create an gaussian kernel image

    Parameters
    ----------
    fwhm: float
        gaussian fwhm in pixels
    dims: sequence
        The dimensions of the image

    Returns
    -------
    image
    """

    cen = (np.array(dims)-1.0)/2.0
    return gauss_image(
        e1=0.0,
        e2=0.0,
        fwhm=fwhm,
        flux=1.0,
        dims=dims,
        cen=cen,
    )


def gauss_image(*, e1, e2, fwhm, flux, dims, cen):
    """
    Create an image with the specified gaussian

    Parameters
    ----------
    e1: float
        gaussian e1
    e2: float
        gaussian e2
    fwhm: float
        gaussian fwhm in pixels
    flux: float
        total flux of image
    dims: sequence
        The dimensions of the image
    cen: sequence
        The center in [row,col]

    Returns
    -------
    image
    """

    T = fwhm_to_T(fwhm)

    Irr = T/2*(1-e1)
    Irc = T/2*e2
    Icc = T/2*(1+e1)

    det = Irr*Icc - Irc**2
    if det < 1.0e-20:
        raise RuntimeError('Determinant is zero')

    Wrr = Irr/det
    Wrc = Irc/det
    Wcc = Icc/det

    rows, cols = np.ogrid[
        0:dims[0],
        0:dims[1],
    ]

    rm = np.array(rows - cen[0], dtype='f8')
    cm = np.array(cols - cen[1], dtype='f8')

    rr = rm**2*Wcc - 2*rm*cm*Wrc + cm**2*Wrr

    rr = 0.5*rr
    image = np.exp(-rr)

    image *= flux/image.sum()
    return image
