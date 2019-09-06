import numpy as np


class Sim(dict):
    def __init__(self,
                 *,
                 rng,
                 dims,
                 nobj,
                 border=0,
                 noise=0.01,
                 shape_noise=0.2,
                 fwhm_range=(0.9, 3.0),
                 flux_range=(10.0, 100.0)):

        self.rng = rng
        self.dims = dims
        self.border = int(border)
        self.nobj = int(nobj)
        self.shape_noise = float(shape_noise)
        self.fwhm_range = fwhm_range
        self.flux_range = flux_range
        self.noise = noise

        assert len(dims) == 2
        assert len(fwhm_range) == 2
        assert len(flux_range) == 2

    def make_image(self):
        """
        make an image
        """

        image = np.zeros(self.dims)

        object_data = self.get_object_struct(n=self.nobj)

        for i in range(self.nobj):
            this_image, odata = self.get_object_image()

            image += this_image

            object_data[i] = odata[0]

        image += self.rng.normal(scale=self.noise, size=image.shape)
        return image, object_data

    def get_object_image(self):
        """
        get image of a single object
        """

        odata = self.get_object_data()
        image = gauss_image(
            odata['e1'][0],
            odata['e2'][0],
            odata['T'][0],
            odata['flux'][0],
            self.dims,
            [odata['row'][0], odata['col'][0]],
        )

        return image, odata

    def get_object_data(self):
        st = self.get_object_struct()
        st['row'], st['col'] = self.get_position()
        st['e1'], st['e2'] = self.get_shape()
        st['T'] = self.get_T()
        st['flux'] = self.get_flux()
        return st

    def get_object_struct(self, n=1):
        dt = [
            ('row', 'f8'),
            ('col', 'f8'),
            ('e1', 'f8'),
            ('e2', 'f8'),
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
            e1, e2 = self.rng.normal(scale=self.shape_noise, size=2)
            e = np.sqrt(e1**2 + e2**2)
            if e < 1.0:
                break
        return e1, e2

    def get_T(self):
        fwhm = self.rng.uniform(
            low=self.fwhm_range[0],
            high=self.fwhm_range[1],
        )
        sigma = fwhm/2.3548200450309493
        T = 2*sigma**2
        return T

    def get_flux(self):
        return self.rng.uniform(
            low=self.flux_range[0],
            high=self.flux_range[1],
        )


def gauss_image(e1, e2, T, flux, dims, cen):
    """
    Create an image with the specified gaussian

    Parameters
    ----------
    dims: sequence
        The dimensions of the image
    cen: sequence
        The center in [row,col]
    cov: sequence
        A three element sequence representing the covariance matrix
        [Irr,Irc,Icc].  Note this only corresponds exactly to the moments of
        the object for a gaussian model.  For an simple bivariate gaussian, Irr
        and Icc are sigma1**2 sigma2**2, but using the full matrix allows for
        other angles of oriention.
    flux: number, optional
        The total flux in the image.  Default 1.0.  If None, the image is not
        normalized.
    Returns
    -------
    Image: 2-d array
        The returned image is a 2-d numpy array of 8 byte floats.
    Example
    -------
        dims=[41,41]
        cen=[20,20]
        cov=[8,2,4] # [Irr,Irc,Icc]
        im = ogrid_image('gauss',dims,cen,cov)
    """

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
