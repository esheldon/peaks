import numpy as np
from numba import njit


def find_peaks(*,
               image,
               kernel=None,
               kernel_fwhm=None,
               noise=None,
               noise_thresh=1.0,
               thresh=None,
               max_peaks=None):
    """
    run the peak finder

    Parameters
    ----------
    image: 2-d array
        The image in which to find peaks
    kernel: 2-d array
        Kernel by which to convolve the image
    kernel_fwhm: float
        The code will generate a gaussian kernel with this fwhm in pixels.
    thresh: float
        An absolute threshold above which peaks must be to be detected.
        You can also specify noise and noise_threshold, see below.
    noise: float
        The noise level of the image, to be combined with noise_thresh
        to make an absolute threshold
    noise_thresh: float
        if noise is sent, the absolute threshold will
        be noise*noise_thresh.  Default 1.0
    max_peaks: int
        Maximum number of peaks that can be detected; the
        default is the number of pixels in the image
    """

    finder = PeakFinder(
        image=image,
        kernel=kernel,
        kernel_fwhm=kernel_fwhm,
        noise=noise,
        noise_thresh=noise_thresh,
        thresh=thresh,
        max_peaks=max_peaks,
    )
    finder.go()
    return finder.objects


class PeakFinder(object):
    def __init__(self,
                 *,
                 image,
                 kernel=None,
                 kernel_fwhm=None,
                 noise=None,
                 noise_thresh=1.0,
                 thresh=None,
                 max_peaks=None):
        """
        peak finder

        Parameters
        ----------
        image: 2-d array
            The image in which to find peaks
        kernel: 2-d array
            Kernel by which to convolve the image.  Send either kernel= or
            kernel_fwhm=
        kernel_fwhm: float
            The code will generate a gaussian kernel with this fwhm in pixels.
            Send either kernel= or kernel_fwhm=
        thresh: float
            An absolute threshold above which peaks must be to be detected.
            You can also specify noise and noise_threshold, see below.
        noise: float
            The noise level of the image, to be combined with noise_thresh
            to make an absolute threshold
        noise_thresh: float
            if noise is sent, the absolute threshold will
            be noise*noise_thresh.  Default 1.0
        max_peaks: int
            Maximum number of peaks that can be detected; the
            default is the number of pixels in the image
        """

        self.image = image
        self._set_kernel(kernel=kernel, kernel_fwhm=kernel_fwhm)

        if max_peaks is None:
            self.max_peaks = image.size
        else:
            self.max_peaks = int(max_peaks)

        self.noise = noise
        self.noise_thresh = noise_thresh

        if noise is not None:
            self.thresh = float(noise*noise_thresh)
        elif thresh is not None:
            self.thresh = float(thresh)
        else:
            raise ValueError(
                'send either thresh= or the combination '
                'of noise= and noise_thresh='
            )

        self._set_convolved_image()

    def _set_kernel(self, kernel=None, kernel_fwhm=None):
        if kernel is None and kernel_fwhm is None:
            raise ValueError('send kernel= or kernel_fwhm=')

        if kernel is not None:
            kernel = np.array(kernel, copy=False)
            if len(kernel.shape) != 2:
                raise ValueError('expected 2d array for kernel, '
                                 'got %s' % str(kernel.shape))
        else:
            from . import sim
            kernel = sim.gauss_kernel(fwhm=kernel_fwhm)

        self.kernel = kernel

    def go(self):
        """
        find the peaks and set the row, col arrays for
        each peak
        """

        objects = self.get_object_struct(self.max_peaks)

        self.npeaks = find_peaks_in_convolved_image(
            self.convolved_image,
            self.thresh,
            objects['row'],
            objects['col'],
        )

        self._objects = objects[:self.npeaks]

    @property
    def objects(self):
        """
        get the row and column arrays holding the peak
        positions
        """
        if not hasattr(self, '_objects'):
            raise RuntimeError('run go() first')

        return self._objects

    def _set_convolved_image(self):
        """
        set the convolved image
        """

        import scipy.signal

        self.convolved_image = scipy.signal.convolve2d(
            self.image,
            self.kernel,
            mode='same',
        )

    def get_object_struct(self, n=1):
        dt = [
            ('row', 'i4'),
            ('col', 'i4'),
        ]
        return np.zeros(n, dtype=dt)


@njit
def find_peaks_in_convolved_image(image, thresh, peakrows, peakcols):
    """
    find peaks by looking for points around which all values
    are lower

    image: 2d array
        An image in which to find peaks
    thresh: float
        Peaks must be higher than this value.  You would typically
        set this to some multiple of the noise level.
    peakrows: array
        an array to fill with peak locations
    peakcols: array
        an array to fill with peak locations
    """

    npeaks = 0

    nrows, ncols = image.shape
    for irow in range(nrows):
        if irow == 0 or irow == nrows-1:
            continue

        rowstart = irow-1
        rowend = irow+1
        for icol in range(ncols):
            if icol == 0 or icol == ncols-1:
                continue

            colstart = icol-1
            colend = icol+1

            val = image[irow, icol]
            if val > thresh:

                ispeak = True
                for checkrow in range(rowstart, rowend+1):
                    for checkcol in range(colstart, colend+1):
                        if checkrow == irow and checkcol == icol:
                            continue

                        checkval = image[checkrow, checkcol]
                        if checkval > val:
                            # we found an adjacent value that is higher
                            ispeak = False

                            # break out of inner loop
                            break

                    if not ispeak:
                        # also break out of outer loop
                        break

                if ispeak:
                    npeaks += 1
                    ipeak = npeaks-1
                    peakrows[ipeak] = irow
                    peakcols[ipeak] = icol

    return npeaks
