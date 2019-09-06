import numpy as np
from numba import njit


class PeakFinder(object):
    def __init__(self,
                 *,
                 image,
                 kernel,
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
            Kernel by which to convolve the image
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
        self.kernel = kernel
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

    def go(self):
        """
        find the peaks and set the row, col arrays for
        each peak
        """

        rows = np.zeros(self.max_peaks)
        cols = np.zeros(self.max_peaks)

        self.npeaks = find_peaks(
            self.convolved_image,
            self.thresh,
            rows,
            cols,
        )
        self.rows = rows[:self.npeaks]
        self.cols = rows[:self.npeaks]

    def get_peaks(self):
        """
        get the row and column arrays holding the peak
        positions
        """
        if not hasattr(self, 'rows'):
            raise RuntimeError('run go() first')

        return self.rows, self.cols

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


@njit
def find_peaks(image, thresh, peakrows, peakcols):
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
