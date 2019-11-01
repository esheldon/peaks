# peaks
A simple peak finder

Examples
--------
```python
import peaks

# using the convenience function
objects = peaks.find_peaks(image=image, kernel_fwhm=3.5)

# the objects array contains 'row' and 'col' for each peak, good
# to the integer level.  Get a more refind center and other
# moments.

moments = peaks.get_moments(
    objects=objects,
    image=image,
    fwhm=1.2,
    scale=0.263,
)
#  Note the moments can work in a sky coordinates with a local
# jacobian (ngmix.Jacobian)
moments = peaks.get_moments(
    objects=objects,
    image=image,
    fwhm=1.2,
    jacobian=jacobian,
)

# using the class
finder = peaks.PeakFinder(image=image, kernel_fwhm=3.5)
finder.go()
objects = finder.objects

```
Depencencies
------------
```
- numpy
- numba
- optionally ngmix for measuring weighted moments and simulation tests
- optionally biggles and images for visualization
```
