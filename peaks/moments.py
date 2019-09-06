def fwhm_to_T(fwhm):
    """
    convert gaussian fwhm to T
    """
    sigma = fwhm/2.3548200450309493
    return 2*sigma**2

