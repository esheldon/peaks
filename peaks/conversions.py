def fwhm_to_sigma(fwhm):
    """
    convert gaussian fwhm to T
    """
    return fwhm/2.3548200450309493

def fwhm_to_T(fwhm):
    """
    convert gaussian fwhm to T
    """
    sigma = fwhm_to_sigma(fwhm)
    return 2*sigma**2
