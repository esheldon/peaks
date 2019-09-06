import numpy as np


def get_moments(*,
                objects,
                image,
                fwhm,
                scale=1.0,
                jacobian=None,
                weight=None,
                noise=None):

    meas = Moments(
        image=image,
        fwhm=fwhm,
        scale=scale,
        jacobian=jacobian,
        weight=weight,
        noise=noise,
    )
    return meas.go(objects)


class Moments(object):
    def __init__(self, *,
                 image,
                 fwhm,
                 scale=1.0,
                 jacobian=None,
                 weight=None,
                 noise=None):

        self.image = image
        self.fwhm = fwhm

        self._set_jacobian(scale=scale, jacobian=jacobian)
        self._set_weight_function()
        self._set_weight_image(weight, noise)
        self._set_obs()

    def get_result(self):
        """
        get the result structure
        """
        if not hasattr(self, '_result'):
            raise RuntimeError('run go() first')

        return self._result

    def go(self, objects):
        """
        measure weight moments for the input objects

        Parameters
        ----------
        objects: array
            Must have fields 'row' and 'col'
        """

        obs = self.obs

        output = self._get_output_struct(n=objects.size)

        for i in range(objects.size):
            row = objects['row'][i]
            col = objects['col'][i]

            jac = obs.jacobian
            jac.set_cen(row=row, col=col)
            obs.jacobian = jac

            moms = self.weight.get_weighted_moments(
                obs,
                self.maxrad,
            )

            self._set_moms(objects, output, moms, jac.scale, i)

        self._result = output
        return output

    def _set_moms(self, objects, output, moms, scale, i):
        output['flags'][i] = moms['flags']

        output['row'][i] = objects['row'][i]
        output['col'][i] = objects['col'][i]

        output['flux'][i] = moms['flux']*scale**2
        output['s2n'][i] = moms['s2n']

        if moms['flags'] == 0:
            output['e1'][i] = moms['e'][0]
            output['e1_err'][i] = np.sqrt(moms['e_cov'][0, 0])
            output['e2'][i] = moms['e'][1]
            output['e2_err'][i] = np.sqrt(moms['e_cov'][1, 1])

            output['T'][i] = moms['T']
            output['T_err'][i] = moms['T_err']
            output['flux_err'][i] = moms['flux_err']*scale**2

    def _get_output_struct(self, n=1):
        dt = [
            ('flags', 'i4'),
            ('row', 'f8'),
            ('col', 'f8'),
            ('s2n', 'f8'),
            ('e1', 'f8'),
            ('e1_err', 'f8'),
            ('e2', 'f8'),
            ('e2_err', 'f8'),
            ('T', 'f8'),
            ('T_err', 'f8'),
            ('flux', 'f8'),
            ('flux_err', 'f8'),
        ]
        st = np.zeros(n, dtype=dt)

        for n in st.dtype.names:
            if n == 'flags':
                continue
            if 'err' in n:
                st[n] = +9999.0
            else:
                st[n] = -9999.0
        return st

    def _set_jacobian(self, scale=1.0, jacobian=None):
        if jacobian is not None:
            self.jacobian = jacobian.copy()
            self.jacobian.set_cen(0, 0)
        else:
            import ngmix
            self.jacobian = ngmix.DiagonalJacobian(
                row=0,
                col=0,
                scale=scale,
            )

    def _set_weight_image(self, weight, noise):
        if weight is not None:
            self.weight_image = weight
        elif noise is not None:
            self.weight_image = self.image.copy()
            self.weight_image[:, :] = 1.0/noise**2
        else:
            self.weight_image = self.image.copy()
            self.weight_image[:, :] = 1.0

    def _set_obs(self):
        import ngmix

        self.obs = ngmix.Observation(
            image=self.image,
            weight=self.weight_image,
            jacobian=self.jacobian,
        )

    def _set_weight_function(self):
        import ngmix

        T = ngmix.moments.fwhm_to_T(self.fwhm)
        pars = [
            0.0,
            0.0,
            0.0,
            0.0,
            T,
            1.0,
        ]
        weight = ngmix.GMixModel(pars, 'gauss')

        weight.set_norms()

        # set so max is 1
        data = weight.get_data()
        weight.set_flux(1/data['norm'][0])
        weight.set_norms()
        self.weight = weight

        sigma = np.sqrt(T/2)
        self.maxrad = 5*sigma
