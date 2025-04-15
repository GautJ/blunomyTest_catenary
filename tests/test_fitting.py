import numpy as np
from catenary_fitting.fitting import catenary, fit_catenary

def test_fit_catenary_on_perfect_data():
    x = np.linspace(-5, 5, 100)
    true_params = [0, 0, 10]
    z = catenary(x, *true_params)

    fitted_params = fit_catenary(x, z)

    # Assert that fitted parameters are close to true values
    assert np.allclose(fitted_params, true_params, atol=1e-2)
