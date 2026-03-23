import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator


def clcd13x6(alphas,NRe):
    # --- Cargar grilla ---
    npz_file = os.path.join(os.path.dirname(__file__), 'polarCLCD.npz')
    # npz_file = "polarCLCD.npz"
    meshfile = np.load(npz_file)

    Re = meshfile["Re"]
    alpha = meshfile["alpha"]
    data = meshfile["data"]
    CL_grid = data[0, :, :]
    CD_grid = data[1, :, :]

    interp_CL = RegularGridInterpolator((Re, alpha), CL_grid, bounds_error=False, fill_value=None)
    interp_CD = RegularGridInterpolator((Re, alpha), CD_grid, bounds_error=False, fill_value=None)


    points = np.array([[r, a] for r, a in zip(NRe, alphas)])

    CL_interp = interp_CL(points)
    CD_interp = interp_CD(points)

    # --- Reshape para ver como matriz (Re x alpha) ---
    # CL_interp = CL_interp.reshape(len(NRe), len(alphas))
    # CD_interp = CD_interp.reshape(len(NRe), len(alphas))

    return CL_interp, CD_interp
