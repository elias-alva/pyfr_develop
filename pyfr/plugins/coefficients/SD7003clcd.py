import numpy as np
from pyfr.plugins.coefficients import SD7003jns as SD

def ClAir(alpha):
    # Airfoil data
    SD7003A, SD7003CL, SD7003CD, SD7003CM, SD7003REY = SD.SD7003()
    del SD7003CM, SD7003REY, SD7003CD
    n = 1
    xdat = np.zeros((np.size(SD7003CL,0),3))
    xdat[:,0] = SD7003A[:,n-1]                                                  # Alpha
    xdat[:,1] = SD7003CL[:,n-1]                                                 # Cl data
    a_minus = np.linspace(-180,min(xdat[:,0])-1,89)                             # AoA array from -180° to -90°
    a_plus = np.linspace(max(xdat[:,0])+1,180,89)                               # AoA array from 90° to 180°
    cl_minus = np.sin(2*a_minus*np.pi/180.)                                     # Cl coefficient from -180° to -90°
    cl_plus = np.sin(2*a_plus*np.pi/180.)                                       # Cl coefficient from 90° to 180°
    # Arrange of AoA from -180 to 180
    alfa = np.zeros((a_minus.shape[0] + xdat.shape[0] + a_plus.shape[0], 1))
    alfa[0:a_minus.shape[0],0] = a_minus
    alfa[a_minus.shape[0]:a_minus.shape[0]+xdat.shape[0],0] = xdat[:,0]
    alfa[a_minus.shape[0]+xdat.shape[0]:,0] = a_plus
    alfa = alfa.ravel()
    # Arrange of Cl from -180 to 180
    cl = np.zeros((cl_minus.shape[0] + xdat.shape[0] + cl_plus.shape[0], 1))
    cl[0:cl_minus.shape[0],0] = cl_minus
    cl[cl_minus.shape[0]:cl_minus.shape[0]+xdat.shape[0],0] = xdat[:,1]
    cl[cl_minus.shape[0]+xdat.shape[0]:,0] = cl_plus
    cl = cl.ravel()
    # Interpolation for alpha value
    cl_a = np.interp(alpha,alfa,cl)

    return cl_a

def CdAir(alpha):
    # Airfoil data
    SD7003A, SD7003CL, SD7003CD, SD7003CM, SD7003REY = SD.SD7003()
    del SD7003CM, SD7003REY, SD7003CL
    # Cd max and min for modified stalled flat plate
    n = 1
    cdmax = np.max(SD7003CD)
    cdmin = np.min(SD7003CD)
    xdat = np.zeros((np.size(SD7003CD,0),3))
    xdat[:,0] = SD7003A[:,n-1]                                                  # Alpha
    xdat[:,1] = SD7003CD[:,n-1]                                                 # Cd data
    a_minus = np.linspace(-180,min(xdat[:,0])-1,89)                             # AoA array from -180° to -90°
    a_plus = np.linspace(max(xdat[:,0])+1,180,89)                               # AoA array from 90° to 180°
    cd_minus = cdmin+(cdmax-cdmin)*np.square(np.sin(a_minus*np.pi/180.))        # Cd coefficient from -180° to -90°
    cd_plus = cdmin+(cdmax-cdmin)*np.square(np.sin(a_plus*np.pi/180.))          # Cd coefficient from 90° to 180°
    # Arrange of AoA from -180 to 180
    alfa = np.zeros((a_minus.shape[0] + xdat.shape[0] + a_plus.shape[0], 1))
    alfa[0:a_minus.shape[0],0] = a_minus
    alfa[a_minus.shape[0]:a_minus.shape[0]+xdat.shape[0],0] = xdat[:,0]
    alfa[a_minus.shape[0]+xdat.shape[0]:,0] = a_plus
    alfa = alfa.ravel()
    # Arrange of Cd from -180 to 180
    cd = np.zeros((cd_minus.shape[0] + xdat.shape[0] + cd_plus.shape[0], 1))
    cd[0:cd_minus.shape[0],0] = cd_minus
    cd[cd_minus.shape[0]:cd_minus.shape[0]+xdat.shape[0],0] = xdat[:,1]
    cd[cd_minus.shape[0]+xdat.shape[0]:,0] = cd_plus
    cd = cd.ravel()
    # Interpolation for alpha value
    cd_a = np.interp(alpha,alfa,cd)

    return cd_a