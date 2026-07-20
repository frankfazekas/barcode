"""Gaussian-derivative Lucas-Kanade optical flow, vendored from OpticalFlow3D.

THIRD-PARTY CODE — DO NOT REFACTOR.
====================================================================================
Source : https://github.com/aicjanelia/OpticalFlow3D  (src/Python/calc_flow.py)
Commit : bf2d758f8f4d4eb0c03c2b90bda8fbb610a6bafe  (2026-02-11)
Cite   : Lee et al., "OpticalFlow3D: A tool for measuring amorphous motion in
         three-dimensional fluorescence microscopy images", bioRxiv 2026.

BSD 3-Clause License. Copyright (c) 2025, Howard Hughes Medical Institute.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation and/or
   other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
====================================================================================

``calc_flow2D`` and ``calc_flow3D`` below are copied verbatim so this file stays
diffable against upstream. The only departures from ``src/Python/calc_flow.py`` are:

* ``process_flow`` — upstream's file-discovery and TIFF-writing orchestration — is not
  vendored. BARCODE has its own readers (``analysis/volumetric/reader.py``) and its own
  windowing (``analysis/volumetric/flow.py``). Its imports (``pathlib``, ``os``, ``re``,
  ``tifffile``, ``pandas``, ``datetime``, ``natsort``) dropped with it, so this module
  adds no dependency beyond numpy and scipy.
* This docstring, in place of upstream's one-line module docstring.

Note that both functions call ``sys.exit()`` on malformed input, which would kill the
Tkinter process. ``analysis/volumetric/flow.py`` validates every precondition (4-D
input, odd ``N_T``, ``N_T >= 6*tSig+1``) before calling in, so those paths are
unreachable from BARCODE.

``calc_flow2D`` is used only as an independent oracle in ``tests/test_flow_3d.py``;
the 2D BARCODE branch continues to use OpenCV's Farneback and is untouched.
"""

import math
import numpy as np
from scipy.ndimage import correlate1d
import sys


def calc_flow2D(images,xySig=3,tSig=1,wSig=4):
    """
    Calculate two-dimensional optical flow fields from input images.

    The calc_flow2D function calculates optical flow velocities for a single
    z-slice. Surrounding images in time are necessary to perform the
    calculations. To peform calculations on an entire timelapse, see the
    function parse_flow.

    This script uses the convention that (0,0) is located in the upper-left
    corner of an image. This is inline with conventions used in other programs
    (e.g., ImageJ/FIJI), but note that it means that positive y-velocities point
    down, which can be non-intuitive in some cases.

    ARGS:
    images: 3D numpy array with dimensions N_T, N_Y, N_X
             N_T should be odd as only the central timepoint will be analyzed.
             N_T must be greater than or equal to 3*tSig+1.
    xySig:  sigma value for smoothing in all spatial dimensions. Default 3.
             Larger values remove noise but remove spatial detail.
    tSig:   sigma value for smoothing in the temporal dimension. Default 1.
             Larger values remove noise but remove temporal detail.
    wSig:   sigma value for Lucas-Kanade neighborhood. Default is 4.
             Larger values include a larger neighboorhood in the
             Lucas-Kanade constraint and will smooth over small features.

    RETURNS:
    vx:    Velocity in the x direction, reported as pixels/frame
    vy:    Velocity in the y direction, reported as pixels/frame
    rel:   Reliability, the smallest eigenvalue of (A'wA)
            This is a measure of confidence in the linear algebra solution
            and can be used to mask the velocities for downstream analysis.
    """

    ### Check the function inputs ##############################################
    # Check that the images are 2D + time
    if not(len(images.shape)==3):
        sys.exit('ERROR: Input image must be a 3D matrix with dimensions N_T, N_Y, N_X')
    # Check image size against tSig
    Nt = images.shape[0]
    if Nt < 6*tSig+1:
        # The kernel size is 3*tSig in time (or 6*tSig total)
        # There is also a central pixel, so need at least 6*tSig+1 images in t
        sys.exit('ERROR: Input images will lead to edge effects. N_T must be >= 6*tSig+1')
    # Check for an odd number of frames
    if not(Nt % 2):
        sys.exit('ERROR: Input images must have an odd number of timepoints. Only the central time point is analyzed')
    NtSlice = math.ceil(Nt/2)-1 # -1 because python indexing starts from 0
    # Use float for calculations
    images = images.astype(np.float64)


    ### Set up filters #########################################################
    # Common terms
    x = np.arange(-math.ceil(3*xySig),math.ceil(3*xySig)+1)
    xySig2 = xySig/4
    y = np.arange(-math.ceil(3*xySig2),math.ceil(3*xySig2)+1)
    fderiv = np.exp(-x*x/2/xySig/xySig)/math.sqrt(2*math.pi)/xySig
    fsmooth = np.exp(-y*y/2/xySig2/xySig2)/math.sqrt(2*math.pi)/xySig2
    gderiv = x/xySig/xySig
    gsmooth = 1

    # Build y-gradient filter kernels (along first spatial dimension)
    yFil1 = (fderiv*gderiv)
    xFil1 = (fsmooth*gsmooth)
    # Build x-gradient filter kernels (along second spatial dimension)
    yFil2 = (fsmooth*gsmooth)
    xFil2 = (fderiv*gderiv)

    # Build t-gradient filter kernels (t = third dimension)
    t = np.arange(-math.ceil(3*tSig),math.ceil(3*tSig)+1)
    fx = np.exp(-x*x/2/xySig/xySig)/math.sqrt(2*math.pi)/xySig
    ft = np.exp(-t*t/2/tSig/tSig)/math.sqrt(2*math.pi)/tSig
    gx = 1
    gt = t/tSig/tSig
    yFil3 = (fx*gx)
    xFil3 = yFil3
    tFil3 = ft*gt

    # Structure tensor -- Lucas Kanade neighborhood filter
    wRange = np.arange(-math.ceil(3*wSig),math.ceil(3*wSig)+1)
    gw = np.exp(-wRange*wRange/2/wSig/wSig)/math.sqrt(2*math.pi)/wSig
    yFil4 = gw
    xFil4 = gw

    # Throughout will use del to keep the memory clear as this processing is memory intensive
    del gderiv, gsmooth, gt, gw, gx, ft, fx, fsmooth, fderiv, x, y, t, wRange


    ### Spatial and Temporal Gradients #########################################
    # Spatial gradients require only the frame of interest, while  the temporal
    # gradient requires N_T >= 2*3*tSig+1. Keep only the relevant slice of dtI
    # after it is calculated.

    # dtI is split into two steps to save memory and processing time
    dtI = correlate1d(images, tFil3, axis=0, mode='nearest')
    dtI = dtI[NtSlice,:]
    images = images[NtSlice,:]
    dtI = correlate1d(correlate1d(dtI, yFil3, axis=0, mode='nearest'), xFil3, axis=1, mode='nearest')
    del xFil3, yFil3, tFil3

    dyI = correlate1d(correlate1d(images, yFil1, axis=0, mode='nearest'), xFil1, axis=1, mode='nearest')
    del xFil1, yFil1

    dxI = correlate1d(correlate1d(images, yFil2, axis=0, mode='nearest'), xFil2, axis=1, mode='nearest')
    del xFil2, yFil2
    del images


    ### Structure Tensor Inputs ################################################
    # The following calculations are for the individual elements of the
    # matrices required for the optical flow calculation, incorporating
    # Gaussian weighting into the Lucas-Kanade constraint.

    # Time components
    wdtx = correlate1d(correlate1d(dxI*dtI, yFil4, axis=0, mode='nearest'), xFil4, axis=1, mode='nearest')
    wdty = correlate1d(correlate1d(dyI*dtI, yFil4, axis=0, mode='nearest'), xFil4, axis=1, mode='nearest')
    del dtI

    # Spatial Components
    wdxy = correlate1d(correlate1d(dxI*dyI, yFil4, axis=0, mode='nearest'), xFil4, axis=1, mode='nearest')
    wdx2 = correlate1d(correlate1d(dxI*dxI, yFil4, axis=0, mode='nearest'), xFil4, axis=1, mode='nearest')
    del dxI
    wdy2 = correlate1d(correlate1d(dyI*dyI, yFil4, axis=0, mode='nearest'), xFil4, axis=1, mode='nearest')
    del dyI
    del xFil4, yFil4


    ### Optical Flow Calculations ##############################################
    # Equation is v = (A' w A)^-1 A' w b
    # A = -[dxI dyI]
    # b = [dtI]
    # w multiplication is incorporated in the structure tensor inputs above
    # A' w b = -[wdtx wdty]  (minus sign because of negative sign on A)
    # (A' w A) = [a=wdx2 b=wdxy ; c=wdxy d=wdy2]
    # A^-1 = [a b ; c d]^-1 = (1/det(A))[d - b; -c a]
    determinant = (wdx2*wdy2) - (wdxy*wdxy)
    vx = ((determinant+np.finfo(float).eps)**-1)*((wdy2*-wdtx)+(-wdxy*-wdty))
    vy = ((determinant+np.finfo(float).eps)**-1)*((-wdxy*-wdtx)+(wdx2*-wdty))
    del wdtx, wdty, wdxy


    ### Eigenvalues for Reliability ############################################
    # solve det(A^T w A - lamda I) = 0
    # (A' w A) = [a=wdx2 b=wdxy ; c=wdxy d=wdy2]
    trace = wdx2 + wdy2
    del wdx2, wdy2

    L1 = (trace + np.sqrt(trace**2 - 4*determinant))/2
    L2 = (trace - np.sqrt(trace**2 - 4*determinant))/2
    rel = np.real(np.minimum(L1,L2))
    del L1, L2


    ### Return Outputs #########################################################
    return vx, vy, rel


def calc_flow3D(images,xyzSig=3,tSig=1,wSig=4):
    """
    Calculate three-dimensional optical flow fields from input z-stacks.

    The calc_flow3D function calculates optical flow velocities for a single
    z-stack of images. Surrounding z-stacks in time are necessary to perform
    the calculations. To peform calculations on an entire timelapse, see
    the function parse_flow.

    This script uses the convention that (0,0) is located in the upper-left
    corner of an image. This is inline with conventions used in other
    programs (e.g., ImageJ/FIJI), but note that it means that positive
    y-velocities point down, which can be non-intuitive in some cases.

    ARGS:
    images: 4D array with dimensions N_T, N_Z, N_Y, N_X
             N_T should be odd as only the central timepoint will be analyzed.
             N_T must be greater than or equal to 6*tSig+1.
    xyzSig:  sigma value for smoothing in all spatial dimensions. Default 3.
             Larger values remove noise but remove spatial detail.
    tSig:   sigma value for smoothing in the temporal dimension. Default 1.
             Larger values remove noise but remove temporal detail.
    wSig:   sigma value for Lucas-Kanade neighborhood. Default is 4.
             Larger values include a larger neighboorhood in the
             Lucas-Kanade constraint and will smooth over small features.

    RETURNS:
    vx:    Velocity in the x direction, reported as pixels/frame
    vy:    Velocity in the y direction, reported as pixels/frame
    vz:    Velocity in the z direction, reported as pixels/frame
    rel:   Reliability, the smallest eigenvalue of (A'wA)
            This is a measure of confidence in the linear algebra solution
            and can be used to mask the velocities for downstream analysis.
    """

    ### Check the function inputs ##############################################
    # Check that the images are 2D + time
    if not(len(images.shape)==4):
        sys.exit('ERROR: Input image must be a 3D matrix with dimensions N_T, N_Z, N_Y, N_X')
    # Check image size against tSig
    Nt = images.shape[0]
    if Nt < 6*tSig+1:
        # The kernel size is 3*tSig in time (or 6*tSig total)
        # There is also a central pixel, so need at least 6*tSig+1 images in t
        sys.exit('ERROR: Input images will lead to edge effects. N_T must be >= 6*tSig+1')
    # Check for an odd number of frames
    if not(Nt % 2):
        sys.exit('ERROR: Input images must have an odd number of timepoints. Only the central time point is analyzed')
    NtSlice = math.ceil(Nt/2)-1 # -1 because python indexing starts from 0
    # Use float for calculations
    images = images.astype(np.float64)


    ### Set up filters #########################################################
    # Common terms
    x = np.arange(-math.ceil(3*xyzSig),math.ceil(3*xyzSig)+1)
    xyzSig2 = xyzSig/4
    y = np.arange(-math.ceil(3*xyzSig2),math.ceil(3*xyzSig2)+1)
    fderiv = np.exp(-x*x/2/xyzSig/xyzSig)/math.sqrt(2*math.pi)/xyzSig
    fsmooth = np.exp(-y*y/2/xyzSig2/xyzSig2)/math.sqrt(2*math.pi)/xyzSig2
    gderiv = x/xyzSig/xyzSig
    gsmooth = 1

    # Build y-gradient filter kernels
    yFil1 = (fderiv*gderiv)
    xFil1 = (fsmooth*gsmooth)
    zFil1 = (fsmooth*gsmooth)
    # Build x-gradient filter kernels
    yFil2 = (fsmooth*gsmooth)
    xFil2 = (fderiv*gderiv)
    zFil2 = (fsmooth*gsmooth)
    # Build z-gradient filter kernels
    yFil3 = (fsmooth*gsmooth)
    xFil3 = (fsmooth*gsmooth)
    zFil3 = (fderiv*gderiv)

    # Build t-gradient filter kernels (t = third dimension)
    t = np.arange(-math.ceil(3*tSig),math.ceil(3*tSig)+1)
    fx = np.exp(-x*x/2/xyzSig/xyzSig)/math.sqrt(2*math.pi)/xyzSig
    ft = np.exp(-t*t/2/tSig/tSig)/math.sqrt(2*math.pi)/tSig
    gx = 1
    gt = t/tSig/tSig
    yFil4 = (fx*gx)
    xFil4 = (fx*gx)
    zFil4 = (fx*gx)
    tFil4 = ft*gt

    # Structure tensor -- Lucas Kanade neighborhood filter
    wRange = np.arange(-math.ceil(3*wSig),math.ceil(3*wSig)+1)
    gw = np.exp(-wRange*wRange/2/wSig/wSig)/math.sqrt(2*math.pi)/wSig
    yFil5 = gw
    xFil5 = gw
    zFil5 = gw

    # Throughout will use del to keep the memory clear as this processing is memory intensive
    del gderiv, gsmooth, gt, gw, gx, ft, fx, fsmooth, fderiv, x, y, t, wRange


    ### Spatial and Temporal Gradients #########################################
    # Spatial gradients require at least N_T = 3 to avoid edge effets, while
    # the temporal gradient requires N_T >= 6*tSig+1.
    dtI = correlate1d(images, tFil4, axis=0, mode='nearest')
    dtI = dtI[NtSlice,:]
    images = images[NtSlice,:]
    dtI = correlate1d(correlate1d(correlate1d(dtI, yFil4, axis=1, mode='nearest'), xFil4, axis=2, mode='nearest'), zFil4, axis=0, mode='nearest')
    del xFil4, yFil4, zFil4, tFil4

    dyI = correlate1d(correlate1d(correlate1d(images, yFil1, axis=1, mode='nearest'), xFil1, axis=2, mode='nearest'), zFil1, axis=0, mode='nearest')
    del xFil1, yFil1, zFil1

    dxI = correlate1d(correlate1d(correlate1d(images, yFil2, axis=1, mode='nearest'), xFil2, axis=2, mode='nearest'), zFil2, axis=0, mode='nearest')
    del xFil2, yFil2, zFil2

    dzI = correlate1d(correlate1d(correlate1d(images, yFil3, axis=1, mode='nearest'), xFil3, axis=2, mode='nearest'), zFil3, axis=0, mode='nearest')
    del xFil3, yFil3, zFil3

    del images


    ### Structure Tensor Inputs ################################################
    # The following calculations are for the individual elements of the
    # matrices required for the optical flow calculation, incorporating
    # Gaussian weighting into the Lucas-Kanade constraint.

    # Time components
    wdtx = correlate1d(correlate1d(correlate1d(dxI*dtI, yFil5, axis=1, mode='nearest'), xFil5, axis=2, mode='nearest'), zFil5, axis=0, mode='nearest')
    wdty = correlate1d(correlate1d(correlate1d(dyI*dtI, yFil5, axis=1, mode='nearest'), xFil5, axis=2, mode='nearest'), zFil5, axis=0, mode='nearest')
    wdtz = correlate1d(correlate1d(correlate1d(dzI*dtI, yFil5, axis=1, mode='nearest'), xFil5, axis=2, mode='nearest'), zFil5, axis=0, mode='nearest')
    del dtI

    # Spatial Components
    wdxy = correlate1d(correlate1d(correlate1d(dxI*dyI, yFil5, axis=1, mode='nearest'), xFil5, axis=2, mode='nearest'), zFil5, axis=0, mode='nearest')
    wdxz = correlate1d(correlate1d(correlate1d(dxI*dzI, yFil5, axis=1, mode='nearest'), xFil5, axis=2, mode='nearest'), zFil5, axis=0, mode='nearest')
    wdx2 = correlate1d(correlate1d(correlate1d(dxI*dxI, yFil5, axis=1, mode='nearest'), xFil5, axis=2, mode='nearest'), zFil5, axis=0, mode='nearest')
    del dxI
    wdyz = correlate1d(correlate1d(correlate1d(dyI*dzI, yFil5, axis=1, mode='nearest'), xFil5, axis=2, mode='nearest'), zFil5, axis=0, mode='nearest')
    wdy2 = correlate1d(correlate1d(correlate1d(dyI*dyI, yFil5, axis=1, mode='nearest'), xFil5, axis=2, mode='nearest'), zFil5, axis=0, mode='nearest')
    del dyI
    wdz2 = correlate1d(correlate1d(correlate1d(dzI*dzI, yFil5, axis=1, mode='nearest'), xFil5, axis=2, mode='nearest'), zFil5, axis=0, mode='nearest')
    del dzI
    del xFil5, yFil5, zFil5


    ### Optical Flow Calculations ##############################################
    # Equation is v = (A' w A)^-1 A' w b
    # A = -[dxI dyI dzI]
    # b = [dtI]
    # w multiplication is incorporated in the structure tensor inputs above
    # A' w b = -[wdtx wdty wdtz]  (minus sign because of negative sign on A)
    # (A' w A) = [a=wdx2 b=wdxy c=wdxz ; d=wdxy e=wdy2 f=wdyz ; g=wdxz h=wdyz i=wdz2]
    # A^-1 = 1/determinant * [A D G ; B E H ; C F I];
    # using inverse notation from here: https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices
    #   A = wdy2*wdz2 - wdyz*wdyz; %ei-fh
    #   B = wdyz*wdxz - wdxy*wdz2; %fg-di
    #   C = wdxy*wdyz - wdy2*wdxz; %dh-eg
    #   D = wdxz*wdyz - wdxy*wdz2; %ch-bi
    #   E = wdx2*wdz2 - wdxz*wdxz; %ai-cg
    #   F = wdxy*wdxz - wdx2*wdyz; %bg-ah
    #   G = wdxy*wdyz - wdxz*wdy2; %bf-ce
    #   H = wdxz*wdxy - wdx2*wdyz; %cd-af
    #   I = wdx2*wdy2 - wdxy*wdxy; %ae-bd

    determinant = (wdx2*wdy2*wdz2) + (2*wdxy*wdxz*wdyz) - (wdy2*wdxz**2) - (wdz2*wdxy**2) - (wdx2*wdyz**2)
    vx = -((determinant + np.finfo(float).eps)**-1)*((wdy2*wdz2 - wdyz*wdyz)*wdtx + (wdxz*wdyz - wdxy*wdz2)*wdty + (wdxy*wdyz - wdxz*wdy2)*wdtz)
    vy = -((determinant + np.finfo(float).eps)**-1)*((wdyz*wdxz - wdxy*wdz2)*wdtx + (wdx2*wdz2 - wdxz*wdxz)*wdty + (wdxz*wdxy - wdx2*wdyz)*wdtz)
    vz = -((determinant + np.finfo(float).eps)**-1)*((wdxy*wdyz - wdy2*wdxz)*wdtx + (wdxy*wdxz - wdx2*wdyz)*wdty + (wdx2*wdy2 - wdxy*wdxy)*wdtz)

    del wdtx, wdty, wdtz
    del determinant


    ### Eigenvalues for Reliability ############################################
    # solve det(A^T w A - lamda I) = 0
    # (A' w A) = [a=wdx2 b=wdxy c=wdxz ; d=wdxy e=wdy2 f=wdyz ; g=wdxz h=wdyz i=wdz2]
    # det(A^T w A - lambda I) = (a-lambda)(e-lambda)(i-lambda) + 2*b*c*f - c^2(e-lambda) - b^2(i-lamda) - f^2(a-lambda)

    # Solve for the eigenvalues
    w = np.array([[wdx2, wdxy, wdxz],[wdxy, wdy2, wdyz],[wdxz, wdyz, wdz2]])
    del wdx2, wdxy, wdxz, wdy2, wdyz, wdz2
    w = np.moveaxis(w,[0,1],[-1,-2])
    w = w.astype(np.complex64) # Allow for complex eignenvalues
    rel = np.linalg.eigvals(w)
    rel = np.real(np.amin(rel,axis=-1))

    ### Return Outputs #########################################################
    return vx, vy, vz, rel
