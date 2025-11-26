# -*- coding: utf-8 -*-
"""
Helper functions for the 4D-STEM script
"""
import numpy as np
import pickle
from tqdm import tqdm
from scipy.special import erf
import matplotlib.patches as patches
from scipy import optimize
from scipy import ndimage,linalg 
import time
import matplotlib.pyplot as plt
from IPython.display import display
import warnings
import ipywidgets as widgets
from matplotlib.widgets import RectangleSelector,Button
    
def disp2img(img1,img2,figsize=(10,5)):
    '''
    Display real & diffraction space images (static, non-interactive)
    Parameters:
    ----------
    img1 : 2D ndarray
        Image to be shown on the left of the figure.
    img2 : 2D ndarray
        Image to be shown on the right of the figure.
    rcoord : TYPE
        DESCRIPTION.
    dcoord : TYPE
        DESCRIPTION.

    Returns ax1 (the leftmost subplot), ax2 (the rightmost subplot).

    '''
    fig=plt.figure(figsize=figsize);ax1=fig.add_subplot(1,2,1);
    ax1.imshow(img1,cmap='gray');ax1.axis('off')
    ax2=fig.add_subplot(1,2,2);
    ax2.imshow(img2,cmap='gray');ax2.axis('off')
    return ax1,ax2

def statDisp(img1,rcoord,dcoord):
    ax1,ax2=disp2img(img1[:,:,rcoord[1],rcoord[0]],img1[dcoord[1],dcoord[0],:,:]**0.1);
    ax1.set_title('Real Space (%d,%d)'%(dcoord[0],dcoord[1]))
    ax1.scatter(dcoord[0],dcoord[1],s=5,c='red',marker='o');
    ax2.scatter(rcoord[0],rcoord[1],s=5,c='red',marker='o');
    ax2.set_title('Diffraction Space (%d,%d)'%(rcoord[0],rcoord[1]))
    return ax1,ax2

def ewpc2D(data,useWindow=True,minlog=0.1,bg_sigma=20):
    '''
    converts dp slice into ewpc
    %ewpc calculates the EWPC transform fft(log(data)) for 2d or 4d data
    %   input:
    %       data -- 2d or 4d scanning electron diffraction data, ordered (kx,
    %               ky, x, y)
    %       useWindow -- logical indicating whether or not to apply hann window
    %               before FFT.  Default is 1.  The window is useful to prevent
    %               FFT artifacts from non-periodicity.  This is especially
    %               important it the diffraction patterns have significant
    %               intensity at their edges.
    '''
    N_kx,N_ky=data.shape
    minval=np.min(data)
    if useWindow:#window to prevent artifacts caused by non-periodic boundaries
        win=np.outer(np.hanning(N_kx),np.hanning(N_ky))
    else:
        win=np.ones((N_kx,N_ky))
    logdp=np.log(data-minval+minlog)
    cep=np.abs(np.fft.fftshift(np.fft.fft2(logdp*win)))
    return cep


def ConstrainedFun(x,func,win1,win2,outval_const=None):
    '''
    %adds constraint to objective function fun, which is assumed to be always
    %negative, by adding a positive "cone of shame" outside the specified
    %window
    '''
    if x[0]<win1[0] or x[0]>win1[1] or x[1]<win2[0] or x[1]>win2[1]:
        cent=[np.mean(win1),np.mean(win2)]
        if outval_const is None:
            y=np.sqrt((x[0]-cent[0])**2+(x[1]-cent[1])**2)
        else:
            y=outval_const
    else:
        y = func(x)
    return y

# def window2(N,M,w_func):
# # %Makes a 2D window function for FFT
#     wc=window(w_func,N)
#     wr=window(w_func,M)
#     [maskr,maskc]=np.meshgrid(wr,wc)
#     return maskr*maskc


def calculateSpotMapVectors(spotMaps,center=[62,62]):
    '''
    Calculates vector components, length, and angle assuming the detector 
    has 124 pixels, i.e. zero=(63,63). These are added to the spotMaps struct.

    Parameters
    ----------
    spotMaps : TYPE
        DESCRIPTION.
    center : TYPE, optional
        DESCRIPTION. The default is [62,62].

    Returns
    -------
    None.

    '''

    numSpots = len(spotMaps['Q1map'])
    spotMaps_updated=spotMaps.copy()
    spotMaps_updated['VectorX1']=np.zeros_like(spotMaps['Q1map'])
    spotMaps_updated['VectorX2']=np.zeros_like(spotMaps['Q1map'])
    spotMaps_updated['VectorLength']=np.zeros_like(spotMaps['Q1map'])
    spotMaps_updated['VectorAngle']=np.zeros_like(spotMaps['Q1map'])    
    for i in range(numSpots):
        x1map=spotMaps['Q1map'][i]
        x1map=x1map-center[0]
        x2map=spotMaps['Q2map'][i]
        x2map=x2map-center[1]
        spotMaps_updated['VectorX1'][i]=x1map
        spotMaps_updated['VectorX2'][i]=x2map
        spotMaps_updated['VectorLength'][i]=np.sqrt(x1map**2+x2map**2)
        spotMaps_updated['VectorAngle'][i]=np.arctan2(x1map,x2map)
    return spotMaps_updated  

def cft2(f,q1,q2,zeroCentered=0):
    '''
    Calculates the continuous fourier transform

    %cft2 2D continuous Fourier tranform of f evaluated at point q1, q2
    %   inputs:
    %       f -- the 2D array the fourier transform is calculated from
    %       q1,q2 -- indices where the transform is to be evaluated, following
    %              the same convention as fft2.  q1,q2 can be non-integers.
    %       zeroCentered -- boolean indicating the q index corresponding to
    %                       zero: 0 - default, zero is at index 1,1 (same as
    %                                 fft2(f))
    %                             1 - zero is at the image center,
    %                                 corresponding to fftshift(fft2(f))
    %   outputs:
    %       F -- value of the fourier transform of f at q1,q2.  This is a complex
    %            number, rather than an array.
    %
    %This function is part of the PC-STEM Package by Elliot Padgett in the 
    %Muller Group at Cornell University.  Last updated by Megan Holtz for speed July 27, 2020.
    '''
    (m,n)=f.shape
    jgr = np.arange(m)
    kgr = np.arange(n)

    if zeroCentered:
        q1=q1+m/2
        q2=q2+n/2

    F = np.sum(f*np.outer(np.exp(-2*np.pi*1j*jgr*q1/m),np.exp(-2*np.pi*1j*kgr*q2/n)))

    return F

def cft2_grad(f,q1,q2,zeroCentered=0):
    '''
    Calculates the gradient of the continuous fourier transform

    %cft2 2D continuous Fourier tranform of f evaluated at point q1, q2
    %   inputs:
    %       f -- the 2D array the fourier transform is calculated from
    %       q1,q2 -- indices where the transform is to be evaluated, following
    %              the same convention as fft2.  q1,q2 can be non-integers.
    %       zeroCentered -- boolean indicating the q index corresponding to
    %                       zero: 0 - default, zero is at index 1,1 (same as
    %                                 fft2(f))
    %                             1 - zero is at the image center,
    %                                 corresponding to fftshift(fft2(f))
    %   outputs:
    %       F_q1,F_q2 - projections of the gradient of the fourier transform of f at q1,q2.  This is a complex
    %            number, rather than an array.

    '''
    (m,n)=f.shape
    jgr = np.arange(m)
    kgr = np.arange(n)

    if zeroCentered:
        q1=q1+m/2
        q2=q2+n/2
    mult_q1=(-2)*np.pi*1j*jgr/m
    mult_q2=(-2)*np.pi*1j*kgr/n

    a1=np.exp(mult_q1*q1)
    a2=np.exp(mult_q2*q2)
    product=f*np.outer(a1,a2)
    b1=np.sum(product)
    b2=np.conj(b1)
    mult_q22,mult_q11=np.meshgrid(mult_q2,mult_q1)
    c1=np.sum(mult_q11*product)
    c2=np.sum(mult_q22*product)
    F_q1=c1*b2+b1*np.conj(c1)
    F_q2=c2*b2+b1*np.conj(c2)
#    F_q1 = np.sum(f*np.outer(a1*(-2)*np.pi*1j*jgr/m,a2))*np.sum(f*np.outer(a1,a2))+np.sum(f*np.outer(np.exp(-2*np.pi*1j*jgr*q1/m),np.exp(-2*np.pi*1j*kgr*q2/n)))*np.sum(f*np.outer(np.exp(2*np.pi*1j*jgr*q1/m)*2*np.pi*1j*jgr/m,np.exp(2*np.pi*1j*kgr*q2/n)))
#    F_q2 = np.sum(f*np.outer(np.exp(-2*np.pi*1j*jgr*q1/m),np.exp(-2*np.pi*1j*kgr*q2/n)*(-2)*np.pi*1j*kgr/n))*np.sum(f*np.outer(np.exp(2*np.pi*1j*jgr*q1/m),np.exp(2*np.pi*1j*kgr*q2/n)))+np.sum(f*np.outer(np.exp(-2*np.pi*1j*jgr*q1/m),np.exp(-2*np.pi*1j*kgr*q2/n)))*np.sum(f*np.outer(np.exp(2*np.pi*1j*jgr*q1/m),np.exp(2*np.pi*1j*kgr*q2/n)*2*np.pi*1j*kgr/n))

    return np.array([F_q1,F_q2])


def central_beam_mask(dp,bright_disk_radius=5,erf_sharpness=5):

    '''
    Generates a mask that blocks the central region. (use for ewpc patterns)

    :Parameters:
    dpShape : ndarray of integers. [x1,x2,k1,k2] or [k1,k2]
        shape of the diffraction space.
    bright_disk_radius : float, optional
        Approximately the pixel radius of the diffraction/EWPC space area 
        to be blocked/reduced.
    erf_sharpness : float, optional
        Larger values result in sharper transitions across the mask.
        (i.e. like a step function rather than a gradient change).
        High value of erf_sharpness (typically 5) works like a beam blocker.

    :Return: 
        bdisk_filter : ndarray
            A mask that blocks the central region.
    '''    
    xcols=dp.shape[-2]
    yrows=dp.shape[-1]
    kx = np.arange(-xcols, xcols, 2)/2
    ky = np.arange(-yrows, yrows, 2)/2
    ky,kx = np.meshgrid(ky, kx)
    dist = np.hypot(kx, ky)
    bdisk_filter = erf((dist-bright_disk_radius)*erf_sharpness)/2 - erf((dist+bright_disk_radius)*erf_sharpness)/2 + 1
    return bdisk_filter

  
# def load_raw_to_dp(fname , yrows , xcols ):
    # with open(fname, 'rb') as file:
        # dp = np.fromfile(file, np.float32)
                
    # sqpix = dp.size/yrows/xcols
    # pix = int(sqpix**(0.5))
    
    # dp = np.reshape(dp, (pix, pix, yrows, xcols), order = 'C')
    # if yrows>xcols:
        # dp = dp[:,:,1:xcols+1, :]
        # dp[:,:,0,:]=0
        # dp[:,:,-1,:]=0
        # dp = np.nan_to_num(dp)
        # dp[dp <= 20] = 1e-10
    
    # dp = dp[:,:,1:-3,2:-2]
    # dp = np.flip(dp, 2)
    
    # return dp #edited; before: return dp

def load_raw_to_dp(fname , Nx1, Nx2, Nk1, Nk2, flipx = False, flipy = False, transpose = False, EMPAD1 = True ):
    '''
    Loads data from a raw file and performs transpose/flip transformations
    
    :Parameters:
        fname : str
            File name of the 4D-STEM data.
        Nx1, Nx2, Nk1, Nk2 : int
            Dimensions of the 4D-STEM raw file
        flipx, flipy, transpose : boolean
            choose whether the diffraction space frames need to be flipped or transposed
        EMPAD1 : boolean
            set to True if data is collected on the EMPAD 1 detector, removes the metadata from the raw file
            
    :Return: 
        dp : ndarray (4D)
            4D-STEM dataset after removing metadata and performing flips/transforms 
    '''
    with open(fname, 'rb') as file:
        dp = np.fromfile(file, np.float32)
                
    dp = np.reshape(dp, (Nx1, Nx2, Nk1, Nk2), order = 'C')
    if (EMPAD1):
        dp = dp[:,:,2:-4, 2:-2]    
    
    dp = np.nan_to_num(dp)

    if(transpose):
        dp = np.swapaxes(dp, 2, 3)
    if(flipx):
        dp = np.flip(dp,2)
    if(flipy):
        dp = np.flip(dp,3)
    
    return dp

def saturate_array(masked_array,mask,saturation_lims):
    '''
    Parameters
    ----------
    masked_array : array of type numpy.ma.masked_array
    mask : boolean numpy array, has to have same size as the masked_array. 
            True values indicate indeces that won't be included in saturation. Useful if NaNs are present.
    saturation_lims : list with two numbers, representing upper and lower limit in percentile

    Returns
    -------
    masked_array : array 
        Modified array with values outside the saturation limits set to the minimum or maximum value

    ''' 
    [min_val,max_val]=np.percentile(masked_array[np.logical_not(mask)],saturation_lims)
    binary_mask=np.logical_and(masked_array>max_val,np.logical_not(mask))
    masked_array[binary_mask]=max_val
    binary_mask=np.logical_and(masked_array<min_val,np.logical_not(mask))
    masked_array[binary_mask]=min_val
    return masked_array

def convert_dp_to_ewpc(dp,flatten_center=False,bright_disk_radius=5,erf_sharpness=5):
    '''
    Applies cepstral transform to the data. 

    :Parameters: 
        dp : ndarray of shape (x1,x2,k1,k2)
            4D-STEM dataset.
        flatten_center : Boolean, optional
            If set to True, decreases the values of zero-order diffracted beam,
            which is assumed to be at the center of the diffraction pattern, by
            applying a mask shaped like an error function.
        bright_disk_radius : float, optional
            Approximately the pixel radius of the diffraction space area 
            to be blocked/reduced when flatten_center=True.
        erf_sharpness : float, optional (applied when flatten_center = True.)
            Larger values result in sharper transitions across the mask.
            (i.e. like a step function rather than a gradient change).
            High value of erf_sharpness (typically 5) works like a beam blocker.

    :Return: 
        cep : ndarray
            Cepstral transformed 4D dataset.
    ''' 
    pix, pix, yrows, xcols = dp.shape
    cep=np.zeros_like(dp)
    kx = np.arange(-xcols, xcols, 2)/2
    kx,ky = np.meshgrid(kx, kx)
    dist = np.hypot(kx, ky)
    if flatten_center:
      bdisk_filter = erf((dist-bright_disk_radius)*erf_sharpness)/2 - erf((dist+bright_disk_radius)*erf_sharpness)/2 + 1
    else:
      bdisk_filter= np.ones((yrows,xcols))
    
    for i in tqdm(range(pix)):
      for j in range(pix):
          current_slice=dp[i,j,:,:].copy()
          current_slice[current_slice<=0]=current_slice[current_slice>0].min()
          curr_cep = np.log10(current_slice)
          curr_cep=curr_cep*np.outer(np.hanning(yrows),np.hanning(xcols))
          curr_cep = np.fft.fft2(curr_cep)
          curr_cep = np.fft.fftshift(curr_cep, (-1, -2))
          cep[i,j] = np.abs(curr_cep)*bdisk_filter
    return cep

def create_haadf_mask(array_shape,radii):
    '''
    Creates an ADF mask for an image of size "array_shape."
    
    :Parameters:
        array_shape : tuple of integers. (k1,k2)
            The shape of the diffraction space.
        radii : 1D array. [r0,r1]
            Inner(r0) and outer(r1) radii of the virtual ADF detector.

    :Return: 
        haadf_mask : 2D array
        ADF mask showing the virtual detector area
    ''' 
    [r0,r1]=radii
    center=[array_shape[-2]/2,array_shape[-1]/2]
    kx = np.arange(array_shape[-1])-int(center[-1])
    ky = np.arange(array_shape[-2])-int(center[-2])
    kx,ky = np.meshgrid(kx,ky)
    kdist = (kx**2.0 + ky**2.0)**(1.0/2)
    haadf_mask = np.array(kdist <= r1, int)*np.array(kdist >= r0, int)
    return haadf_mask  

def disp_haadf(data4d,radii,*args):
    '''
    Displays an ADF image from averaged diffraction patterns of a 4D data.

    :Parameters:
        data4d : ndarray of shape (x1,x2,k1,k2)
            4D-STEM dataset.
        radii : 1D array. [r0,r1]
            Inner(r0) and outer(r1) radii of the virtual ADF detector.

    :Return: None.
    ''' 
    dim =data4d.shape #N_x1,N_x2,N_k1,N_k2
    try:
        i=args[0];j=args[1];
    except:
        print("real space position not specified correctly. Using default values.")
        i=int(dim[0]/2); j=int(dim[0]/2);
    haadf_mask=create_haadf_mask((dim[2],dim[3]),radii)
    haadf=np.mean(data4d*haadf_mask,axis=(-2,-1))
    ratio_array=np.log(np.mean(data4d, axis = (0,1)))#data4d[i,j,:,:])
    haadf_bndry=np.logical_xor(ndimage.binary_dilation(haadf_mask),haadf_mask)

    normalized_ratio_array=(ratio_array-ratio_array.min())/(ratio_array.max()-ratio_array.min())
    img=plt.cm.gray(normalized_ratio_array)#use grayscale colormap

    img[haadf_bndry]=[1,0,0,1] #make the boundary red
    
    fig=plt.figure();ax1=fig.add_subplot(1,2,1);
    ax1.imshow(haadf,cmap='gray');ax1.axis('off')
    ax2=fig.add_subplot(1,2,2);
    ax2.imshow(img,cmap='gray');ax2.axis('off')
    ax1.set_title('Image'); ax2.set_title('Mean Pattern')

def show_roi(ewpc,roi,wins):
    '''
    Shows the Region of Interest(ROI) where EWPC map will be calculated.

    :Parameters:
        ewpc : ndarray. [r1,r2,c1,c2]
            EWPC transformed dataset; Output from "convert_dp_to_ewpc." c:cepstral space.
        roi : 1D array. [x0,x1,y1,y2]
            x and y coordinates of the roi in the real space. 
            Choose one from "rois" list generated by "browser_with_peak_selection()."
        wins : ndarray of shape (n,4)
            Array of n windows enclosing specific EWPC peaks selected by the user.

    :Return: None.
    ''' 
    win_mask=np.zeros((ewpc.shape[2],ewpc.shape[3])).astype('bool')
    for i in range(len(wins)):
        win_mask[wins[i,2]:wins[i,3],wins[i,0]:wins[i,1]]=True
    cep_df=np.sum(ewpc*win_mask,axis=(-2,-1))
    ax1,ax2=disp2img(cep_df,cep_df[roi[2]:roi[3]+1,roi[0]:roi[1]+1])
    ax1.add_patch(patches.Rectangle((roi[0],roi[2]),roi[1]+1-roi[0],roi[3]+1-roi[2],linewidth=1,edgecolor='r',facecolor='none'))
    
def show_wins(data4d,wins,roi):
    '''
    Shows selected EWPC peaks in red boxes.

    :Parameters:
        data4d : ndarray of shape (x1,x2,k1,k2)
            4D-STEM dataset.
        wins : ndarray of shape (n,4)
            Array of n windows enclosing specific EWPC peaks selected by the user.
        roi : 1D array. [x0,x1,y1,y2]
            x and y coordinates of the roi in the real space. 
            Choose one from "rois" list generated by "browser_with_peak_selection()."
            x1_cropped ~= y2-y1; x2_cropped ~= x2-x1.
            
    :Returns:
        data4d_roi : 4D array of shape (x1_cropped,x2_cropped,k1,k2)
            4d dataset cropped to the real space region of interest
        ewpc_img : ndarray of size (k1,k2)
            Normalized cepstral transformation of a mean diffraction pattern
            with the zero-order peak blocked.
    '''     
    data4d_roi=data4d[roi[2]:roi[3]+1,roi[0]:roi[1]+1,:,:].copy()

    
    (rx,ry,sx,sy)=data4d_roi.shape
    dp_mean=np.mean(data4d_roi, axis=(0,1))
   
    ewpc_img=ewpc2D(dp_mean)*central_beam_mask(dp_mean)
    ewpc_img=(ewpc_img-ewpc_img.min())/(ewpc_img.max()-ewpc_img.min())
    ax1,ax2=disp2img(np.log( dp_mean * 1e5 + 1.0 ),ewpc_img)
    
    for j in range(len(wins)):
        win=patches.Rectangle((wins[j,0],wins[j,2]),wins[j,1]-wins[j,0]+1,wins[j,3]+1-wins[j,2],linewidth=1,edgecolor='r',facecolor='none')
        ax2.add_patch(win)
    
    ax1.set_title('log(Average DP)');ax2.set_title('selected EWPC peaks')
    
    return data4d_roi,ewpc_img

def create_spotList(wins):
    '''
    Store the windows positions in spotlist dictionary

    :Parameters:
        wins : ndarray of shape (n,4)
            Array of n windows enclosing specific EWPC peaks selected by the user.

    :Return: 
        spotList : dictionary
            For each selected cepstral spot, stores the extent of the window around the spot for peak-finding 
            

    ''' 
        ### store the windows positions in spotlist dictionary
    spotList={}
    spotList['spotRangeQ1']=[]
    spotList['spotRangeQ2']=[]
    for i in range(len(wins)):
        spotList['spotRangeQ1'].append([wins[i][2],wins[i][3]]) #check
        spotList['spotRangeQ2'].append([wins[i][0],wins[i][1]])
    return spotList

def get_spotMaps(data4d_roi,wins,valid=None,tol=1e-4,method='Nelder-Mead',jac='numerical'):
    '''
    Calculates spot positions (maximum within windows area) 

    :Parameters:
        data4d_roi : 4-dimensional diffraction data (N_x1,N_x2,N_k1,N_k2)
        wins : ndarray of shape (n,4)
            Array of n windows enclosing specific EWPC peaks selected by the user.
        valid: boolean ndarray (N_x1,N_x2), 
            which masks the region of interest, i.e. specific grain. If None, the whole FOV will be processed
        tol: float, tolerance level of the computation precision. Default is 1e-4
        method: minimization optimization method from scipy.optimize. Default is 'Nelder-Mead'
        jac: numerical(default) or analytical calculation of the function gradient
    :Return: 
        spotMaps : dictionary
            Contains arrays with the EWPC peak location map in pixels (Q1map, Q2map), 
            vector components of the EWPC peaks from the center (VectorX1, Vector X2), 
            vector length (VectorLength) and angle (VectorAngle), region of interest for strain mapping (roi),
            and windows with the EWPC peak locations (wins). 

    '''     
    (N_x1,N_x2,N_k1,N_k2)=data4d_roi.shape
    q1range=np.arange(N_k1)
    q2range=np.arange(N_k2)
    [Q2,Q1]=np.meshgrid(q2range,q1range)
    ###hann window 
    win=np.outer(np.hanning(N_k1),np.hanning(N_k2))
    spotList=create_spotList(wins)
    ### create the spotMaps dictionary, where the peak locations will be saved
    spotMaps={}; spotMaps['Q1map']=[]; spotMaps['Q2map']=[]
    for s in range(len(wins)):#edit to allow multiple windows
        spotMaps['Q1map'].append(np.zeros(data4d_roi.shape[0:2]))
        spotMaps['Q1map'][s][:,:]= np.nan; #### first set all values to nan 
        spotMaps['Q2map'].append(np.zeros(data4d_roi.shape[0:2]))
        spotMaps['Q2map'][s][:,:]= np.nan   
    if np.sum(valid==None):
        valid = np.ones([data4d_roi.shape[0],data4d_roi.shape[1]]) 
    valid = valid.astype(bool)
    mask_pos=np.where(valid)
    t1=time.time()
    ### now go through the points within the valid mask and calculate the peak locations
    for num_pos in tqdm(range(valid.sum())):
        j=mask_pos[0][num_pos]
        k=mask_pos[1][num_pos]
        CBED = data4d_roi[j,k,:,:]
        minval=CBED.min()
        EWPC = ewpc2D(CBED)
        #define continuous Fourier transform
        PeakFun = lambda x: -np.abs(cft2(win*np.log(CBED-minval+0.1),x[0],x[1],1))**2
        GradFun = lambda x: -cft2_grad(win*np.log(CBED-minval+0.1),x[0],x[1],1).real
        #iterate through spots of interest
        for s in range(len(spotList['spotRangeQ1'])):
            #Get spot locations from input struct
            spot_ROI_q1 = spotList['spotRangeQ1'][s]
            spot_ROI_q2 = spotList['spotRangeQ2'][s]
            spotNeighborhood = EWPC[spot_ROI_q1[0]:spot_ROI_q1[1]+1,spot_ROI_q2[0]:spot_ROI_q2[1]+1]
            #Find rough location of maximum peak 
            maxidx= np.unravel_index(np.argmax(spotNeighborhood),spotNeighborhood.shape)
            Q1_roi = Q1[spot_ROI_q1[0]:spot_ROI_q1[1]+1,spot_ROI_q2[0]:spot_ROI_q2[1]+1]
            Q2_roi = Q2[spot_ROI_q1[0]:spot_ROI_q1[1]+1,spot_ROI_q2[0]:spot_ROI_q2[1]+1]
            Q1max = Q1_roi[maxidx]
            Q2max = Q2_roi[maxidx]
            #Search for spot peak in continuous Fourier transform
            constrainedPeakFun = lambda x: ConstrainedFun(x,PeakFun,[spot_ROI_q1[0],spot_ROI_q1[-1]],[spot_ROI_q2[0],spot_ROI_q2[-1]])
            constrainedGradFun = lambda x: [ConstrainedFun(x,GradFun[0],[spot_ROI_q1[0],spot_ROI_q1[-1]],[spot_ROI_q2[0],spot_ROI_q2[-1]]),
                                            ConstrainedFun(x,GradFun[1],[spot_ROI_q1[0],spot_ROI_q1[-1]],[spot_ROI_q2[0],spot_ROI_q2[-1]])]
            if method=='Nelder-Mead' and jac=='numerical':
                peakQ = optimize.fmin(constrainedPeakFun,x0=np.array([Q1max,Q2max]),ftol=tol,xtol=tol,disp=False)
            elif method in ['L-BFGS-B','Powell','TNC'] and jac=='numerical':
                bnds=((spot_ROI_q1[0],spot_ROI_q1[1]+1),(spot_ROI_q2[0],spot_ROI_q2[1]+1))
                peakQ = optimize.minimize(PeakFun,x0=np.array([Q1max,Q2max]),method=method,bounds=bnds,tol=tol).x
            if method=='Nelder-Mead' and jac=='analytical':
                peakQ = optimize.fmin(constrainedPeakFun,x0=np.array([Q1max,Q2max]),ftol=tol,xtol=tol,disp=False,jac=constrainedGradFun)
            elif method in ['L-BFGS-B','Powell','TNC'] and jac=='analytical':
                bnds=((spot_ROI_q1[0],spot_ROI_q1[1]+1),(spot_ROI_q2[0],spot_ROI_q2[1]+1))
                peakQ = optimize.minimize(PeakFun,x0=np.array([Q1max,Q2max]),method=method,bounds=bnds,tol=tol,jac=GradFun).x

            # Assign in maps
            spotMaps['Q1map'][s][j,k] = peakQ[0]
            spotMaps['Q2map'][s][j,k] = peakQ[1]    
    t2=time.time()
    spotMaps_upd=calculateSpotMapVectors(spotMaps,center=[int(N_k1/2),int(N_k2/2)])
    print('Time spent: '+ "{:.0f}".format(t2-t1) + 's')
    return spotMaps_upd

def plotSpotMaps(wins,ewpc_img,spotMaps_upd,figureSize=(10,5),sat_lims=[0,100],pix_size=None,unit_label='pixels',cmap='RdBu_r',plot_ids=None):
    '''
    Plots a map of the vector length and angle of the EWPC peak positions

    :Parameters:
        wins : ndarray of shape (n,4)
            Array of n windows enclosing specific EWPC peaks selected by the user.
        ewpc_img : ndarray of size (k1,k2)
            Normalized cepstral transformation of a mean diffraction pattern
            with the zero-order peak blocked.
        spotMaps : dictionary (typically the output from the function get_SpotMaps)
            Contains arrays with the EWPC peak location map in pixels (Q1map, Q2map), 
            vector components of the EWPC peaks from the center (VectorX1, Vector X2), 
            vector length (VectorLength) and angle (VectorAngle), region of interest for strain mapping (roi),
            and windows with the EWPC peak locations (wins). 
        figureSize: tuple
        sat_lims : list of length 2
            Lower and upper percentile limits for the display of maps, passed as input to fuction saturate_array
        pix_size : float
            Pixel size calibration
        unit_label : str
            Pixel size calibration units, default is pixel units
        cmap : str
            Colormap for the EWPC peak vector length and angle maps
        plot_ids : list
            Indices of the EWPC spots for which the maps will be plotted


    :Return: None.
         
    '''   
    
    fig = plt.figure(figsize=figureSize,constrained_layout=True)
    j=len(spotMaps_upd['VectorLength'])
    if pix_size==None:
        pix_size=1
    if plot_ids==None:
        plot_ids=np.arange(j)
    for i in range(j):
        ax1=fig.add_subplot(j,3,3*i+1);
        ax2=fig.add_subplot(j,3,3*i+2);ax3=fig.add_subplot(j,3,3*i+3)
        
        im1 = ax1.imshow(ewpc_img)
        win=patches.Rectangle((wins[i,0],wins[i,2]),wins[i,1]-wins[i,0],wins[i,3]-wins[i,2],linewidth=1,edgecolor='r',facecolor='none')
        ax1.add_patch(win); ax1.set_title('EWPC peak #'+str(plot_ids[i]));ax1.axis('off')

        array=spotMaps_upd['VectorLength'][i].copy()
        mask=np.isnan(array)
        mask_pos=np.where(np.logical_not(mask))
        a1=mask_pos[0].min()
        a2=mask_pos[0].max()
        b1=mask_pos[1].min()
        b2=mask_pos[1].max()
        
        if sat_lims!=[0,100]:
            array=saturate_array(array,mask,sat_lims)
        array=array[a1:a2,b1:b2]*pix_size
        vmin = np.nanmean(array) - np.nanstd(array)
        vmax =  np.nanmean(array) + np.nanstd(array)
        im2=ax2.imshow(array, vmin=vmin, vmax=vmax,cmap=cmap)
        cb2 = fig.colorbar(im2, ax=ax2, label = unit_label)
        ax2.set_title('Vector Length');ax2.axis('off')
        array=spotMaps_upd['VectorAngle'][i].copy()
        if sat_lims!=[0,100]:
            array=saturate_array(array,mask,sat_lims)
        array=180*array[a1:a2,b1:b2]/np.pi
        vmin = np.nanmean(array) - np.nanstd(array)
        vmax =  np.nanmean(array) + np.nanstd(array)
        im3=ax3.imshow(array, vmin=vmin, vmax=vmax,cmap=cmap) 
        fig.colorbar(im3, ax=ax3, label = 'deg') 
        ax3.set_title('Vector Angle');ax3.axis('off')
    fig.set_constrained_layout_pads(hspace=0.2, wspace=0.2)


def makeRelativeSpotReference( spotMaps_upd, ref_roi ):

    '''
    Calculates the mean position of the EWPC peaks in the reference region

    :Parameters:
        spotMaps : dictionary (typically the output from the function get_SpotMaps)
            Contains arrays with the EWPC peak location map in pixels (Q1map, Q2map), 
            vector components of the EWPC peaks from the center (VectorX1, Vector X2), 
            vector length (VectorLength) and angle (VectorAngle), region of interest for strain mapping (roi),
            and windows with the EWPC peak locations (wins). 
        ref_roi: list of form [x_i, x_f, y_i, y_f] where the list entries are integers
            Pixel coordinates defining the reference region in real space
            
    :Return: 
        spotRef : dictionary
            contains indices to identify the EWPC spots and the mean position of the EWPC peak
            positions in the reference ROI      
         
    ''' 
    spotRef = {'id':[], 'point': []}

    num = len(spotMaps_upd['Q1map'])
    
    for i in range(num):
        spotRef["id"].append(i)
        ref1 = np.nanmean(spotMaps_upd['VectorX1'][i][ref_roi[0]:ref_roi[1], ref_roi[2]:ref_roi[3]])
        ref2 = np.nanmean(spotMaps_upd['VectorX2'][i][ref_roi[0]:ref_roi[1], ref_roi[2]:ref_roi[3]])
        spotRef["point"].append( np.array([ref1, ref2]) )

    return spotRef

def makeRelativeSpotReference_median( spotMaps_upd, ref_roi ):

    '''
    Calculates the median position of the EWPC peaks in the reference region

    :Parameters:
        spotMaps : dictionary (typically the output from the function get_SpotMaps)
            Contains arrays with the EWPC peak location map in pixels (Q1map, Q2map), 
            vector components of the EWPC peaks from the center (VectorX1, Vector X2), 
            vector length (VectorLength) and angle (VectorAngle), region of interest for strain mapping (roi),
            and windows with the EWPC peak locations (wins). 
        ref_roi: list of form [x_i, x_f, y_i, y_f] where the list entries are integers
            Pixel coordinates defining the reference region in real space
            
    :Return: 
        spotRef : dictionary
            contains indices to identify the EWPC spots and the median position of the EWPC peak
            positions in the reference ROI 
         
    '''  
    spotRef = {'id':[], 'point': []}

    num = len(spotMaps_upd['Q1map'])
    
    for i in range(num):
        spotRef["id"].append(i)
        ref1 = np.nanmedian(spotMaps_upd['VectorX1'][i][ref_roi[0]:ref_roi[1], ref_roi[2]:ref_roi[3]])
        ref2 = np.nanmedian(spotMaps_upd['VectorX2'][i][ref_roi[0]:ref_roi[1], ref_roi[2]:ref_roi[3]])
        spotRef["point"].append( np.array([ref1, ref2]) )

    return spotRef


def calculateStrainMap(spotMaps, spotRef, latticeCoords=1, image_basis=0):
    '''
    Calculates the strain map - the  affine transformation relating the reference EWPC peaks to
    the EWPC peaks at each probe position is calculated and then decomposed into a strain matrix
    and a rotation matrix 

    :Parameters:
        spotMaps : dictionary (typically the output from the function get_SpotMaps)
            Contains arrays with the EWPC peak location map in pixels (Q1map, Q2map), 
            vector components of the EWPC peaks from the center (VectorX1, Vector X2), 
            vector length (VectorLength) and angle (VectorAngle), region of interest for strain mapping (roi),
            and windows with the EWPC peak locations (wins). 
        spotRef : dictionary
            contains indices to identify the EWPC spots and the mean position of the EWPC peak
            positions in the reference ROI
        latticeCoords : choice between left(0)/right(1) polar decomposition
            
    :Return: 
        strainComponents : dictionary
            contains the 3 independent elements of the 2d strain tensor and the rotation angle.
            For representation in a strain elipse form with pricipal axes, the eigenvectors and 
            eigenvalues of the strain tensor is also calculated. The larger (smaller) eigenvalue 
            represents the length of the major (minor) axis of the strain ellipse, and the "strainAngle" 
            stores the angle between the major and minor axis.
         
    '''
    [N_x1,N_x2] = spotMaps["Q1map"][0].shape
    
    StrainComponents = {'Eps11':np.zeros((N_x1, N_x2)), 'Eps22':np.zeros((N_x1, N_x2)), 'Eps12':np.zeros((N_x1, N_x2)), 'Theta':np.zeros((N_x1, N_x2)), 'majAx':np.zeros((N_x1, N_x2)), 'minAx':np.zeros((N_x1, N_x2)), 'strainAngle':np.zeros((N_x1, N_x2))}
    
    E = np.zeros((N_x1, N_x2, 2, 2))
    R = np.zeros((N_x1, N_x2, 2, 2))
    
    #prepare reference point list 
    num = len(spotRef['id'])
    for i in range(num):
        refPoints = np.array([ [0,0], spotRef['point'][0], spotRef['point'][1] ])
        refPoints = np.float32(refPoints)
    
    for j in range(N_x1):
        for k in range(N_x2):
            dataPoints = [[0,0]]
            
            for s in range(num):
                
                q1c = spotMaps["VectorX1"][s][j,k]
                q2c = spotMaps["VectorX2"][s][j,k]
                
                #include in list for tranformation calculation
                dataPoints.append([q1c, q2c])
                
                dataPoints_array = np.float32(np.array(dataPoints))
                
            if( np.sum(np.isnan(dataPoints_array)) ):
                StrainComponents["Eps11"][j,k]=np.nan
                StrainComponents["Eps22"][j,k] = np.nan
                StrainComponents["Eps12"][j,k] = np.nan
                StrainComponents["Theta"][j,k]=np.nan
                StrainComponents["majAx"][j,k] = np.nan
                StrainComponents["minAx"][j,k] = np.nan
                StrainComponents["strainAngle"][j,k] = np.nan                
                
            else:

                if(image_basis==0):       
                    dataPoints_array_trim = dataPoints_array[1:, :]
                    refPoints_trim = refPoints[1:,:]
                    M = np.matmul(dataPoints_array_trim, np.linalg.inv(refPoints_trim) )
                
                else:
                    M = cv2.getAffineTransform(refPoints, dataPoints_array)
                    M =  M[:,0:2]
                
                r, u = linalg.polar(M, 'right') # M = ru
                r, v = linalg.polar(M, 'left') # M = vr
                
                if latticeCoords==1:
                    strain_mat = u - np.eye(2)
                else:
                    strain_mat = v - np.eye(2)
                    
                E[j,k,:,:] = strain_mat
                R[j,k,:,:] = r
                
                StrainComponents["Eps11"][j,k] = strain_mat[0,0]
                StrainComponents["Eps22"][j,k] = strain_mat[1,1]
                StrainComponents["Eps12"][j,k] = strain_mat[0,1]
                StrainComponents["Theta"][j,k] = 180*np.arctan2( r[1,0], r[0,0] )/np.pi
                

                #strain ellipse parameters
                
                eigval, eigvec = np.linalg.eig(strain_mat)
                
                if( eigval[0] > eigval[1]):
                    StrainComponents["majAx"][j,k] = eigval[0]
                    StrainComponents["minAx"][j,k] = eigval[1]
                    StrainComponents["strainAngle"][j,k] = np.arctan2( eigvec[0,0], eigvec[1,0] )                    
                else:
                    StrainComponents["majAx"][j,k] = eigval[1]
                    StrainComponents["minAx"][j,k] = eigval[0]
                    StrainComponents["strainAngle"][j,k] = np.arctan2( eigvec[0,0], eigvec[1,0] )                    
    
    return StrainComponents

def plotStrainEllipse(StrainComponents,figureSize=(8,3)):
    '''
    Plots the major axis length, minor axis length and angle between major and minor axis of the strain ellipse

    :Parameters:
        strainComponents : dictionary
            contains the 3 independent elements of the 2d strain tensor and the rotation angle.
            For representation in a strain elipse form with pricipal axes, the eigenvectors and 
            eigenvalues of the strain tensor is also calculated. The larger (smaller) eigenvalue 
            represents the length of the major (minor) axis of the strain ellipse, and the "strainAngle" 
            stores the angle between the major and minor axis. 
        figureSize : tuple
            
    :Return: None.
         
    '''
    color = 'viridis'
    
    plt.figure(figsize=figureSize)
    plt.subplot(1,3,1)
    img1=plt.imshow(StrainComponents["majAx"], cmap=color)
    plt.colorbar(img1,shrink = 0.75)
    plt.gca().set_axis_off()
    plt.margins(0,0)
    plt.title("Major axis", fontsize=12)
    
    plt.subplot(1,3,2)
    plt.imshow(StrainComponents["minAx"], cmap=color)
    plt.colorbar(shrink = 0.75)
    plt.gca().set_axis_off()
    plt.margins(0,0)
    plt.title("Minor axis", fontsize=12)
    
    plt.subplot(1,3,3)
    plt.imshow(StrainComponents["strainAngle"], cmap=color)
    plt.colorbar(shrink = 0.75)
    plt.gca().set_axis_off()
    plt.margins(0,0)
    plt.title("Axis angle", fontsize=12)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.1)
    
    
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plotStrainTensor(StrainComponents,figureSize=(8,8), vrange_eps = [-5,5], vrange_theta = [-4,4], cmap='RdBu_r'):
    '''
    Plots the components of the strain tensor and the rotation angle

    :Parameters:
        strainComponents : dictionary
            contains the 3 independent elements of the 2d strain tensor and the rotation angle.
            For representation in a strain elipse form with pricipal axes, the eigenvectors and 
            eigenvalues of the strain tensor is also calculated. The larger (smaller) eigenvalue 
            represents the length of the major (minor) axis of the strain ellipse, and the "strainAngle" 
            stores the angle between the major and minor axis. 
        figureSize : tuple
        vrange_eps : list
            lower and upper limits for the strain map colorbar in percentage
        vrange_theta : list
            lower and upper limits for the rotation angle map colorbar in degrees
            
    :Return: None.
         
    '''    
    titles=["$\epsilon_{11} (\%)$","$\epsilon_{22} (\%)$","$\epsilon_{12} (\%)$","$\Theta $"]
    keys=["Eps11","Eps22","Eps12","Theta"]
    mask=np.isnan(StrainComponents["Eps11"])
    mask_pos=np.where(np.logical_not(mask))
    a1=mask_pos[0].min()
    a2=mask_pos[0].max()
    b1=mask_pos[1].min()
    b2=mask_pos[1].max()

    fig,axes=plt.subplots(2,2,figsize=figureSize)
    for i in range(4):
        array=StrainComponents[keys[i]].copy()
        if keys[i]!='Theta':
            array = array * 100
            vmin = vrange_eps[0]
            vmax = vrange_eps[1]
        else:
            vmin = vrange_theta[0]
            vmax = vrange_theta[1]

        ax_loc=np.unravel_index(i,(2,2))

        array=array[a1:a2,b1:b2]
        im=axes[ax_loc].imshow(array,cmap=cmap, vmin=vmin, vmax=vmax)
        axes[ax_loc].set_xticks([])
        axes[ax_loc].set_yticks([])        
        divider = make_axes_locatable(axes[ax_loc])
        cax = divider.append_axes("right", size="10%", pad=0.05)
        axes[ax_loc].set_title(titles[i])
        if keys[i]=='Theta':
            plt.colorbar(im,cax=cax,label='deg')
        else:
            plt.colorbar(im,cax=cax)
    plt.subplots_adjust(wspace=0.1, hspace=0.15)        

    

'''
    the following function is the browser with three windows (might be confusing to some people)
def browser_with_peak_selection(data4d,cmap='gray', wins=[],rois=[],half_width=8):
    rx,ry,kx,ky=np.shape(data4d)
    bf_img=data4d[:,:,int(kx/2),int(ky/2)]    
    fig=plt.figure(figsize=(10, 6))
    ax1=fig.add_axes([0.10,0.1,0.25,0.8])
    ax2=fig.add_axes([0.40,0.1,0.25,0.8])
    ax3=fig.add_axes([0.05,0.05,0.15,0.07])
    ax4=fig.add_axes([0.70,0.1,0.25,0.8]);ax4.axis('off')
    ax5=fig.add_axes([0.3,0.05,0.15,0.07]);ax5.axis('off')
    wins=[];rois=[]
    ax1.imshow(bf_img,cmap=cmap,origin='upper');ax1.axis('off')
    ax2.imshow(data4d[int(rx/2),int(ry/2),:,:],cmap=cmap,origin='upper')
    ax2.set_title('Cepstral/Diffraction space');ax2.axis('off')
    ax1.set_title('Real space (Dark Field Image)')
    ax4.set_title('Select peak for analysis')
    ax5.text(0.1,0.5,"Number of peaks saved:"+str(len(rois)),horizontalalignment='center',verticalalignment='center')
    def select_zoom(eclick,erelease):
        zoom_roi=np.array(add_selector.extents).astype('int')
        updated_r_img=np.mean(data4d[:,:,int(zoom_roi[2]):int(zoom_roi[3]),int(zoom_roi[0]):int(zoom_roi[1])],axis=(-2,-1))
        ax1.imshow(updated_r_img,cmap=cmap);ax1.axis('off')
        
    def onselect_function_real_space(eclick, erelease):
        real_roi = np.array(rect_selector.extents).astype('int')
        updated_k_img=np.mean(data4d[int(real_roi[2]):int(real_roi[3]),int(real_roi[0]):int(real_roi[1]),:,:],axis=(0,1))        
        ax2.imshow(np.log(updated_k_img),cmap=cmap);ax2.axis('off')
    
    def onselect_function_reciprocal_space(eclick, erelease):
        reciprocal_roi = np.array(reciprocal_rect_selector.extents).astype('int')
        updated_r_img=np.mean(data4d[:,:,int(reciprocal_roi[2]):int(reciprocal_roi[3]),int(reciprocal_roi[0]):int(reciprocal_roi[1])],axis=(-2,-1))
        ax1.imshow(updated_r_img,cmap=cmap);ax1.axis('off')
        real_roi = np.array(rect_selector.extents).astype('int')
        updated_k_img=np.mean(data4d[int(real_roi[2]):int(real_roi[3]),int(real_roi[0]):int(real_roi[1]),:,:],axis=(0,1))        
        ewpc_win=[int(0.5*(reciprocal_roi[0]+reciprocal_roi[1]))-half_width,int(0.5*(reciprocal_roi[0]+reciprocal_roi[1]))+half_width,int(0.5*(reciprocal_roi[2]+reciprocal_roi[3]))-half_width,int(0.5*(reciprocal_roi[2]+reciprocal_roi[3]))+half_width]
        ax4.imshow(np.log(updated_k_img)[int(ewpc_win[2]):int(ewpc_win[3]),int(ewpc_win[0]):int(ewpc_win[1])],extent=[ewpc_win[0],ewpc_win[1],ewpc_win[3],ewpc_win[2]],cmap=cmap)
        ax4.axis('off')
        
    def save_results(event):
        print('clicked')
        reciprocal_roi = np.array(reciprocal_rect_selector.extents).astype('int')
        zoom_roi=np.array(add_selector.extents).astype('int')
        real_roi = np.array(rect_selector.extents).astype('int')
        wins.append(zoom_roi)
        rois.append(real_roi);ax5.clear()
        ax5.text(0.1,0.5,"Number of rois saved:"+str(len(rois)),horizontalalignment='center',verticalalignment='center')
        ax5.axis('off')
    
    
    add_selector= RectangleSelector(ax4, select_zoom, button=[1], # drawtype='box',
                                      useblit=True,minspanx=20, minspany=20,spancoords='pixels',interactive=True)    
    rect_selector = RectangleSelector(ax1, onselect_function_real_space, button=[1], #drawtype='box',
                                      useblit=True ,minspanx=20, minspany=20,spancoords='pixels',interactive=True)
    reciprocal_rect_selector = RectangleSelector(ax2, onselect_function_reciprocal_space, button=[1], #drawtype='box',
                                      useblit=True,minspanx=20, minspany=20,spancoords='pixels',interactive=True)    
    save_results_button=Button(ax3, 'Save Results')
    save_results_button.on_clicked(save_results)
    return (rect_selector,reciprocal_rect_selector,add_selector,save_results_button),wins,rois
'''

def browser_with_peak_selection(data4d,cmap='gray', wins=[],rois=[]):
    '''
    Browser for navigating the 4D-STEM dataset and selecting real space ROI and EWPC spots for peak-finding

    :Parameters:
        data4d : 4-dimensional numpy array
            4D-STEM dataset
        cmap : string
            Colormap for use in the browser
            
    :Return: 
        rect_selector: object of RectangleSelector method in matplotlib widgets
            Stores information about mouse clicks and selections in the real space image
        reciprocal_rect_selector: object of RectangleSelector method in matplotlib widgets
            Stores information about mouse clicks and selections in the cepstral space image
        save_results_button :  object of Button method in matplotlib widgets
            Button to indicate that a selection of real space ROI and cepstral spot has been made
        wins : list of arrays
            Stores coordinates of region selected in cepstral space for peak finding
        rois : list of arrays
            Stores coordinates of real space ROI for strain mapping
         
    '''
    rx,ry,kx,ky=np.shape(data4d)
    bf_img=data4d[:,:,int(kx/2),int(ky/2)]
    fig=plt.figure(figsize=(10, 7))
    ax1=fig.add_axes([0,0.15,0.45,0.7])
    ax2=fig.add_axes([0.5,0.15,0.45,0.7])
    ax3=fig.add_axes([0.75,0.05,0.15,0.04])
    ax5=fig.add_axes([0.3,0,1,0.07]);ax5.axis('off')
    wins=[];rois=[]
    ax1.imshow(bf_img,cmap=cmap,origin='upper');ax1.axis('off')
    ax2.imshow(data4d[int(rx/2),int(ry/2),:,:],cmap=cmap,origin='upper')
    ax2.set_title('Cepstral/Diffraction space:\n2. Select 2 peaks for analysis\n click the save button after each selection');ax2.axis('off')
    ax1.set_title('Real space (Dark Field Image):\n1. Select ROI')
    ax5.text(0.1,0.5,"Number of peaks saved:"+str(len(rois)),horizontalalignment='center',verticalalignment='center')

    def onselect_function_real_space(eclick, erelease):
        real_roi = np.array(rect_selector.extents).astype('int')
        updated_k_img=np.mean(data4d[int(real_roi[2]):int(real_roi[3]),int(real_roi[0]):int(real_roi[1]),:,:],axis=(0,1))
        ax2.imshow(np.log(updated_k_img),cmap=cmap);ax2.axis('off')
    
    def onselect_function_reciprocal_space(eclick, erelease):
        reciprocal_roi = np.array(reciprocal_rect_selector.extents).astype('int')
        updated_r_img=np.mean(data4d[:,:,int(reciprocal_roi[2]):int(reciprocal_roi[3]),int(reciprocal_roi[0]):int(reciprocal_roi[1])],axis=(-2,-1))
        ax1.imshow(updated_r_img,cmap=cmap);ax1.axis('off')
        real_roi = np.array(rect_selector.extents).astype('int')
        updated_k_img=np.mean(data4d[int(real_roi[2]):int(real_roi[3]),int(real_roi[0]):int(real_roi[1]),:,:],axis=(0,1))

        
    def save_results(event):
        print('clicked')
        reciprocal_roi = np.array(reciprocal_rect_selector.extents).astype('int')
        real_roi = np.array(rect_selector.extents).astype('int')
        wins.append(reciprocal_roi)
        rois.append(real_roi);ax5.clear()
        ax5.text(0.1,0.5,"Number of rois saved:"+str(len(rois)),horizontalalignment='center',verticalalignment='center')
        ax5.axis('off')
    
    markerprops = dict(marker='o', markersize=1, markerfacecolor='white', markeredgecolor='black')
    rect_selector = RectangleSelector(ax1, onselect_function_real_space, button=[1], #drawtype='box',
                                      useblit=True ,minspanx=20, minspany=20,spancoords='pixels',interactive=True)
    reciprocal_rect_selector = RectangleSelector(ax2, onselect_function_reciprocal_space, button=[1], #drawtype='box',
                                      handle_props=markerprops,useblit=True,minspanx=0, minspany=0,spancoords='pixels',interactive=True)
    save_results_button=Button(ax3, 'Save Results')
    save_results_button.on_clicked(save_results)
    return (rect_selector,reciprocal_rect_selector,save_results_button),wins,rois

def browser(data4d,cmap='gray'):
    
    '''
    Browser for navigating the 4D-STEM dataset

    :Parameters:
        data4d : 4-dimensional numpy array
            4D-STEM dataset
        cmap : string
            Colormap for use in the browser
            
    :Return: 
        rect_selector: object of RectangleSelector method in matplotlib widgets
            Stores information about mouse clicks and selections in the real space image
        reciprocal_rect_selector: object of RectangleSelector method in matplotlib widgets
            Stores information about mouse clicks and selections in the reciprocal space image
         
    '''
    rx,ry,kx,ky=np.shape(data4d)
    bf_img=data4d[:,:,int(kx/2),int(ky/2)]    
    fig=plt.figure(figsize=(8, 5))
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    
    ax1.imshow(bf_img,cmap=cmap,origin='upper');ax1.axis('off')
    ax2.imshow(data4d[int(rx/2),int(ry/2),:,:],cmap=cmap,origin='upper')
    ax2.set_title('Cepstral/Diffraction space');ax2.axis('off')
    ax1.set_title('Real space (Dark Field Image)')
    
    def onselect_function_real_space(eclick, erelease):
        real_roi = np.array(rect_selector.extents).astype('int')
        updated_k_img=np.mean(data4d[int(real_roi[2]):int(real_roi[3]),int(real_roi[0]):int(real_roi[1]),:,:],axis=(0,1))        
        ax2.imshow(np.log(updated_k_img),cmap=cmap)
    
    def onselect_function_reciprocal_space(eclick, erelease):
        reciprocal_roi = np.array(reciprocal_rect_selector.extents).astype('int')
        updated_r_img=np.mean(data4d[:,:,int(reciprocal_roi[2]):int(reciprocal_roi[3]),int(reciprocal_roi[0]):int(reciprocal_roi[1])],axis=(-2,-1))
        ax1.imshow(updated_r_img,cmap=cmap)


    rect_selector = RectangleSelector(ax1, onselect_function_real_space, button=[1], #drawtype='box',
                                      useblit=True ,minspanx=1, minspany=1,spancoords='pixels',interactive=True)
    reciprocal_rect_selector = RectangleSelector(ax2, onselect_function_reciprocal_space, button=[1], #drawtype='box',
                                      useblit=True,minspanx=1, minspany=1,spancoords='pixels',interactive=True)    
    
    
    
    return (rect_selector,reciprocal_rect_selector), ax1


###PCA Decomposition

from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from sklearn.decomposition import PCA
import matplotlib as mpl
from matplotlib.cm import ScalarMappable as scm
from sklearn.cluster import KMeans
from matplotlib.patches import Rectangle as Rect

def pca_decomposition(ewpc,n_components,circular_mask,include_center=False,normalization=True):
    '''
    Perform Principal Component Analysis Decomposition
    
    :Parameters:
        ewpc : 4d ndarray
            Cepstral or diffraction data. 
        n_components: int
            number of components for decomposition
        circular_mask: boolean ndarray,
            mask used for removing the central spot in cepstral images.
            Generally, it can be any type of mask where one can omit/highlight the focus of pca decomposition on certain areas within diffraction or cepstral pattern.
        include_center : boolean
            set to True if circular mask is used, False if there is no masking
        normalization: boolean
            set to True if data needs to be normalized before PCA
    :Returns: 
        pca: PCA model
        scores : ndarray
            result of fitting 4D data into the pca model.
    '''    
    if include_center:
        array=ewpc.reshape((ewpc.shape[0]*ewpc.shape[1],ewpc.shape[2]*ewpc.shape[3]))
    else:
        array=flatten_with_circular_mask(ewpc, circular_mask)
    if normalization:
        norm_means=np.mean(array, axis=1)
        norm_stds=np.std(array,axis=1)
        print('normalization of the ewpc pattern')
        for i in tqdm(range(array.shape[1])):
            array[:,i]-=norm_means
            array[:,i]/=norm_stds

    pca = PCA(n_components)
    scores = pca.fit_transform(array)
    return pca,scores

def generate_false_color_image(images, first_index = 1, last_index = None):
    '''
    Function to generate false color image (function written by Michael Cao).
    
    :Parameters:
        images : stack of images
        first_index : int
           First index in the stack of images to use in the false color image
        last_index : int
            Last index in the stack of images to use in the false color image
        
    :Returns:
        false color image and color-corrected false color image
    '''       
    imgs = np.copy(images[first_index:last_index])
    num_images = imgs.shape[0]
    hues = (np.arange(0, num_images)/(110**0.5)) % 1.0

    colors = np.array([hsv_to_rgb((hue, 1.0, 255)) for hue in hues])
    imgs = imgs.T
    imgs -= np.mean(imgs, axis = (0,1))
    imgs /= np.max(imgs, axis = (0,1))

    imgs = imgs.T
    
    colored_imgs = np.zeros(imgs.shape+(3,))
    for i in range(num_images):
        for j in range(3):
            colored_imgs[i,:,:,j] = imgs[i]*colors[i,j]
            
    false_color_img = np.mean(colored_imgs, axis = 0)
    false_color_img -= np.min(false_color_img)
    false_color_img /= np.max(false_color_img)
    
    #Color correcting
    false_color_hsv = rgb_to_hsv(false_color_img)
    
    v_values = false_color_hsv[:,:,2]
    vmean = np.mean(v_values)
    vstd = np.std(v_values)
    v_values = np.clip(v_values, vmean-2*vstd, vmean+2*vstd)
    v_values -= np.min(v_values)
    v_values /= np.max(v_values)
    v_floor = 0.1
    v_values *= 0.95-v_floor
    v_values += v_floor
    false_color_hsv[:,:,2] = v_values
    
    s_values = false_color_hsv[:,:,1]
    smean = np.mean(s_values)
    sstd = np.std(s_values)
    s_values = np.clip(s_values, smean - 3*sstd, smean + 3*sstd)
    s_values -= np.min(s_values)
    s_values /= np.max(s_values)
    s_floor = 0.50
    s_values *= 1-s_floor
    s_values += s_floor
    false_color_hsv[:,:,1] = s_values
    
    false_color_hsv[:,:,0] -= np.min(false_color_hsv[:,:,0])
    false_color_hsv[:,:,0] /= np.max(false_color_hsv[:,:,0])
    
    cc_false_color_img = hsv_to_rgb(false_color_hsv)
    
    return cc_false_color_img, false_color_img

def flatten_with_circular_mask(ewpc,circular_mask):
    '''
    Depending on cutoff value chosen by user based on a plot of the explained variance ratio - plots corresponding PCA components, the real space
    map of the score for each component, false color image with the cumulative sum of different PCA components till the cutoff.

    :Parameters:
        ewpc : 4d numpy array
            Cepstral tranformed 4D-STEM data
        circular_mask: boolean ndarray,
            mask used for removing the central spot in cepstral images.
        
    :Returns:
        flat_cep : 2d numpy array
            Flattened (from 4d to 2d) and masked cepstral data      
    '''
    flat_cep=np.zeros((ewpc.shape[0]*ewpc.shape[1],np.sum(circular_mask)))
    print('flattenning the cepstral signal')
    for i in tqdm(range(flat_cep.shape[0])):
        ii,jj=np.unravel_index(i,(ewpc.shape[0],ewpc.shape[1]))
        cur_slice=ewpc[ii,jj,:,:]
        flat_cep[i,:]=cur_slice[circular_mask.astype('bool')]
    return flat_cep

def plot_false_color_img(pca,scoresT,circular_mask,cmap='jet'):
    '''
    Depending on cutoff value chosen by user based on a plot of the explained variance ratio - plots corresponding PCA components, the real space
    map of the score for each component, false color image with the cumulative sum of different PCA components till the cutoff.
    
    :Parameters:
        pca : PCA model
        scoresT : result from the fit_transform method of the PCA model applied on the diffraction/cepstral data
        circular_mask: boolean ndarray,
            mask used for removing the central spot in cepstral images.
        cmap : string
            colormap for plot of PCA component and associated map of score
        
    :Returns:
        None
    '''    
    n_components=scoresT.shape[1]
    xy_shape=int(np.sqrt(scoresT.shape[0]))
    scores=np.reshape(scoresT.T,(n_components,xy_shape,xy_shape))
    output = widgets.Output()
    cutoff_slider = widgets.IntSlider(value=1, min=1, max=n_components, step=1, description='cut off')
    check_region = widgets.Checkbox(value=False,description='Log Y axis')
    
    fig=plt.figure(figsize=(8,8))
    ax1=fig.add_subplot(221)
    ax2=fig.add_subplot(222)
    ax3=fig.add_subplot(223)
    ax4=fig.add_subplot(224)
    
    ax1.plot(np.arange(1,1+n_components),pca.explained_variance_ratio_)
    ax1.set_xlabel('# components')
    ax1.set_ylabel('Explained Variance Ratio')
    vline=ax1.axvline(color='k')
    cut_off=cutoff_slider.value
    fc = generate_false_color_image(scores[:cut_off,:,:], 0)[0]
    ax2.set_title('False colored image')
    ax2.imshow(fc)
    ax3.imshow(scores[cut_off,:,:],cmap=cmap)
    ax4.imshow(unflatten_circular_mask(pca.components_[cut_off,:],circular_mask),cmap=cmap)
    ax4.set_title('Component #'+str(cut_off))     
    ax3.set_title('Score #'+str(cut_off))
    for ax in [ax2,ax3,ax4]:
        ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout()
    def update_fc(cut_off):
        vline.set_xdata(cut_off)
        fc = generate_false_color_image(scores[:cut_off,:,:], 0)[0]
        ax2.imshow(fc)
        ax3.imshow(scores[cut_off-1,:,:],cmap=cmap)
        ax3.set_title('Score #'+str(cut_off))
        ax4.imshow(unflatten_circular_mask(pca.components_[cut_off-1,:],circular_mask),cmap=cmap)
        ax4.set_title('Component #'+str(cut_off))     
        fig.canvas.draw()

    def yaxis_scale(change):
        if change.new:
            ax1.set_yscale('log')
        else:
            ax1.set_yscale('linear')
        fig.canvas.draw()

    def slider_eventhandler(change):
        update_fc(change.new)
    cutoff_slider.observe(slider_eventhandler,names='value',type='change')
    check_region.observe(yaxis_scale,names='value',type='change')
    input_widgets_1 = widgets.HBox([check_region,cutoff_slider])
    display(input_widgets_1)

    
def unflatten_circular_mask(flat_array,haadf_mask):
    '''
    Recover the 2D array from 1D array after masking 

    :Parameters:
        flat_array: 1D ndarray 
        haadf_mask: boolean ndarray
    '''
    mask_pos=np.where(haadf_mask)
    cur_slice=np.zeros(haadf_mask.shape,flat_array.dtype)
    for i in range(np.sum(haadf_mask.astype('bool'))):
        cur_slice[mask_pos[0][i],mask_pos[1][i]]=flat_array[i]
    return cur_slice        

def plot_scores_components(pca,scores,n1,n2,circular_mask,cmap='jet',figsize=(9,9)):
    '''
    Plots PCA results

    :Parameters:
        pca : pca model from scikit learn
        scores: ndarray, the scores matrix - result of fitting the data into the model
        n1,n2: number of rows and columns to display the scores/components, ideally n1xn2=n_components
        circular_mask: boolean ndarray, used to unflatten the 1D components in pca model 
            and instead display them 2D image. Should be the same mask used for removing the central spot in cepstral images.
        cmap: matplotlib colormap
        figsize: size of figure in inches
    :Returns: two n1xn2 figure, first one shows the fitting scores, second one shows pca components
    ''' 
    n_components=scores.shape[1]
    fig,axes=plt.subplots(n1,n2,figsize=figsize)
    k=0
    for i in range(n1):
        for j in range(n2):
            axes[i,j].imshow(scores[:,k].reshape((rx,ry)),cmap=cmap)
            axes[i,j].set_title('Score #'+str(k+1))

            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            k=k+1
            if k==n_components:
                break            
#     fig.savefig(saveDir+'/scores_all.png')
#     fig.savefig(saveDir+'/scores_all.pdf')
    fig,axes=plt.subplots(n1,n2,figsize=figsize)
    k=0
    for i in range(n1):
        for j in range(n2):
            axes[i,j].cla()
            axes[i,j].imshow(unflatten_circular_mask(pca.components_[k,:],circular_mask),cmap=cmap)
            axes[i,j].set_title('Component #'+str(k+1))     
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            k=k+1
            if k==n_components:
                break                        
#     fig.savefig(saveDir+'/components_all.png')
#     fig.savefig(saveDir+'/components_all.pdf')


def plot_kmeans_dict(kmeans_dict):
    '''
    Plots kmeans results

    :Parameters:
        kmeans_dict : dictionary
    :Returns: figure with varying number of kmeans components to display
    '''
    clusters=list(kmeans_dict.keys())
    clusters.sort()
    output = widgets.Output()
    slider = widgets.IntSlider(value=clusters[0], min=clusters[0], max=clusters[-1], step=1, description='# Clusters')
    check_region = widgets.Checkbox(value=False,description='Log Y axis')
    
    fig=plt.figure(figsize=(9,4))
    ax1=fig.add_axes([0.1,0.1,0.3,0.675])
    ax2=fig.add_axes([0.5,0.1,0.3,0.675])
    ax_cm_t=fig.add_axes([0.9,0.3,0.05,0.4])
    
    wss=[]
    for i in clusters:
        wss.append(kmeans_dict[i]['wss'])
        
    ax1.plot(clusters,wss)
    ax1.set_xlabel('# Clusters')
    ax1.set_ylabel('Within cluster sum of squares')
    n_clusters=int(slider.value)
    
    vline=ax1.axvline(color='k')
    vline.set_xdata(n_clusters)
    ax1.set_xlim([clusters[0]-0.5,clusters[-1]+0.5])
    current_cmap = plt.get_cmap('RdBu', int(n_clusters)) 
    fc = kmeans_dict[n_clusters]['label']
    ax2.set_title('Cluster labels')
    ax2.imshow(fc,cmap=current_cmap)
    ax_cm_t.cla()
    plt.colorbar(scm(norm=mpl.colors.Normalize(vmin=-0.5,vmax=n_clusters-0.5),cmap=current_cmap),cax=ax_cm_t, ticks=np.arange(0,n_clusters))        

    for ax in [ax2]:
        ax.set_xticks([]);ax.set_yticks([])
    def update_fc(n_clusters):
        current_cmap = plt.get_cmap('RdBu', int(n_clusters)) 
        vline.set_xdata(n_clusters)
        fc = kmeans_dict[n_clusters]['label']
        ax2.imshow(fc,cmap=current_cmap)
        ax2.set_xticks([]);ax2.set_yticks([])
        ax_cm_t.cla()
        plt.colorbar(scm(norm=mpl.colors.Normalize(vmin=-0.5,vmax=n_clusters-0.5),cmap=current_cmap),cax=ax_cm_t, ticks=np.arange(0,n_clusters))        
        fig.canvas.draw()

    def yaxis_scale(change):
        if change.new:
            ax1.set_yscale('log')
        else:
            ax1.set_yscale('linear')
        fig.canvas.draw()

    def slider_eventhandler(change):
        update_fc(change.new)
    slider.observe(slider_eventhandler,names='value',type='change')
    check_region.observe(yaxis_scale,names='value',type='change')
    input_widgets_1 = widgets.HBox([check_region,slider])
    display(input_widgets_1)


def perform_kmeans(scores,cut_off,clusters_range,mask=None):
    '''
    Perform K-means clustering 
    :Parameters:
        scores : ndarray of shape n_features x n_components from PCA decomposition
        cut_off: integer indicating the highest number of pca components to use in clustering
        clusters_range: list containing minimum and maximum number of clusters for segmentation
        mask: boolean ndarray, if None all points are considered
        xy_shape: shape of the real space data, can set to None if rows=columns.
        
    :Returns: kmeans dictionary containing the segmentation results for different number of clusters, 
          as well as, the corresponding within-cluster-sum of squared errors (figure of merit used to
          determine the optimal number of clusters).
    '''
    rx=int(np.sqrt(scores.shape[0]))
    ry=rx
    kmeans_dict={}
    if mask==None:
        mask=np.ones((rx,ry)).astype('bool')
    mask_scores=np.zeros((cut_off,np.sum(mask)))
    mask_pos=np.where(mask)
    for i in range(cut_off):
        curr_slice=scores[:,i].copy().reshape((rx,ry))
        curr_slice-=curr_slice[mask].min()
        curr_slice/=curr_slice[mask].max()
        mask_scores[i,:]=curr_slice[mask]
    print('Performing clustering')
    for n_clusters in tqdm(range(clusters_range[0],clusters_range[1])):
        if n_clusters in kmeans_dict.keys():
            continue
        kmeans_dict[n_clusters]={}
        kmeans=KMeans(n_clusters=n_clusters).fit(mask_scores.T)
        labels=kmeans.labels_
        labels_unf=np.ones((rx,ry))*n_clusters
        for i in range(np.sum(mask)):
            labels_unf[mask_pos[0][i],mask_pos[1][i]]=labels[i]
        kmeans_dict[n_clusters]['wss']=kmeans.inertia_
        kmeans_dict[n_clusters]['label']=labels_unf
    
    return kmeans_dict



def trim_spotMaps(spotMaps,strain_id): 
    '''
    creates new spotmaps dictionary that has only selected spot information.

    :Parameters:
        spotMaps : dictionary
            contains information about the peak positions of the different EWPC spots that were tracked
        strain_id : list
            spots to be used for strain mapping     
    :Returns: 
        new_spotMaps : dictionary
            trimmed spotMaps dictionary

    ''' 
    new_spotMaps={}
    for key in spotMaps.keys():
        if not key in ['wins','roi']:
            new_spotMaps[key]=[]
            for i in strain_id:
                new_spotMaps[key].append(spotMaps[key][i-1])
            new_spotMaps[key]=np.array(new_spotMaps[key])
    return new_spotMaps


def calculate_DF(ewpc,wins):
    ''' 
    Calculate dark field image from the window positions
    :Parameters:
        ewpc : 4D array
        wins : list of size n (where n is the number of windows) containing the 4 coordinates to identify the window position and extent

    :Returns: 2D array, img
    ''' 
    wins=np.array(wins)
    cep_mask=np.zeros((ewpc.shape[2],ewpc.shape[3])).astype('bool')
    for i in range(len(wins)):
        cep_mask[wins[i,2]:wins[i,3],wins[i,0]:wins[i,1]]=True
    img=np.sum(ewpc*cep_mask,axis=(-2,-1))
    return img



def segment_manually(img,thresh=None,figureSize=(8,4),bins=100):
    '''
    Displays original image, segmented image and the image histogram.

    :Parameters:
        img : 2D array
            Image to be shown on the left of the figure.
        thresh : integer or float, threshold for segmentation    
    :Returns: figure with three subplots
    '''
    fig=plt.figure(figsize=figureSize)
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)
    
    if thresh==None:
        thresh=img.mean()

    bImg=img>thresh

    ax1.set_title('DF C-STEM')
    ax2.set_title('Segmentation results')
    ax1.imshow(img)
    ax2.imshow(bImg)
    a,b=np.histogram(img.flatten(),bins=bins)
    ax3.bar(b[:-1],a,b[1]-b[0],align='edge',edgecolor=None,facecolor='k')

    ylim=ax3.get_ylim()
    ax3.plot(np.ones(2)*thresh,ylim,'r--')
    ax3.set_ylim(ylim)
    for ax in [ax1,ax2]:
        ax.set_xticks([]);ax.set_yticks([])

    fig.tight_layout()
