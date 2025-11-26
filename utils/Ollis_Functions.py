
import numpy as np
import scipy
import error_propagation

### ATOMAP ADDED FUNCTIONS ###

def AtomapFilterCropAtomPositionList(positionList, col_crop_val_left, col_crop_val_right, row_crop_val_up, row_crop_val_down):
    
    positionList = positionList[positionList[:,0] < col_crop_val_right]
    positionList = positionList[positionList[:,0] > col_crop_val_left]
    positionList = positionList[positionList[:,1] < row_crop_val_down]
    positionList = positionList[positionList[:,1] > row_crop_val_up]

    return positionList


### SUPERSTEM Functions ###


## NUMPY FFT filter like in DM
def SuperSTEM_Gatan_FFT_filter(Array):


    hn=np.outer(np.hanning(Array.shape[0]),np.hanning(Array.shape[1]))
    diffrac = np.fft.fftshift(np.fft.fft2(Array*hn))
    FFT = np.log(np.abs(diffrac)+1)
    
    return FFT


## define mean with error propagation for np arrays

def error_prop_mean(Array):
    
    for i, eintrag in enumerate(Array):
        
        #case distinction
        if i==0:
            sum_val_with_error = eintrag
        else:
            sum_val_with_error = sum_val_with_error + eintrag 
    
    mean_val_with_error = sum_val_with_error/(i+1)
    
    return mean_val_with_error

def error_prop_round(value, decimal):

    value_round = error_propagation.Complex(np.round(value.value,decimals=decimal), np.round(value.error,decimals=decimal))

    return value_round




def SuperSTEM_get_distance_direction_map(atom_pos_of_sublattice, initial_direction, treshhold):
    
    #shift
    shifted_pos = atom_pos_of_sublattice-initial_direction
    
    ## create KD_tree
    KD_dir_dist_tree = scipy.spatial.KDTree(shifted_pos)
    KD_dir_dist_tree_tresh = treshhold
    
    dist_dir_map = []
       
    for vec in atom_pos_of_sublattice:
        # print(vec)
        KD_dir_dist_tree_res = KD_dir_dist_tree.query(vec)
        # print(vec, KD_dir_dist_tree_res)
        if KD_dir_dist_tree_res[0] < KD_dir_dist_tree_tresh:
            vec_found = atom_pos_of_sublattice[KD_dir_dist_tree_res[1]]
            # print(vec-vec_found)
            dist = np.linalg.norm(vec-vec_found)
            dist_dir_map.append(np.append(vec,dist))

    #convert to np array
    dist_dir_map = np.array(dist_dir_map)
    
    return dist_dir_map



def SuperSTEM_get_mean_lattice_constant_from_next_neighbour(atom_pos_of_sublattice, initial_directions, treshhold):

    #initialiue 
    KD_dir_dist_tree_tresh = treshhold
    KD_trees = []
    shifted_positions = []

    #loopie loop
    for direct in initial_directions:
        #shift 
        shifted_pos = atom_pos_of_sublattice-direct
        ## create KD_tree
        KD_dir_dist_tree = scipy.spatial.KDTree(shifted_pos)
        
        #append to lists
        shifted_positions.append(shifted_pos)
        KD_trees.append(KD_dir_dist_tree)

    
    # #shift
    # shifted_pos = atom_pos_of_sublattice-initial_direction
    
    # ## create KD_tree
    # KD_dir_dist_tree = scipy.spatial.KDTree(shifted_pos)
    # KD_dir_dist_tree_tresh = treshhold
    
    mean_lattice_constant_map = []
    
       
    for vec in atom_pos_of_sublattice:
        next_neighbour_distances = []
        # print(vec)
        for KD_dir_dist_tree in KD_trees:
            KD_dir_dist_tree_res = KD_dir_dist_tree.query(vec)
            # print(vec, KD_dir_dist_tree_res)
            if KD_dir_dist_tree_res[0] < KD_dir_dist_tree_tresh:
                vec_found = atom_pos_of_sublattice[KD_dir_dist_tree_res[1]]
                # print(vec-vec_found)
                dist = np.linalg.norm(vec-vec_found)
                # dist_dir_map.append(np.append(vec,dist))
                next_neighbour_distances.append(dist)

        #calculate mean and std
        next_neighbour_distances = np.array(next_neighbour_distances)
        mean = np.mean(next_neighbour_distances)
        std  = np.std(next_neighbour_distances)

        #convert to complex number
        mean_with_std_error = error_propagation.Complex(mean, std) 
        
        mean_lattice_constant_map.append(mean_with_std_error)
        
    #convert to np array
    mean_lattice_constant_map = np.array(mean_lattice_constant_map)
        
    return mean_lattice_constant_map




## Define useful functions

def SuperSTEM_crop_matrix_around_position(matrix,pos,margin):

    
    #pos as np array!

    #convert to int for indexing
    pos = pos.round().astype('int')
            
    #cropping  
    x_low = pos[0]-margin
    x_up = pos[0]+margin
    y_low = pos[1]-margin
    y_up = pos[1]+margin
    
    matrix_cropped = matrix[ y_low:y_up, x_low:x_up] # i hate it!

    return matrix_cropped
    
def SuperSTEM_average_all_positions(matrix,pos_array,margin):

    matrix_stack = []

    ## get boundary
    x_bound, y_bound = matrix.shape

    for pos in pos_array:

        #boundary check
        x_low = pos[0]-margin
        x_up = pos[0]+margin
        y_low = pos[1]-margin
        y_up = pos[1]+margin

        #append only if not out of bounds
        if not x_low < 0 and not y_low < 0 and not x_up > x_bound and not y_up > y_bound:
        
            matrix_cropped = SuperSTEM_crop_matrix_around_position(matrix, pos, margin)
            matrix_stack.append(matrix_cropped)

    #convert to np array
    matrix_stack = np.array(matrix_stack)
    #mena of matrix
    matrix_mean = np.nanmean(matrix_stack,axis=0)

    return matrix_mean

def SuperSTEM_average_all_positions_subpixel(matrix,pos_array,margin,upscale_factor):

    '''simple docstring test'''

    matrix_upscaled = scipy.ndimage.zoom(matrix,upscale_factor,order=2) #cubic interpolation
    pos_array_upscaled = pos_array*upscale_factor
    margin_upscaled =  margin*upscale_factor
    
    matrix_stack = []


    ## get boundary
    x_bound, y_bound = matrix_upscaled.shape

    for pos in pos_array_upscaled:
        
        #boundary check
        x_low = pos[0]-margin_upscaled
        x_up = pos[0]+margin_upscaled
        y_low = pos[1]-margin_upscaled
        y_up = pos[1]+margin_upscaled

        #append only if not out of bounds
        if not x_low < 0 and not y_low < 0 and not x_up > x_bound and not y_up > y_bound:
            # print('reached')
        
            matrix_cropped = SuperSTEM_crop_matrix_around_position(matrix_upscaled, pos, margin_upscaled)
            matrix_stack.append(matrix_cropped)

    # convert to np array
    matrix_stack = np.array(matrix_stack)
    # mena of matrix
    matrix_mean = np.nanmean(matrix_stack,axis=0)

    return matrix_stack, matrix_mean


def SuperSTEM_crop_matrix_around_position_XYmargin(matrix,pos,margin):

    '''Crop Matrix around position with margins in x and y'''
    
    #pos as np array!

    #convert to int for indexing
    pos = pos.round().astype('int')
            
    #cropping  
    x_low = pos[0]-margin[0]
    x_up = pos[0]+margin[0]
    y_low = pos[1]-margin[1]
    y_up = pos[1]+margin[1]
    
    matrix_cropped = matrix[ y_low:y_up, x_low:x_up] # i hate it!

    return matrix_cropped



def SuperSTEM_average_all_positions_subpixel_XYmargin(matrix,pos_array,margin,upscale_factor):

    '''Function to make image stack from positions within one input image with subpixel accuracy'''

    matrix_upscaled = scipy.ndimage.zoom(matrix,upscale_factor,order=2) #cubic interpolation #prefilter false for nan containing data
    pos_array_upscaled = pos_array*upscale_factor
    margin_upscaled =  margin*upscale_factor
    
    matrix_stack = []


    ## get boundary
    x_bound, y_bound = matrix_upscaled.shape

    for pos in pos_array_upscaled:
        
        #boundary check
        x_low = pos[0]-margin_upscaled[0]
        x_up = pos[0]+margin_upscaled[0]
        y_low = pos[1]-margin_upscaled[1]
        y_up = pos[1]+margin_upscaled[1]

        #append only if not out of bounds
        if not x_low < 0 and not y_low < 0 and not x_up > x_bound and not y_up > y_bound:
            # print('reached')
        
            matrix_cropped = SuperSTEM_crop_matrix_around_position_XYmargin(matrix_upscaled, pos, margin_upscaled)
            matrix_stack.append(matrix_cropped)

    # convert to np array
    matrix_stack = np.array(matrix_stack)
    # mena of matrix
    matrix_mean = np.nanmean(matrix_stack,axis=0)

    return matrix_stack, matrix_mean



def SuperSTEM_Local_Scanning_Distortions_for_fitted_Positions(pos,s_distortion_x_data,s_distortion_y_data):

    """
    Function to apply the fitted local scanning distortions to the fitted atom positions
    
    Parameters:
        pos (Array): initially fitted atom positions
        s_distortion_x_data (Array): scan distortion data in x direction from atomap (fast scan line - rows)
        s_distortion_y_data (Array): scan distortion data in y direction from atomap (slow scan line - cols)

    Returns:
        (Array): Atom Positions corrected by local scan distortions    
    """
    
    #atomap automatically up-scales the scan distortion map by a factor of 2
    upscale_correction_factor = 2
    pos_scaled = pos*upscale_correction_factor # col rows
    pos_distortion_corrected = []
    
    for current_pos in pos_scaled:

        #convert current position to index numbers
        idxs = np.round(current_pos).astype('int')

        row_corr_val = s_distortion_x_data[idxs[1],idxs[0]]  ## stored in rows col format
        col_corr_val = s_distortion_y_data[idxs[1],idxs[0]]  ## stored in rows col format
        
        new_pos = current_pos+[col_corr_val,row_corr_val]
        pos_distortion_corrected.append(new_pos)

    pos_distortion_corrected = np.array(pos_distortion_corrected)

    return pos_distortion_corrected

# print(pos_list1[0,:]*2)
# idxs = np.round(pos_list1[0,:]*2).astype('int')
# s_distortion_x2_data = s_distortion_x2.data
# s_distortion_y2_data = s_distortion_y2.data
# s_distortion_x2_data[2029,2043]
# s_distortion_y2_data[2029,2043]
# s_distortion_y2_data[idxs[0], idxs[1]]


def k_length_min_wrapper(self,k_max):

    # Store k_max
    self.k_max = np.asarray(k_max)
    
    # Find shortest lattice vector direction
    k_test = np.vstack(
        [
            self.lat_inv[0, :],
            self.lat_inv[1, :],
            self.lat_inv[2, :],
            self.lat_inv[0, :] + self.lat_inv[1, :],
            self.lat_inv[0, :] + self.lat_inv[2, :],
            self.lat_inv[1, :] + self.lat_inv[2, :],
            self.lat_inv[0, :] + self.lat_inv[1, :] + self.lat_inv[2, :],
            self.lat_inv[0, :] - self.lat_inv[1, :] + self.lat_inv[2, :],
            self.lat_inv[0, :] + self.lat_inv[1, :] - self.lat_inv[2, :],
            self.lat_inv[0, :] - self.lat_inv[1, :] - self.lat_inv[2, :],
        ]
    )
    k_leng_min = np.min(np.linalg.norm(k_test, axis=1))
    num_tile = np.ceil(self.k_max / k_leng_min)
    num_GB = (2*num_tile+1)**3*8/(1e9)
    

    return k_leng_min, num_tile, num_GB, k_test.shape, k_test



    
