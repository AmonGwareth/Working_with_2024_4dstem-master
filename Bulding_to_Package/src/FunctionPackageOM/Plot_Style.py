# Import all needed packages
import matplotlib.pyplot as plt

def Plot_Style_OM_activate(white_bkg_for_visualization):

    # ## Status
    # save_to_svg_and_png = False
    # white_bkg_for_visualization = True
    
    ## visualize components of the InSe spectrum
    cm = 1/2.54  # centimeters in inches
    plt.rcParams['figure.figsize'] = [10*cm, 10*cm]
    plt.rcParams['image.cmap'] = 'viridis'
    plt.rcParams['axes.linewidth'] = 1.5 # set the value globally
    
    ##Font
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.weight'] = 'normal' # set the value globally
    
    # set tick width
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.major.width'] = 1.5
    
    #tick direction
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    ## Setup Legend
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['legend.edgecolor'] = 'k'
    # plt.rcParams['legend.markerscale'] = 20
    # plt.rcParams['legend.loc'] = 'lower right'
    
    ## grid setup
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.linestyle'] = 'solid'
    plt.rcParams['grid.linewidth'] = 1.5
    
    ## Lines
    plt.rcParams['lines.linewidth'] = 2.5
    
    #Labels
    plt.rcParams['axes.labelweight'] = 'bold'       # Font weight for axis labels (e.g., xlabel, ylabel)
    
    #transparent background
    plt.rcParams['figure.facecolor'] = 'none'  # Makes the figure background transparent
    plt.rcParams['axes.facecolor'] = 'none'    # Makes the axes (plotting area) background transparent
    
    if white_bkg_for_visualization:
        # #to white for plot creation
        plt.rcParams['figure.facecolor'] = 'w'  # Makes the figure background transparent
        plt.rcParams['axes.facecolor'] = 'w'    # Makes the axes (plotting area) background transparent

# def Plot_Style_Cyberpunk_activate():
    
