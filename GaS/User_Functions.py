## import packages needed
import os
from pathlib import Path
import hyperspy.api as hs
import matplotlib.pyplot as plt

# define color of overlay
import matplotlib.cm as cm
import matplotlib.colors as mcolors

## Colorcode generation

def generate_color_codes(num_colors):
    # Use the updated colormap API to get a colormap
    colormap = plt.colormaps['bwr']
    # colormap = plt.colormaps['PRGn']
    #colormap = plt.colormaps['nipy_spectral_r']
    #colormap = plt.colormaps['CMRmap']
    #colormap = plt.colormaps['gist_ncar_r']
    # Generate evenly spaced colors and convert to HEX
    if num_colors == 1:
        color_codes = [mcolors.to_hex(colormap(i)) for i in range(num_colors)]
    else:
        color_codes = [mcolors.to_hex(colormap(i / (num_colors - 1))) for i in range(num_colors)]
    return color_codes



#Function to list folder contents in python
def ListFilesFormated(startpath):
    
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        
        for idx, f in enumerate(files):
            print('{}{}'.format(subindent, str(idx) +": " + f))
            
    return files


#Function to list folder contents in python
def ListFilesFormatedSortedByTime(startpath):
    files =[]
    paths = sorted(Path(startpath).iterdir(), key=os.path.getmtime)

    level = startpath.replace(startpath, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print('{}{}/'.format(indent, os.path.basename(startpath)))
    subindent = ' ' * 4 * (level + 1)
    
    for idx, eintrag in enumerate(paths):
        basename = os.path.basename(eintrag)
        print('{}{}'.format(subindent, str(idx) +": " + basename))
        files.append(basename)
            
    return files

#Function to list folder contents in python
def ListFilesFormatedSortedByTimeAndFilterForString(startpath, stringFilter):
    files =[]
    paths = sorted(Path(startpath).iterdir(), key=os.path.getmtime)

    level = startpath.replace(startpath, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print('{}{}/'.format(indent, os.path.basename(startpath)))
    subindent = ' ' * 4 * (level + 1)

    i = 0
    for idx, eintrag in enumerate(paths):
        basename = os.path.basename(eintrag)
        if stringFilter in basename:
            print('{}{}'.format(subindent, str(i) +": " + basename))
            files.append(basename)
            i += 1
            
    return files


def PantaRheiConvertHyperspyToPngPreview(saveFolderName, filePath, conversionList, resolutionDpi):
    
    #create Folder for preview images
    if not os.path.exists(os.path.join(filePath, saveFolderName)):
        os.mkdir(os.path.join(filePath, saveFolderName))
    
    
    # # save file evals
    savePathSTEMPreview = os.path.join(filePath, saveFolderName)
    
    ## create png Preview images
    for idx, eintrag in enumerate(conversionList):
        print(eintrag)
        fullFileName = os.path.join(filePath, conversionList[idx])
        STEMImage = hs.load(fullFileName)
        
        STEMImage.plot()
        
        plt.savefig(os.path.join(savePathSTEMPreview,eintrag.replace('.hspy','.png')), dpi=resolutionDpi)
        plt.close()

def PantaRheiConvertHyperspyToPngPreviewWithNmScale(saveFolderName, filePath, conversionList, resolutionDpi):
    
    #create Folder for preview images
    if not os.path.exists(os.path.join(filePath, saveFolderName)):
        os.mkdir(os.path.join(filePath, saveFolderName))
    
    
    # # save file evals
    savePathSTEMPreview = os.path.join(filePath, saveFolderName)
    
    ## create png Preview images
    for idx, eintrag in enumerate(conversionList):
        print(eintrag)
        fullFileName = os.path.join(filePath, conversionList[idx])
        STEMImage = hs.load(fullFileName)
        STEMImage.axes_manager.convert_units(units='nm')
        
        STEMImage.plot()
        
        plt.savefig(os.path.join(savePathSTEMPreview,eintrag.replace('.hspy','.png')), dpi=resolutionDpi)
        plt.close()


# def PantaRheiConvertHyperspyToPreviewWithNmScale(saveFolderName, filePath, conversionList, resolutionDpi, imageFormat):
    
#     #create Folder for preview images
#     if not os.path.exists(os.path.join(filePath, saveFolderName)):
#         os.mkdir(os.path.join(filePath, saveFolderName))
    
    
#     # # save file evals
#     savePathSTEMPreview = os.path.join(filePath, saveFolderName)
    
#     ## create png Preview images
#     for idx, eintrag in enumerate(conversionList):
#         print(eintrag)
#         fullFileName = os.path.join(filePath, conversionList[idx])
#         STEMImage = hs.load(fullFileName)
#         STEMImage.axes_manager.convert_units(units='nm')
        
#         STEMImage.plot()
        
#         plt.savefig(os.path.join(savePathSTEMPreview,eintrag.replace('.hspy',imageFormat)), dpi=resolutionDpi)
#         plt.close()


def PantaRheiConvertHyperspyToPreviewWithNmScale(saveFolderName, filePath, conversionList, imageFormat):
    
    #create Folder for preview images
    if not os.path.exists(os.path.join(filePath, saveFolderName)):
        os.mkdir(os.path.join(filePath, saveFolderName))
    
    
    # # save file evals
    savePathSTEMPreview = os.path.join(filePath, saveFolderName)
    
    ## create png Preview images
    for idx, eintrag in enumerate(conversionList):
        print(eintrag)
        fullFileName = os.path.join(filePath, conversionList[idx])
        STEMImage = hs.load(fullFileName)
        STEMImage.axes_manager.convert_units(units='nm')
        
        STEMImage.plot()
        
        plt.savefig(os.path.join(savePathSTEMPreview,eintrag.replace('.hspy',imageFormat)))
        plt.close()

def PantaRheiConvertHyperspyToPreviewWithScale(saveFolderName, filePath, conversionList, imageFormat):
    
    #create Folder for preview images
    if not os.path.exists(os.path.join(filePath, saveFolderName)):
        os.mkdir(os.path.join(filePath, saveFolderName))
    
    
    # # save file evals
    savePathSTEMPreview = os.path.join(filePath, saveFolderName)
    
    ## create png Preview images
    for idx, eintrag in enumerate(conversionList):
        print(eintrag)
        fullFileName = os.path.join(filePath, conversionList[idx])
        STEMImage = hs.load(fullFileName)

        #unit conversion if possible
        if STEMImage.axes_manager[0].units == 'm':
            STEMImage.axes_manager.convert_units(units='nm')
        
            STEMImage.plot()
        else:
            # plot scaled for diffraction patterns
            STEMImage.plot(vmin='0th', vmax='99.9th')

        
        plt.savefig(os.path.join(savePathSTEMPreview,eintrag.replace('.hspy',imageFormat)))
        plt.close()


