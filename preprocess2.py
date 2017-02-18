


import numpy as np

import pandas as pd
import dicom

import os
import scipy.ndimage
import matplotlib.pyplot as plt

import csv 

from skimage import measure , morphology , filters 

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



print("modulos cargados")


def load_scan(path):

    slices = [ dicom.read_file(path + '/' +s ) for s in os.listdir(path)]

    slices.sort( key = lambda x : int( x.ImagePositionPatient[ 2 ] ) )

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    # convertir a las unidades medicas apropiadas
    
    image = np.stack([ s.pixel_array for s in slices ] )

    image = image.astype( np.int16 )

    image[image == -2000] = 0

    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number].slope =  slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)


        image[slice_number] += np.int16(intercept)


    return np.array(image , dtype = np.int16)


def resample( image , scan , new_spacing = [ 1 , 1, 1] ):

    spacing = np.array( [scan[0].SliceThickness ] + scan[0].PixelSpacing, dtype = np.float32 )
    resize_factor = spacing / new_spacing

    new_real_shape = image.shape* resize_factor
    new_shape = np.round( new_real_shape )

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing



def largest_label_volume(im ,bg = -1):

    vals , counts = np.unique( im,  return_counts= True)

    counts = counts[vals != bg ]
    vals =  vals[ vals != bg ]
    if len(counts) > 0:
        return vals[ np.argmax( counts ) ]
    else:
        return None

def segment_lung_mask(image , fill_lung_structures = False):
    th = -320 
    binary_image = np.array( image > th  , dtype = np.int8 ) + 1

    labels = measure.label( binary_image)

    background_label = labels[0 , 0 , 0 ]
    binary_image[background_label == labels] = 2
    # 2 es aire alrededor de una persona


    if fill_lung_structures:

        for i , axial_slice in enumerate( binary_image ):
            axial_slice = axial_slice - 1
            labeling = measure.label( axial_slice )

            l_max = largest_label_volume( labeling , bg = 0  )

            if l_max is not None:
                binary_image[i][labeling != l_max] = 1 #air

    binary_image -= 1
    binary_image = 1 - binary_image

    labels = measure.label( binary_image , background = 0)
    l_max = largest_label_volume( labels , bg = 0)

    if l_max is not None:
        binary_image[labels != l_max] = 0

    return binary_image 


def sphere(radius = 5 ):
    # mascara para la dilatacion morfologica
    x,y,z = np.mgrid[ -radius : radius  ,-radius : radius , -radius : radius  ]
    
    
    mask = x*x + y*y + z*z <= radius*radius
    sph = np.zeros( mask.shape  )
    
    sph[ mask ] = 1

    return sph

def morphological_resize( input_scan):
    # size 300x300x300 para cada imagen
    base_size = 420
    img = np.zeros( (base_size,base_size,base_size) )
    
   
    shp_morpho  = sphere( 5 )

    dil = morphology.dilation( input_scan , shp_morpho )

    img[ np.where( dil == 1)  ] = 1

    
    return img

def rotate_90( m , k =1 , axis = 2 ):

    m = np.swapaxes( m , 2 , axis)
    m = np.rot90( m , k)
    m = np.swapaxes(  m , 2  , axis )
    return m


def get_augmented_data(  scan  ):
    # es un arreglo 3d
    # path labels es la ruta al archivo .csv de las etiquetas
    # has cancer identifica si el scaneo correponde a un paciente con cancer o no
    
    k = 1 # parametro de las rotaciones
    
    for axis in range(3):
        # para cada eje
        # rotar el arreglo
        for k in range( 1 , 4):
            new_scan = rotate_90(  scan , k  , axis )
            yield new_scan
        #new_data.append( new_scan )

    new_scan = filters.gaussian( scan , sigma = 0.5 )
    
    yield new_scan 
    
def process_folder(path_data  , path_out , path_labels):

    # iistar carpetas en path_data
    #
    patients = [ p for p in os.listdir( path_data)  ]

    labels = pd.read_csv( path_labels )
        
    
    # por cada paciente
    new_patients = []
    
    for patient in patients:

        patient_scan = load_scan( path_data + "/"+ patient )
        cancer = labels.loc[ labels['id'] == patient    ]['cancer']
        
        patient_pixels = get_pixels_hu( patient_scan )
        patient_rescale , spacing = resample( patient_pixels , patient_scan , [1,1,1])

        segmented_lungs_fill = segmented_lung_mask( patient_rescale , True)
        # unificar size y hacer dilacion morfologica
        
        segmented_lung_morph = morphological_size(segmented_lungs_fill)
        # save the image, same name
        np.save( path_out + "/" + p , segmented_lung_morph )
        ind_n = 100
        for new_scan in get_augmented_data( segmented_lung_morph ):
            # guardar el nuevo dato 
            np.save( path_out + "/" + p + str(ind_n) , new_scan)
            new_patient = [ patient+str(ind_n)  , int( cancer ) ] 
            new_patients.append( new_patient )
            ind_n = ind_n + 5
            
            # guardar el nuevo dato

    labels = labels.append(  pd.DataFrame( new_patients , columns = labels.columns  )  )
    
    labels.to_csv( path_labels )
    print patients
    


path_data = "../data/sample_images"
path_out = "../data/processed"
path_labels = "./data/sample_images/stage1_labels.csv"

process_folder(path_data, path_out , path_labels )

"""   
patients = [ "0ddeb08e9c97227853422bd71a2a695e" ]

patient_one = load_scan( "../data/sample_images/" + patients[0] )

patient_one_pixels = get_pixels_hu( patient_one )


image_res , spacing = resample( patient_one_pixels, patient_one , [1,1,1])

print patient_one_pixels.shape

print image_res.shape
#plot_3d( image_res ) 

segmented_lungs = segment_lung_mask( image_res , True )

#plt.imshow( segmented_lungs[80] , cmap = plt.cm.gray )
#plt.show()


segmented_aug = morphological_resize( segmented_lungs )

for new_dato  in get_augmented_data( segmented_aug ):
    print new_dato.shape
    print ("que bonito que bello")

#plt.imshow( segmented_aug[80] , cmap = plt.cm.gray )
#plt.show()

#plt.imshow( segmented_lungs[80] , cmap = plt.cm.gray )
#plt.show()

END COMMENT
"""
