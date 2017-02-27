#!/usr/bin/python

import numpy as np
import pandas as pd
import dicom

import os
import scipy.ndimage
import matplotlib.pyplot as plt

import csv 
import cv2


from skimage import measure , morphology , filters 

from skimage.filters.rank import entropy

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scipy.misc   

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
    
    image = np.stack([   s.pixel_array  for s in slices ] )

    image = image.astype( np.int16 )

    image[image == -2000] = 0

    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] =  slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
       
        image[slice_number] += np.int16(intercept)
        #if slice_number == 20:
           # plt.imshow( image[slice_number] , cmap = 'gray' )
           # plt.show()

    return np.array( image , dtype= np.int16 )


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
    th = -400 
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
        
    #binary_image = morphology.binary_erosion( binary_image , morphology.ball(2) )
    
    binary_image = morphology.binary_closing( binary_image , morphology.ball(5) )

    #binary_image = morphology.remove_small_holes( binary_image , 5 )
    
    return binary_image 

def reduce_scan( full_scan , bg = 0 ):
    # recibe el arreglo del scaneo de diferencias, estructuras internas de los pulmones
    # el objetivo es contar por cada scaneo 
    #for i , axial_slice in enumerate( full_scan ):
        # dado que es un arreglo binario, obtenemos la suma de pixeles ocupados
        # la idea es ordenar 
     #   fill_pixels = np.sum( axial_slice)

    
    #scan_sorted = sorted(  full_scan , key = lambda x: np.sum( x ) , reverse = True )
    print("sorting")
    scan_sorted = sorted(  full_scan , key = lambda x: np.max( entropy( np.array( x, np.uint8 ), morphology.disk( 5 )    ) ) , reverse = True )
    #biggest = scan_sorted[0]

    #indx = np.where( full_scan == biggest)
    print(len( full_scan ))
    
    #best_candidates = scan_sorted[0:15]
    

    scan_final = np.stack( [ cv2.resize(scan ,(150,150) ) for scan in scan_sorted[:150]  ]   )
    #np.save( 'scan_try.npy' , result  )

    # scan final dims ( 150,150 , 150 ) 
    return scan_final 
 
    """
    for i, scan in enumerate( scan_sorted[:150]   ) :
        #print( scan.shape )
        #scan = 1 - scan
        #scan = morphological_ops( scan )
        #print i 
        #img_ent = entropy( scan, morphology.disk( 5 ) )
        #scipy.misc.imsave( './scan-mask' + str( i) + '.jpg' , scan )
        #scipy.misc.imsave( './scan-mask-ent' + str( i) + '.jpg' , img_ent )
        scan_color = cv2.cvtColor( np.abs( np.array( scan , dtype= np.uint8 ) ) , cv2.COLOR_GRAY2RGB)
        print ( scan_color.shape )
        ss = cv2.resize( scan_color , ( 150 , 150 ) )
        
        
        #print( np.max( ss ) )
        #cv2.imwrite('./scan-mask' + str( i) + '.png' , np.abs( ss ) )
    #print np.argmax( best_candidates[0] )
    
    """
def morphological_ops( input_scan ):
     
    elemn = morphology.ball( 2 )
    input_scan = morphology.binary_closing( input_scan , elemn )
   
    return (  input_scan )

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
    
def process_folder(path_data  , path_out   , path_labels):

    # iistar carpetas en path_data
    #
    patients = [ p for p in os.listdir( path_data)  ]

    labels = pd.read_csv( path_labels )
        
    
    # por cada paciente
    new_patients = []
    much_data =  []
    not_much_data = []
    # test 2  pacientes 
    for patient in patients:  
        print (patient)
        patient_scan = load_scan( path_data + "/"+ patient )
        cancer = labels.loc[ labels['id'] == patient    ]['cancer']
        
        patient_pixels = get_pixels_hu( patient_scan )
	patient_rescale , spacing = resample( patient_pixels , patient_scan , [1,1,1])

        segmented_lungs_fill = segment_lung_mask( patient_rescale , True)
        # unificar size y hacer dilacion morfologica
        
        high_vals = segmented_lungs_fill == 0
        patient_rescale[ high_vals ] = 0

        scan_final = reduce_scan ( patient_rescale )
        dat = [ scan_final , cancer ]
        
        much_data.append( dat )
        not_much_data.append( dat )
        # save the image, same name
        for new_scan in get_augmented_data( scan_final  ):
            # guardar el nuevo dato 
            new_data = [ new_scan , cancer ]
            #much_data.append( new_data )
            
            # guardar el nuevo dato

    
    np.save(path_out + '/' + 'process-aug.npy' , much_data  )
    np.save( path_out + '/' + 'process-regular.npy' , not_much_data)
    




    
path_data = "/mnt/lung_data/stage1"
path_out = "/mnt/lung_data/pre"
path_labels = "/mnt/lung_data/stage1_labels.csv"


process_folder( path_data , path_out , path_labels)




"""

#process_folder(path_data, path_out , path_labels )


# 0d06d764d3c07572074d468b4cff954f - sick
# 0de72529c30fe642bc60dcb75c87f6bd - clean
# 0c60f4b87afcb3e2dfa65abbbf3ef2f9 - sick
# 0c37613214faddf8701ca41e6d43f56e - sick
# 0acbebb8d463b4b9ca88cf38431aac69 - sick 
# 0c98fcb55e3f36d0c2b6507f62f4c5f1 - clean

patients = [ "0acbebb8d463b4b9ca88cf38431aac69" ]

patient_one = load_scan( "../../data/sample_images/" + patients[0] )

print("scan loaded")
patient_one_pixels = get_pixels_hu( patient_one )
#reescalar

image_res , spacing = resample( patient_one_pixels, patient_one , [1,1,1])

# pulmones segmentados y llenitos

print("segmenting lungs ")
segmented_lungs_fill = segment_lung_mask( image_res, True )

#segmented_lungs_emp = segment_lung_mask( image_res )

#diff =  segmented_lungs_fill
print("morphological cleanup")
#diff = morphological_ops( segmented_lungs_fill )

#image_res = np.array( image_res >  604  )


high_vals = segmented_lungs_fill == 0

image_res[ high_vals ] = 0


plt.imshow( image_res[80] , cmap = plt.cm.gray )
plt.show()



reduce_scan( image_res )



#image_res = image_res[ diff ]

#plt.imshow( image_res[80] , cmap = plt.cm.gray )
#plt.show()


#reduce_scan ( image_res  )


"""

"""   
patients = [ "0ddeb08e9c97227853422bd71a2a695e" ]

patient_one = load_scan( "../data/sample_images/" + patients[0] )

patient_one_pixels = get_pixels_hu( patient_one )


image_res , spacing = resample( patient_one_pixels, patient_one , [1,1,1])

print patient_one_pixels.shape

print image_res.shape
#plot_3d( image_res ) 

segmented_lungs = segment_lung_mask( image_res , True )



for new_dato  in get_augmented_data( segmented_aug ):
    print new_dato.shape
    print ("que bonito que bello")

#plt.imshow( segmented_aug[80] , cmap = plt.cm.gray )
#plt.show()

#plt.imshow( segmented_lungs[80] , cmap = plt.cm.gray )
#plt.show()

END COMMENT
"""
