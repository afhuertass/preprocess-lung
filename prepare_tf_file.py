

import os
import tensorflow as tf
import tables
import numpy as np 

from tensorflow.python.lib.io.tf_record import TFRecordCompressionType


def txt_to_hdf5( input_path , output_path):

    with open(input_path, 'r' ) as f:
        ls = f.readlines()
    
    ls = [ x.strip() for x in ls ]

    hdf5_file = tables.open_file( output_path, mode = 'w' )
    atom_int = tables.Int16Atom( )

    array = hdf5_file.create_earray( hdf5_file.root , 'datal' , atom_int , ( 0, ) )
     
    for label in ls:
        
        array.append( [ label ] )


    hdf5_file.close()

    
def dataset_to_file( features , filename):

    writer = tf.python_io.TFRecordWriter(
        filename , 
        options = tf.python_io.TFRecordOptions(
            compression_type = TFRecordCompressionType.GZIP
        )
    )
    
    with writer:

        for feature in features:

            writer.write( tf.train.Example( features = tf.train.Features(
                    feature = feature
                )
             ).SerializeToString() )

    
            
def cancer_feature_fn( hdf5_file_data , hdf5_file_labels , indxs  ):

    hdf5_data = tables.open_file( hdf5_file_data, mode = 'r' )
    hdf5_labels = tables.open_file(hdf5_file_labels , mode = 'r')

    maxim = len( hdf5_data.root.datal )
    for indx in indxs :
        #print(scan.shape)
        print(indx)
        yield {
            'label' : tf.train.Feature(
                int64_list =  tf.train.Int64List( value = [ hdf5_labels.root.datal[indx ] ] )
            ) ,
            'images' : tf.train.Feature(
                float_list = tf.train.FloatList( value=  hdf5_data.root.datal[indx].flatten() )
            )
        }

def len_hdf5( hdf5_file ):

    file_o = tables.open_file( hdf5_file  , mode='r' )

    lenn = len( file_o.root.datal )

    return lenn

def chunk(l , n):
    
    for i in range(0, len(l), n):
        
        yield l[i:i + n]
    

hdf5_file = './pre_merged_regular.hdf5'
hdf5_labels = './fake_labels.hdf5'

train_path = './train_enlarged.pb2'

#dataset_to_file( cancer_feature_fn( hdf5_file , hdf5_labels ) , train_path )

txt_ruta = "/mnt/disks/grande/results/labels_enlarged.txt"
hdf5_out = "/mnt/disks/grande/results/labels_enlarged_fin.hdf5"
#txt_to_hdf5( txt_ruta , hdf5_out )


hdf5_file = "/mnt/disks/grande/results/enlarged-3.hdf5"
hdf5_labels = "/mnt/disks/grande/results/labels_regular_fin.hdf5"
train_path = "/mnt/disks/grande/results/training_entropy.pb2"













count_files = 1
hdf5_file = "/mnt/disks/grande/results/regular-3.hdf5"
hdf5_labels = "/mnt/disks/grande/results/labels_regular_fin.hdf5"

file_suffix = "/mnt/training_data/train_data/regular/train_"


lenght = len_hdf5( hdf5_file )
chunk_size = 35 # number of features per file
chunks =  chunk( range(0, lenght ) , chunk_size  ) 
for k in chunks:
    print(k)
    train_path = file_suffix + str(count_files ) + ".pb2"
    dataset_to_file( cancer_feature_fn( hdf5_file , hdf5_labels, k  )  , train_path )
    count_files = count_files + 1
    print(train_path)

#for train_path in train_paths:    
    #dataset_to_file( cancer_feature_fn( hdf5_file , hdf5_labels) , train_path  )



            
