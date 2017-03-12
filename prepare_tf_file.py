

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

            
def cancer_feature_fn( hdf5_file_data , hdf5_file_labels  ):

    hdf5_data = tables.open_file( hdf5_file_data, mode = 'r' )
    hdf5_labels = tables.open_file(hdf5_file_labels , mode = 'r')

    maxim = len( hdf5_da.root.datal )
    for indx in range(0, maxim):
        print(scan.shape)
        
        yield {
            'label' : tf.train.Feature(
                int64_list =  tf.train.Int64List( value = [ hdf5_labels[indx ] ] )
            ) ,
            'images' : tf.train.Feature(
                float_list = tf.train.FloatList( value=  hdf5_data[indx].flatten() )
            )
        }



hdf5_file = './pre_merged_regular.hdf5'
hdf5_labels = './fake_labels.hdf5'

train_path = './train.pb2'

#dataset_to_file( cancer_feature_fn( hdf5_file , hdf5_labels ) , train_path )

txt_ruta = "/mnt/disks/grande/results/labels_enlarged.txt"
hdf5_out = "/mnt/disks/grande/results/labels_enlarged_fin.hdf5"
#txt_to_hdf5( txt_ruta , hdf5_out )


hdf5_file = "/mnt/disks/grande/results/enlarged-3.hdf5"
hdf5_labels = "/mnt/disks/grande/results/labels_enlarged_fin.hdf5"

train_path = "./train_enlarged"

dataset_to_file( cancer_feature_fn( hdf5_file , hdf5_labels) , train_path  )



            
