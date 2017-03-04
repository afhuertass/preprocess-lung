

import numpy as np

import tables


def merge( file1 , file2 , out) :

    #merge dos
    print("merging")
    hdf5_out = tables.open_file( out, mode = "w")
    atom_int = tables.Int16Atom( )
    array_out = hdf5_out.create_earray( hdf5_out.root , 'datal' , atom_int , ( 0, 150,150,150  ) )

    
    hdf5_file1 = tables.open_file( file1 , mode='r')
    hdf5_file2 = tables.open_file( file2 , mode='r' )

    
    #data1 = hdf5_data = hdf5_file1.root.datal[:]
    #data2 = hdf5_data = hdf5_file2.root.datal[:]
    shape = (1,150,150,150)
    #print ( data1[0].shape )
    files_count = 0 
    for data in hdf5_file1.root.datael:

        array_out.append( np.reshape( data, shape ) )
        files_count = files_count + 1 
        
    for data in   hdf5_file2.root.datael :
        array_out.append( np.reshape( data , shape  ) )
        files_count = files_count + 1 

    hdf5_file1.close()
    hdf5_file2.close()
    hdf5_out.close()

    print(files_count)

def count_scans( hdf5_path  ):

    hdf5_file = tables.open_file(hdf5_path , mode='r')

    scans_n = len( hdf5_file.root.datal)

    print(scans_n)
    hdf5_file.close()
    
file1 = '/mnt/lung_data/pre/entropy_large.hdf5'
file2 = '/mnt/lung_data/pre/entropy_large-2.hdf5'
out = '/mnt/lung_data/pre/merged/entropy_large.hdf5'


#file1 = './enlarged.hdf5'
#file2 = './enlarged-2.hdf5'
#out = './merged.hdf5'

#merge(file1, file2, out)
count_scans(  './enlarged.hdf5' )
