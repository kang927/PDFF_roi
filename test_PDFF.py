# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:47:29 2017

@author: kang927
"""

import os
import numpy as np
import re
from xlrd import open_workbook
#%%

def read_dicom_series_MRI(directory, filepattern = "image_*"):
    """ 
    Reads a DICOM Series files in the given directory. 
    Only filesnames matching filepattern will be considered
    this version of read_dicom doesn't use intercept and slope sccaling since it is MRI data, not CT data
    that can concert to a standarize HU unit
    
    """
    
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : "+str(directory))
    print('\tRead Dicom',directory)
    lstFilesDCM = natsorted(glob.glob(os.path.join(directory, filepattern)))
    print('\tLength dicom series',len(lstFilesDCM) )
    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])
    # get the space sampling
    dx = np.float(RefDs.PixelSpacing[0])
    dy = np.float(RefDs.PixelSpacing[1])
    dz = np.float(RefDs.SliceThickness)
    dsampling = np.array([dx,dy,dz])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # transform the raw data to HU using Rescale slope and intercept and store it as array 
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
    return ArrayDicom, dsampling
    

    
#%%
def grab_subdirectory(dn, subdirectory_pattern='\\\\Lipoquant'):
    """
    return the subdirectories that match a specific pattern
    
    """
    dn_list =[]
    subdir_pattern = re.compile(subdirectory_pattern)
    for root, directories, files in os.walk(dn):
        for directory in directories: 
            current_subdirectory =  os.path.join(root, directory) 
            matchobj = subdir_pattern.search(current_subdirectory)
            if matchobj:
                dn_list.append(current_subdirectory)
    return dn_list


#%%
def extract_roi_coordinates_from_xls(pdff_fn):
    wb = open_workbook(pdff_fn)
    # all the data is stored in the worksheet named "Data"
    ws = wb.sheet_by_name('Data') 
    # header for various useful information needed to extract the roi coordinates
    header_pattern=['ROI Name','Series Description','Instance Number','Position']
    Npat = len(header_pattern)
    # this variable store the column location of where the above headers are in the excel file
    header_col_index = dict()
    # Header is recorded in row 21 in the excel worksheet
    header_row = 20
    for ii in range(Npat):
        p = re.compile(header_pattern[ii])
        for col in range(ws.ncols):
            value = (ws.cell(header_row,col).value)
            matchobj = p.match(value)
            if matchobj:
                header_col_index[ header_pattern[ii] ] = col


    # grab the rows that match FF%Lipoquant
    pat2 = re.compile('FF')
    roi_list=['1','2','3','4a','4b','5','6','7','8']
    roi_data_rowIndex = dict()
    for row in range(header_row+1,ws.nrows):
        value = (ws.cell(row,header_col_index['Series Description']).value)
        #print(value)
        # find the rows with Series Description starting with FF which record the ROI for PDFF palcement
        matchobj1 = pat2.search(value)
        if matchobj1:
            # once we found the appropriate series description, find the ROI name with liver segment (1 to 8) since the data
            # frame also record roi for MRS data
            for roi_name in roi_list:
                pat3 = re.compile(roi_name)
                value2 = (ws.cell(row,header_col_index['ROI Name']).value)
                value2 = str(value2)
                #print(value2)
                matchobj2 = pat3.search(value2)
                if matchobj2:
                    # store the rows corresponding to specific ROI label
                    roi_data_rowIndex[roi_name] = row

    # then we can assemble the center coordinate for the ROI 
    roi_coordinates=dict()
    for roi_name in roi_list:
        row = roi_data_rowIndex[roi_name]
        # axial slice location (aka z-coordinate) is stored in instance number
        value = (ws.cell(row,header_col_index['Instance Number']).value)
        z = np.float(value)
        # data frame heading Position(x,y) store the x,y coordinte of the ROI
        value = (ws.cell(row,header_col_index['Position']).value)
        # find all the numbers in the string
        pat4 = re.compile('\d+')
        tmp = pat4.findall(value)
        x = np.float(tmp[0])
        y = np.float(tmp[1])
        roi_coordinates[roi_name] = np.array((x,y,z))
    
    return roi_coordinates


#%% 

# assuming all the MRI series are stored in a directory, the following script will grab out the lipoquant series
mri_img_dn = 'C:/Users/kang927/Documents/deep_learning_liverseg/PDFF automated placement/PDFF_data/'
patient_list = listdir_fullpath(data_dn)

Nimg = len(patient_list)
lipoquant_list = [None]*Nimg
for ii in range(Nimg):
    matched_dn = grab_subdirectory(patient_list[ii], subdirectory_pattern='\\\\Lipoquant')
    if len(matched_dn)!=1:
        print( patient_list[ii] + " has %d lipoquant series"%len(matched_dn) )
    
    lipoquant_list[ii] = matched_dn

print(lipoquant_list)
#%% then we will read in the different echo series 
img, dsampling =  read_dicom_series(fn2,filepattern='*.dcm')

#%%
Nslice = img.shape[2]
nx = img.shape[0]
ny = img.shape[1]
n_echoes = 6
Nz = np.int( Nslice/n_echoes ) # six echos
img2 = np.reshape(img,[nx,ny,Nz,n_echoes])


#%%
Nimg = img.shape[2]
for s in range(0,Nz,5):
    imshow(img2[:,:,s,3],img2[:,:,s,4],img2[:,:,s,5])

    

#%% now we will need to parse the excel file to get the coordinates and label for each ROI placement in the PDFF analysis
pdff_analysis_dn = 'C:/Users/kang927/Documents/deep_learning_liverseg/PDFF automated placement/PDFF_analysis'
pdff_analysis_list = listdir_fullpath(pdff_analysis_dn)
ii = 2
pdff_fn = pdff_analysis_list[ii]
print(pdff_fn)
roi_coordinate = extract_roi_coordinates_from_xls(pdff_fn)
    
