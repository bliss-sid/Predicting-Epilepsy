"""
    Extract Features from EEG time series
    -------------------------------------
    Peak in Band (PIB) features are extracted from EEG time series data.
    Six PIB per minute are calculated. This results in 960 features for
    10 minute of EEG recording in 16 channels.
    
"""

import numpy as np
from glob import glob
import scipy.io as _scio
import scipy.signal as _scsg
import multiprocessing as mp
from sklearn import tree
import cPickle
import csv



def readdata(matfile):
    
    try:
        data = _scio.loadmat(matfile)
    except IOError:
        raise IOError("Error opening MATLAB matrix " + matfile)
            
    # Get the length of data for each channel
    data_key = ""
    for key in data.iterkeys(): #keys are data,data_length_sec,sampling_frequency,channels,sequence
        if type(data[key]) == np.ndarray: 
            data_key = key  

    # Copy data for the channel and return it back
    eeg_record = np.copy(data[data_key]["data"][0,0])
    eeg_record_length_min = int(data[data_key]["data_length_sec"]/60.)
    eeg_sampling_frequency = float(data[data_key]["sampling_frequency"])
    del data
        
    return (eeg_record, eeg_record_length_min, eeg_sampling_frequency)


def determine_pib(X, eeg_sampling):
    
    freq, Pxx = _scsg.welch(X, fs = eeg_sampling, noverlap = None, scaling = "density")

    pib = np.zeros(6)    
            
    # delta band (0.1-4hz)
    ipos = (freq >= 0.1) & (freq < 4.0)
    pib[0] = np.trapz(Pxx[ipos], freq[ipos]) 
    
    # theta band (4-8hz)
    ipos = (freq >= 4.0) & (freq < 8.0)
    pib[1] = np.trapz(Pxx[ipos], freq[ipos]) 
        
    # alpha band (8-12hz) 
    ipos = (freq >= 8.0) & (freq < 12.0)
    pib[2] = np.trapz(Pxx[ipos], freq[ipos]) 
        
    # beta band (12-30hz) 
    ipos = (freq >= 12.0) & (freq < 30.0)
    pib[3] = np.trapz(Pxx[ipos], freq[ipos]) 
        
    # low-gamma band (30-70hz)
    ipos = (freq >= 30.0) & (freq < 70.0)
    pib[4] = np.trapz(Pxx[ipos], freq[ipos]) 
         
    # high-gamma band (70-180hz)
    ipos = (freq >= 70.0) & (freq < 180.0)
    pib[5] = np.trapz(Pxx[ipos], freq[ipos]) 
    
    return pib        
                    

def worker(eegfile):
    
    #print "Processing file : ", eegfile
        
    # Read the MATLAB binary file and extract data
    eeg_record, egg_length_min, eeg_sampling = readdata(eegfile)

    n_channels = eeg_record.shape[0]

    feature_arr = np.zeros((1, 6 * n_channels * egg_length_min))
    
    # Extract features. Features are sum of power in power spectrum of
    # time series. Summed power in 6 bands ==> delta(0.1-4hz), theta(4-8hz),
    # alpha(8-12hz),beta(12-30hz), low-gamma(30-70hz), high-gamma(70-180hz)
    # This is done for 1 minute segment of the time series.
    start, stop = 0, 6
    for channel in range(n_channels):
        time_chunks = np.array_split(eeg_record[channel], egg_length_min)
        
        # iterate through the chunks and calculate summed spectral power of 
        # the band. This will give 6 features/1 minute for 1 channel or 96/minute 
        # for 16 channels. This will give us 960 features for 1 dataset
        for chunk in time_chunks:
            feature_arr[0,start:stop] = determine_pib(chunk, eeg_sampling)
            start, stop = stop, stop + 6
    
    return [feature_arr,eegfile]
    
            
def main(args):
    """
    Main function
    
    Parameters
    ----------
    args : list
        List of EEG recording file names
        
    feature_arr : numpy array
        Consolidated array of features for the training set
    """
    # Determine number of input files
    nfiles = len(args)
    
    # Create an array to store extracted features
    feature_arr = np.zeros((nfiles, 1440))
    
    # Number of cpus
    n_cpus = mp.cpu_count()
    
    # Create a pool of worker functions
    pool = mp.Pool(n_cpus)
    result = pool.map(worker, args)
    #print(result.shape)
    result=sorted(result,key=lambda x: x[1]) #sorting by the name of eeg file
    # Feature array
    for i in range(len(result)):
        nrecs = result[i][0].shape[1]
        print(result[i][1])
        feature_arr[i,:nrecs] = result[i][0]
        
    del result
    
    return feature_arr
        

if __name__ == "__main__":
    preictal_files = glob("D:/Epilepsy/Patient_2/*preictal_segment*.mat")
    preictal_files=sorted(preictal_files)
    print(preictal_files[0])
    feature_arr_1 = main(preictal_files)
    
    pre_rows=np.shape(feature_arr_1)[0]
    a1=[1]*pre_rows
    
    interictal_files = glob("D:/Epilepsy/Patient_2/*interictal_segment*.mat")
    feature_arr_2 = main(interictal_files)
    
    inter_rows=np.shape(feature_arr_2)[0]
    a2=[0]*inter_rows
    
    X = np.concatenate((feature_arr_1, feature_arr_2),axis=0)
    Q = np.append(a1, a2)
    
    
     #Predict functions
     
    res = tree.DecisionTreeClassifier()
    res= res.fit(X,Q)
    
    test_files = sorted(glob("D:/Epilepsy/Patient_2/*test_segment*.mat"))
    test_files=sorted(test_files)
    print(test_files[0])
    feature_arr_3 = main(test_files)
    test_rows=np.shape(feature_arr_3)[0]
     
    final_result=res.predict(feature_arr_3)
     
    with open('D:\\Epilepsy\\result.csv', 'a') as csvfile:
        fieldnames = ['clip', 'preictal']
        writer = csv.DictWriter(csvfile, lineterminator='\n', fieldnames=fieldnames)
    
        
        for i in range(1,test_rows+1):
            addr='Patient_2_test_segment_0'+str(str(i).zfill(3))+'.mat'
            writer.writerow({'clip': addr, 'preictal': final_result[i-1]})
     
     
     