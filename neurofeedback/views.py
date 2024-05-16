from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
import numpy as np
import yasa
import pickle

@api_view(['POST'])
def observed_brain_activity(request):
    
    # Retrieve the EEG data and parameters. 
    eeg_data =                  request.data.get('eeg_data')
    sampling_frequency =        request.data.get('sampling_frequency')
    frequency_bands =           request.data.get('frequency_bands')
    training_frequency_band =   request.data.get('training_frequency_band')
    regulation =                request.data.get('regulation')
    threshold =                 request.data.get('threshold')

    # Compute the bandpowers using the YASA library. 
    bandpowers = yasa.bandpower(np.array(eeg_data), 
                                sf = sampling_frequency, 
                                bands = frequency_bands, 
                                win_sec = 2)

    # Retrieve the bandpower of interest, and compute the feedback and score. 
    bandpower = bandpowers.loc[:, training_frequency_band].values[0]
    if regulation == 'upregulation':
        feedback = int(bandpower > threshold)
        score = bandpower - threshold
    elif regulation == 'downregulation':
        feedback = int(bandpower < threshold)
        score = threshold - bandpower

    # Return the response. 
    eeg_dict = {'bandpowers' : dict(bandpowers.iloc[0, 0:6]), 
                'bandpower_value' : bandpower, 
                'feedback' : feedback, 
                'score' : score}
    data = json.dumps(eeg_dict)
    
    return Response(data)

@api_view(['POST'])
def predicted_brain_activity(request):
    
    # Retrieve the EEG data and parameters. 
    eeg_data =                  request.data.get('eeg_data')
    trained_model =             request.data.get('trained_model')
    training_voxel =            request.data.get('training_voxel')
    regulation =                request.data.get('regulation')
    threshold =                 request.data.get('threshold')

    # Define the sampling frequency and frequency bands used to train the models. 
    sampling_frequency = 200
    frequency_bands = [(1, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 16, 'Sigma'), (16, 30, 'Beta'), (30, 40, 'Gamma')]

    # Reconstruct the NumPy array from the list of sublists. 
    np_eeg_data = np.array(eeg_data)

    # Compute the bandpowers using the YASA library. 
    bandpowers_sequence = np.zeros((63, 6, 6)) # Dimensions: channels, frequency bands, scans. 
    scan_duration = 2 * sampling_frequency
    for scan in range(6):
        bandpowers = yasa.bandpower(np_eeg_data[:, (scan * scan_duration):(scan * scan_duration) + scan_duration], 
                                    sf = sampling_frequency, 
                                    bands = frequency_bands, 
                                    win_sec = 2)
        bandpowers_sequence[:, :, scan] = bandpowers.iloc[:, 0:6]
    
    # Normalize the bandpowers. 
    bandpowers_mean = np.mean(bandpowers_sequence, axis = 2)
    bandpowers_std = np.std(bandpowers_sequence, axis = 2)
    bandpowers_normalized = (bandpowers_sequence - bandpowers_mean[:, :, np.newaxis]) / bandpowers_std[:, :, np.newaxis]

    # Define the general features array by stacking all the channels. 
    X_basis = np.vstack(bandpowers_normalized).transpose() # Dimensions: scans, channels * frequency bands. 

    # Define the particular features array by stacking all the scans. 
    X_sequence = np.zeros((1, np.prod(X_basis.shape))) # Dimensions: 1, channels * frequency bands * scans. 
    X_sequence[0, :] = X_basis.ravel()

    # Load the selected PredictorBrain. 
    models_path = 'models/'
    if trained_model == 'LinearRegression':
        with open(models_path + 'model_lr_pb.p', 'rb') as file:
            PredictorBrain = pickle.load(file)
    elif trained_model == 'Ridge':
        with open(models_path + 'model_ridge_pb.p', 'rb') as file:
            PredictorBrain = pickle.load(file)
    elif trained_model == 'Lasso':
        with open(models_path + 'model_lasso_pb.p', 'rb') as file:
            PredictorBrain = pickle.load(file)
    elif trained_model == 'KNeighborsRegressor':
        with open(models_path + 'model_knn_pb.p', 'rb') as file:
            PredictorBrain = pickle.load(file)
    elif trained_model == 'DecisionTreeRegressor':
        with open(models_path + 'model_dt_pb.p', 'rb') as file:
            PredictorBrain = pickle.load(file)
    elif trained_model == 'RandomForestRegressor':
        with open(models_path + 'model_rf_pb.p', 'rb') as file:
            PredictorBrain = pickle.load(file)
    elif trained_model == 'SVR':
        with open(models_path + 'model_svm_pb.p', 'rb') as file:
            PredictorBrain = pickle.load(file)

    # Predict the fMRI activity. 
    Y_pred = PredictorBrain.predict(X_sequence)

    # Retrieve the voxel of interest, and compute the feedback and score. 
    voxels_labels = ['pgACC', 'vmPFC', 'Right FPC 1', 'Right FPC 2', 'Right FPC 3', 
                     'dACC 1', 'dACC 2', 'Left post-LPC', 'Right post-LPC', 'Left PMC 1', 
                     'Left PMC 2', 'Left dorsal striatum', 'Right dorsal striatum', 'Left FPC', 'Right FPC', 
                     'Left mid-LPC', 'Right ventral striatum']
    voxel_value = Y_pred[0, voxels_labels.index(training_voxel)]
    if regulation == 'upregulation':
        feedback = int(voxel_value > threshold)
        score = voxel_value - threshold
    elif regulation == 'downregulation':
        feedback = int(voxel_value < threshold)
        score = threshold - voxel_value

    # Return the response. 
    fmri_dict = {'voxels' : Y_pred.tolist()[0], 
                 'voxel_value' : voxel_value, 
                 'feedback' : feedback, 
                 'score' : score}
    data = json.dumps(fmri_dict)
    
    return Response(data)