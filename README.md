# ouroboros_neurofeedback_api

## Ouroboros Neurofeedback API: Computing NF scores from observed and predicted human brain activity (EEG, fMRI)

Version: 16th May 2024

Project developed by Maël Donoso, Ph.D. in Cognitive and Computational Neuroscience. Affiliations: Ouroboros Neurotechnologies (https://ouroboros-neurotechnologies.com/), Institut Lémanique du Cerveau (https://institut-cerveau.ch/), Policlinique Ostéopathique de Lausanne (https://policlinique-osteopathique-lausanne.ch/). 

### Abstract

**This project builds an API (Django, Docker) around Neuropolis. Our objective is to compute neurofeedback (NF) scores, whether directly on the Electroencephalography (EEG) activity, or indirectly on the predicted functional Magnetic Resonance Imaging (fMRI) activity. For this task, we use a series of machine learning models trained in the Neuropolis project, including linear regression, k-nearest neighbors, decision trees, random forests, and support vector machines.**


### Structure and Environment

This project consists of a Django application and two Notebooks, to be run in order:

1. Models
2. Requests

The Conda environment for this project can be recreated with the .yml file given in the project directory (ouroboros_neurofeedback_api/ouroboros-neurofeedback-api.yml). 


### Running the API

1. Download the EEG-fMRI NF dataset from OpenNeuro (https://openneuro.org/datasets/ds002336/versions/2.0.2), and run the Neuropolis project (https://github.com/maeldonoso/neuropolis). 
2. Run the first Notebook to copy the Neuropolis models into the appropriate directory (ouroboros_neurofeedback_api/models/). 
3. Run the API, either directly within the Conda environment (command: *python manage.py runserver*), or as a Docker container (command: *docker-compose up*). 
4. Run the second Notebook to send requests to the two endpoints of the API. 

By design, no data is stored in the Django application. Nevertheless, a PostgreSQL database has been included, in order to allow for experimentation. In its current form, the Django application is not suitable for production. 


### Documentation

#### Endpoint 1: **POST /neurofeedback/observed_brain_activity**

*Request:*
- ***eeg_data***: A list of floats corresponding to the preprocessed EEG data points. 
- ***sampling_frequency***: A float or integer indicating the sampling frequency of the preprocessed EEG data. 
- ***frequency_bands***: A list of sublists, each sublist indicating the name of the frequency band, the lower limit of the band, and the upper limit of the band. 
- ***training_frequency_band***: A string indicating the frequency band to be used for neurofeedback training. 
- ***regulation***: A string indicating the type of neurofeedback regulation. The options are: "upregulation", "downregulation". 
- ***threshold***: A float or integer indicating the neurofeedback threshold. 

*Response:*
- ***bandpowers***: A dictionary of floats indicating the bandpower of each frequency band. 
- ***bandpower_value***: A float indicating the bandpower of the training frequency band. 
- ***feedback***: An integer, 0 or 1, indicating a negative or positive feedback. 
- ***score***: A float indicating the neurofeedback score. 


#### Endpoint 2: **POST /neurofeedback/predicted_brain_activity**

*Request:*
- ***eeg_data***: A list of 63 sublists, each sublist containing floats and corresponding to the preprocessed EEG data points of a particular EEG channel. 
- ***trained_model***: A string indicating the trained machine learning model to use for the prediction. The options are: "LinearRegression", "Ridge", "Lasso", "KNeighborsRegressor", "DecisionTreeRegressor", "RandomForestRegressor", "SVR". 
- ***training_voxel***: A string indicating the predicted voxel activity to be used for neurofeedback training. The options are: "pgACC", "vmPFC", "Right FPC 1", "Right FPC 2", "Right FPC 3", "dACC 1", "dACC 2", "Left post-LPC", "Right post-LPC", "Left PMC 1", "Left PMC 2", "Left dorsal striatum", "Right dorsal striatum", "Left FPC", "Right FPC", "Left mid-LPC", "Right ventral striatum". 
- ***regulation***: A string indicating the type of neurofeedback regulation. The options are: "upregulation", "downregulation". 
- ***threshold***: A float or integer indicating the neurofeedback threshold. 

*Response:*
- ***voxels***: A dictionary of floats indicating the predicted value of each voxel. 
- ***voxel_value***: A float indicating the predicted value of the training voxel. 
- ***feedback***: An integer, 0 or 1, indicating a negative or positive feedback. 
- ***score***: A float indicating the neurofeedback score. 
