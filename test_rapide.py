# Document pour des petits tests rapides

#%% Taille des données stockées
import pickle

filename = 'data_y/ID_noisy_data_lownoise' 
IDs = pickle.load(open(filename, 'rb'))

print(IDs.shape)

filename = 'data_y/ID_noisy_data' 
IDs = pickle.load(open(filename, 'rb'))

print(IDs.shape)

filename = 'dataNW2/dataNW2_targets' 
targets = pickle.load(open(filename, 'rb'))

print(targets.shape)

filename = 'dataNW2_version1/dataNW2_targets_version1' 
targets = pickle.load(open(filename, 'rb'))

print(targets.shape)

