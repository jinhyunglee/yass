import os
import numpy as np
import tensorflow as tf
from yass.preprocess.score import get_score_pca
from yass.neuralnetwork.nndetector import NeuralNetDetector
from yass.geometry import n_steps_neigh_channels


def make_phy_files(spike_train, templates, CONFIG):
    
    # commonly used parameters
    n_spikes = spike_train.shape[0]
    n_channels = CONFIG.recordings.n_channels
    n_templates = templates.shape[2]
    root_folder = CONFIG.data.root_folder 
    
    # commonly used data
    
    # get main channel for each template
    templates_mainc = np.argmax(np.max(templates, axis=1),axis=0)
    
    # main channel for each spike based on templates_mainc
    spikes_mainc = np.zeros(n_spikes, 'int32')
    for j in range(n_spikes):
        spikes_mainc[j] = templates_mainc[spike_train[j,1]]
    
    # number of neighbors to consider
    neighbors = n_steps_neigh_channels(
        CONFIG.neighChannels, 2)
    nneigh = np.max(np.sum(neighbors,0))
    # ordered neighboring channels w.r.t. each channel
    c_idx = np.zeros((n_channels, nneigh), 'int32')
    for c in range(n_channels):
        c_idx[c] = np.argsort(np.sum(np.square(CONFIG.geom - CONFIG.geom[c]), axis = 1))[:nneigh]
  
    # get score 
    score = get_score(spike_train[:,0], spikes_mainc, c_idx, CONFIG)
    
    # get templates on reduced dimension space
    templates_low_dim = get_templates_low_dim(templates, CONFIG)
    
    
    # Making param.py 
    file = open(os.path.join(root_folder,'tmp/params.py'),'w') 
    file.write('dat_path = '+"'"+os.path.join(root_folder,CONFIG.data.recordings)+"'\n") 
    file.write('n_channels_dat = '+str(n_channels)+"\n") 
    file.write('dtype = '+"'"+CONFIG.recordings.dtype+"'"+"\n") 
    file.write('offset = 0'+"\n") 
    file.write('sample_rate = '+str(CONFIG.recordings.sampling_rate)+"\n")
    file.write('hp_filtered = False'+"\n")
    file.close() 
    
    # amplitudes.npy
    np.save(os.path.join(root_folder,'tmp/amplitudes.npy'), 
            np.ones(n_spikes))

    # channel_map.npy
    np.save(os.path.join(root_folder,'tmp/channel_map.npy'), 
            np.arange(n_channels))
    
    # channel_positions.npy
    np.save(os.path.join(root_folder,'tmp/channel_positions.npy'), 
            CONFIG.geom)
    
    # pc_features.npy
    np.save(os.path.join(root_folder,'tmp/pc_features.npy'), 
            score)
    
    # pc_feature_ind.npy
    pc_feature_ind = np.zeros((n_templates, nneigh), 'int32')
    for k in range(n_templates):
        pc_feature_ind[k] = c_idx[templates_mainc[k]]
    # save it
    np.save(os.path.join(root_folder,'tmp/pc_feature_ind.npy'), 
            pc_feature_ind)
    
    # similar_templates.npy
    similar_templates = np.corrcoef(np.reshape(templates,[-1, n_templates]).T)
    np.save(os.path.join(root_folder,'tmp/similar_templates.npy'),
            similar_templates)
    
    # spike_templates.npy and spike_times.npy
    idx_sort = np.argsort(spike_train[:,0])
    spike_train = spike_train[idx_sort]
    np.save(os.path.join(root_folder,'tmp/spike_templates.npy'), 
            spike_train[:, 1])
    np.save(os.path.join(root_folder,'tmp/spike_times.npy'), 
            spike_train[:, 0])
      
    # template_feature_ind.npy
    k_neigh = np.min((5, n_templates))
    template_feature_ind = np.zeros((n_templates , k_neigh), 'int32')
    for k in range(n_templates):
        template_feature_ind[k] = np.argsort(-similar_templates[k])[:k_neigh]
    np.save(os.path.join(root_folder,'tmp/template_feature_ind.npy'), 
            template_feature_ind)

    # template_features.npy
    template_features = np.zeros((n_spikes, k_neigh))
    for j in range(n_spikes):
        ch_idx = c_idx[spikes_mainc[j]]
        kk = spike_train[j, 1]
        for k in range(k_neigh):
            template_features[j] = np.sum(
                np.multiply(score[j].T,
                templates_low_dim[ch_idx][:,:,
                template_feature_ind[kk,k]]))
    np.save(os.path.join(root_folder,'tmp/template_features.npy'), 
            template_features)
    
    # templates.npy
    np.save(os.path.join(root_folder,'tmp/templates.npy'), 
            np.transpose(templates, [2, 1, 0]))
    
    # templates_ind.npy
    templates_ind = np.zeros((n_templates, n_channels), 'int32')
    for k in range(n_templates):
        templates_ind[k] = np.arange(n_channels)    
    np.save(os.path.join(root_folder,'tmp/templates_ind.npy'), 
            templates_ind)         

    # whitening_mat.npy and whitening_mat_inv.npy
    np.save(os.path.join(root_folder,'tmp/whitening_mat.npy'), 
            np.eye(n_channels))
    np.save(os.path.join(root_folder,'tmp/whitening_mat_inv.npy'), 
            np.eye(n_channels))
    
    
def get_score(spike_time, spikes_mainc, c_idx, CONFIG):
    
    C, nneigh = c_idx.shape
    neighbors_new = np.zeros((C,C),'bool')
    for c in range(C):
        neighbors_new[c, c_idx[c]] = 1
    
    nnd = NeuralNetDetector(CONFIG.neural_network_detector.filename,
                            CONFIG.neural_network_autoencoder.filename)
    with tf.Session() as sess:
        nnd.saver_ae.restore(sess, nnd.path_to_ae_model)
        rot  = sess.run(nnd.W_ae)
    rot_expanded = np.reshape(np.matlib.repmat(rot,1,C),
                    [rot.shape[0],rot.shape[1],C])

    score = get_score_pca(np.hstack((spike_time[:,np.newaxis], 
                                 spikes_mainc[:,np.newaxis])), 
                      rot_expanded, 
                      neighbors_new, 
                      CONFIG.geom, 
                      CONFIG.batch_size,
                      CONFIG.BUFF,
                      CONFIG.nBatches,
                      os.path.join(CONFIG.data.root_folder, 'tmp', 'standarized.bin'),
                      CONFIG.scaleToSave)
    
    return score

def get_templates_low_dim(templates, CONFIG):
    
    C, R2, K = templates.shape
    R = int((R2 - 1)/2)
    
    nnd = NeuralNetDetector(CONFIG.neural_network_detector.filename,
                            CONFIG.neural_network_autoencoder.filename)
    with tf.Session() as sess:
        nnd.saver_ae.restore(sess, nnd.path_to_ae_model)
        rot  = sess.run(nnd.W_ae)
        
    templates_low_dim = np.zeros((C,rot.shape[1],K))
    for k in range(K):
        templates_low_dim[:,:,k] = np.matmul(templates[:,R:(3*R+1),k],rot)

    return templates_low_dim