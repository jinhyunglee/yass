import logging
import os
import datetime as dt
import numpy as np

from ..geometry import n_steps_neigh_channels


class Deconvolution(object):

    def __init__(self, config, templates, spike_index, filename='wrec.bin'):

        self.config = config
        self.templates = templates
        self.spike_index = spike_index
        self.path_to_file = os.path.join(
            self.config.data.root_folder, 'tmp', filename)

        self.logger = logging.getLogger(__name__)

    def openWFile(self, opt):
        self.WFile = open(self.path_to_file, opt)

    def closeWFile(self):
        self.WFile.close()

    def fullMPMU(self):

        start_time = dt.datetime.now()

        self.openWFile('rb')

        self.logger.debug('wfile is {} {}'.format(self.path_to_file,
                                                  os.path.getsize(self.path_to_file)))
        nBatches = self.config.nBatches
        flattenedLength = 2*(self.config.batch_size
                             + 2*self.config.BUFF)*self.config.recordings.n_channels

        neighchan = n_steps_neigh_channels(self.config.neighChannels, steps = 3)
        R2, C, K = self.templates.shape
        R = int((R2-1)/2)
        shift = 3  # int(R/2)

        nrank = self.config.deconvolution.rank
        lam = self.config.deconvolution.lam
        Th = self.config.deconvolution.threshold

        amps = np.max(np.abs(self.templates), axis=0)
        amps_max = np.max(amps, axis=0)
        k_ordered= np.argsort(amps_max)[::-1]

        templatesMask = np.zeros((K, C), 'bool')
        for k in range(K):
            templatesMask[k] = amps[:, k] > amps_max[k]*0.5

        W_all, U_all, mu_all = decompose_dWU(self.templates, nrank)

        spiketime_all = np.zeros(0, 'int32')
        assignment_all = np.zeros(0, 'int32')

        for i in range(nBatches):
            self.logger.info("batch {}/{}".format(i+1, nBatches))
            self.WFile.seek(flattenedLength*i)
            wrec = self.WFile.read(flattenedLength)
            wrec = np.fromstring(wrec, dtype='int16')
            wrec = np.reshape(wrec, (-1, self.config.recordings.n_channels))
            wrec = wrec.astype('float32')/self.config.scaleToSave

            idx_batch = np.logical_and(self.spike_index[:,0] > self.config.batch_size*i, 
                                       self.spike_index[:,0] < self.config.batch_size*(i+1))
            spike_index_batch = self.spike_index[idx_batch]
            spike_index_batch[:,0] = spike_index_batch[:,0] - self.config.batch_size*i + self.config.BUFF

        
        
        
        
        
        
        
        
        
        self.openWFile('rb')

        self.logger.debug('wfile is {} {}'.format(self.path_to_file,
                                                  os.path.getsize(self.path_to_file)))
        nBatches = self.config.nBatches
        flattenedLength = 2*(self.config.batch_size
                             + 2*self.config.BUFF)*self.config.recordings.n_channels

        neighchan = n_steps_neigh_channels(self.config.neighChannels, steps = 3)
        shift = 3  # int(R/2)
        R2, C, K = self.templates.shape
        R = int((R2-1)/2)
        nrank = self.config.deconvolution.rank
        lam = self.config.deconvolution.lam
        Th = self.config.deconvolution.threshold
        iter_max = 1

        amps = np.max(np.abs(self.templates), axis=0)
        amps_max = np.max(amps, axis=0)
        k_ordered= np.argsort(amps_max)[::-1]

        templatesMask = np.zeros((K, C), 'bool')
        for k in range(K):
            templatesMask[k] = amps[:, k] > amps_max[k]*0.7

        W_all, U_all, mu_all = decompose_dWU(self.templates, nrank)

        spiketime_all = np.zeros(0, 'int32')
        assignment_all = np.zeros(0, 'int32')

        for i in range(nBatches):
            self.logger.info("batch {}/{}".format(i+1, nBatches))
            self.WFile.seek(flattenedLength*i)
            wrec = self.WFile.read(flattenedLength)
            wrec = np.fromstring(wrec, dtype='int16')
            wrec = np.reshape(wrec, (-1, self.config.recordings.n_channels))
            wrec = wrec.astype('float32')/self.config.scaleToSave

            idx_batch = np.logical_and(self.spike_index[:,0] > self.config.batch_size*i, 
                                       self.spike_index[:,0] < self.config.batch_size*(i+1))
            spike_index_batch = self.spike_index[idx_batch]
            spike_index_batch[:,0] = spike_index_batch[:,0] - self.config.batch_size*i + self.config.BUFF
            
            for kk in range(K):
                k = k_ordered[kk]
                cs = np.where(templatesMask[k])[0]

                spt_c = np.zeros(0,'int32')
                for c in cs:
                    idx_c = spike_index_batch[:,1] == c
                    spt_c = np.hstack((spt_c,spike_index_batch[idx_c, 0]))


                tt = self.templates[:,:,k]
                mu = mu_all[k]
                W = W_all[:,k,:]
                U = U_all[:,k,:]

                nc = spt_c.shape[0]
                wf_projs = np.zeros((nc, 2*(R+shift)+1, nrank))
                for s in range(-R-shift,R+shift):
                    wf_projs[:,s+R+shift,:] = np.matmul(rec[spt_c+s],U)

                obj = np.zeros((nc, 2*shift+1))    
                for j in range(2*shift+1):
                    obj[:,j] = np.sum(np.multiply(wf_projs[:,j:j+2*R+1],W[np.newaxis,:,:]),axis=(1,2))


                scale = np.abs((obj-mu)/np.sqrt(mu/lam)) - 3
                scale = np.minimum(np.maximum(scale,0),1)
                Ci = np.multiply(np.square(obj),(1-scale))

                mX = np.max(Ci, axis=1)
                st = np.argmax(Ci, axis=1)
                xx = obj[np.arange(nc),st]/mu

                idx_keep = mX > Th*Th
                st = st[idx_keep] + spt_c[idx_keep] - shift
                xx = xx[idx_keep]

                for j in range(st.shape[0]):
                    rec[st[j]-R:st[j]+R+1] -= xx[j]*tt

                spiketime_all = np.concatenate((spiketime_all, st + i*self.config.batch_size - self.config.BUFF))
                assignment_all = np.concatenate((assignment_all, np.ones(st.shape[0],'int32')*k))

        self.closeWFile()

        current_time = dt.datetime.now()
        self.logger.info("Deconvolution done in {0} seconds.".format(
                         (current_time-start_time).seconds))

        return np.concatenate((spiketime_all[:, np.newaxis], assignment_all[:, np.newaxis]), axis=1)

def decompose_dWU(templates, nrank):
    R, C, K = templates.shape
    W = np.zeros((R, nrank, K), 'float32')
    U = np.zeros((C, nrank, K), 'float32')
    mu = np.zeros((K, 1), 'float32')

    templates[np.isnan(templates)] = 0
    for k in range(K):
        W[:, :, k], U[:, :, k], mu[k] = get_svds(templates[:, :, k], nrank)

    U = np.transpose(U, [0, 2, 1])
    W = np.transpose(W, [0, 2, 1])

    U[np.isnan(U)] = 0

    return W, U, mu


def get_svds(template, nrank):
    Wall, S_temp, Uall = np.linalg.svd(template)
    imax = np.argmax(np.abs(Wall[:, 0]))
    ss = np.sign(Wall[imax, 1])
    Uall[0, :] = -Uall[0, :]*ss
    Wall[:, 0] = -Wall[:, 0]*ss

    Sv = np.zeros((Wall.shape[0], Uall.shape[0]))
    nn = np.min((Wall.shape[0], Uall.shape[0]))
    Sv[:nn, :nn] = np.diag(S_temp)

    Wall = np.matmul(Wall, Sv)

    mu = np.sqrt(np.sum(np.square(np.diag(Sv)[:nrank])))
    Wall = Wall/mu

    W = Wall[:, :nrank]
    U = (Uall.T)[:, :nrank]

    return W, U, mu
