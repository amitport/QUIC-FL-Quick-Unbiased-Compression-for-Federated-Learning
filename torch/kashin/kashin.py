#!/usr/bin/env python3

import pathlib
path = pathlib.Path(__file__).parent.resolve()

import sys
sys.path.insert(0, str(path) + "/../")

import torch
import numpy as np

from hadamard import HadamardSender, HadamardReceiver
from stochastic_quantization import StochasticQuantizationSender, StochasticQuantizationReceiver

##############################################################################
##############################################################################

class KashinSender(HadamardSender, HadamardReceiver):

    
    def __init__(self, device, eta, delta, pad_threshold, niters):

        HadamardSender.__init__(self, device=device)
        HadamardReceiver.__init__(self, device=device)
        
        self.eta = eta
        self.delta=delta
        self.pad_threshold = pad_threshold
        self.niters = niters
        
    
    def kashin_padded_dim(self, dim, pad_threshold):
        
        padded_dim = dim
        
        if not dim & (dim-1) == 0:
            padded_dim = int(2**(np.ceil(np.log2(dim))))
            if dim / padded_dim > pad_threshold:
                padded_dim = 2*padded_dim
        else:
            padded_dim = 2*dim
            
        return padded_dim

    
    def kashin_coefficients(self, data):
        
        orig_vec = data["vec"].detach().clone()
        
        dim = data["vec"].numel()
        padded_dim = self.kashin_padded_dim(dim, self.pad_threshold)
               
        kashin_coefficients = torch.zeros(padded_dim, device=self.device)
        padded_x = torch.zeros(padded_dim, device=self.device)
        
        M = torch.norm(data["vec"]) / np.sqrt(self.delta * padded_dim)
        
        for i in range(self.niters):
    
            padded_x[:] = 0
            padded_x[:dim] = data["vec"]  
            padded_x = self.randomized_hadamard_transform(padded_x, data['rotation_seed'])
            
            b = padded_x   
            b_hat = torch.clamp(b, min=-M, max=M)
                    
            kashin_coefficients = kashin_coefficients + b_hat
            
            if i < self.niters - 1:
            
                b_hat = self.randomized_inverse_hadamard_transform(b_hat, data['rotation_seed'])
                data["vec"] = data["vec"] - b_hat[:dim]
                
                M = self.eta * M
            
            err = (orig_vec - self.randomized_inverse_hadamard_transform(kashin_coefficients.clone(), data['rotation_seed'])[:dim]).norm(2) / orig_vec.norm(2)
            if (err < self.err): 
                break

        return kashin_coefficients, dim              
    
##############################################################################
##############################################################################
##############################################################################
##############################################################################
        
class KashinStochasticQuantizationSender(KashinSender, StochasticQuantizationSender):

    
    def __init__(self, device='cpu', eta=0.9, delta=1.0, pad_threshold=0.85, niters=3):

        KashinSender.__init__(self, device=device, eta=eta, delta=delta, pad_threshold=pad_threshold, niters=niters)
        StochasticQuantizationSender.__init__(self, device=device)

           
    def compress(self, data):
        
        self.niters = data.get('niters', 2147483647)
        self.err = data.get('err', 0.000001)
        
        kashin_coefficients, dim = self.kashin_coefficients(data)
        sq_data = StochasticQuantizationSender.compress(self, {'vec': kashin_coefficients, 'nlevels': data['nlevels'], 'seed': data['seed'], 'step': data.get('step','standard')})
                        
        return {'data': sq_data, 'seed': data['seed'], 'rotation_seed': data['rotation_seed'], 'dim': dim}

##############################################################################
##############################################################################
       
class KashinStochasticQuantizationReceiver(HadamardReceiver, StochasticQuantizationReceiver):

    
    def __init__(self, device='cpu'):

        HadamardReceiver.__init__(self, device=device)
        StochasticQuantizationReceiver.__init__(self, device=device)
                
            
    def decompress(self, data):
                
        qhvec = StochasticQuantizationReceiver.decompress(self, data['data'])
        
        return self.randomized_inverse_hadamard_transform(qhvec, data['rotation_seed'])[:data['dim']] 
   
##############################################################################
############################################################################## 
