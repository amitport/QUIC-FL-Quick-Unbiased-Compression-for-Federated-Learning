#!/usr/bin/env python3

import torch

##############################################################################
##############################################################################

class QSGDSender:

    
    def __init__(self, device='cpu'):

        self.device = device
        self.prng = torch.Generator(device=device)
                
            
    def compress(self, data):
        
        self.prng.manual_seed(data['seed'])
        
        vec_norm = torch.norm(data['vec'], p = float('inf'))
        vec_sign = 2 * (data['vec'] > 0) - 1

        if data['nlevels'] == 1:
            # qsgd is not really defined for one label, only rely on the sign
            return {'norm': vec_norm, 'sign': vec_sign,
                    'xi': torch.ones_like(data['vec']), 'nlevels': data['nlevels']}
        p = (data['nlevels'] - 1) * data['vec'].abs() / vec_norm
        floor_p = torch.floor(p)
        
        b = torch.bernoulli(p - floor_p, generator=self.prng)
        xi = floor_p + b
                                
        return {'norm': vec_norm, 'sign': vec_sign, 'xi': xi, 'nlevels': data['nlevels']}

##############################################################################
##############################################################################
       
class QSGDReceiver:

    
    def __init__(self, device='cpu'):

        self.device = device
                
            
    def decompress(self, data):
        if data['nlevels'] == 1:
            # qsgd is not really defined for one label, only rely on the sign
            return data['norm'] * data['sign'] * data['xi']
        return data['norm'] * data['sign'] * data['xi'] / (data['nlevels'] - 1)
   
##############################################################################
##############################################################################

