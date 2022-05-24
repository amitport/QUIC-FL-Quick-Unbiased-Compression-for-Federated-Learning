#!/usr/bin/env python3

import pathlib
path = pathlib.Path(__file__).parent.resolve()

import sys
sys.path.insert(0, str(path) + "/../")

import torch
import numpy as np

from hadamard import HadamardSender, HadamardReceiver
from .normal_quantization_levels import gen_normal_centoirds_and_boundries, centroid_lookup_table

##############################################################################
##############################################################################

class EdenSender(HadamardSender):

    
    def __init__(self, device='cpu', delta=None):
        
        HadamardSender.__init__(self, device=device)
        
        self.centroids, self.boundries, self.cscales = gen_normal_centoirds_and_boundries(device)
        
        self.delta = delta
        if self.delta is not None:      
            self.centroid_lookup_table = {}
            for b in self.centroids:
                self.centroid_lookup_table[b] = centroid_lookup_table(self.centroids[b], self.boundries[b], delta)
        

    def quantize(self, vec, nbits, cscale):
        
        if self.delta is not None:
            normalized_vec = vec * (vec.numel()**0.5) / (torch.norm(vec,2))
            lt_size = len(self.centroid_lookup_table[nbits])
            table_entries = torch.clamp(normalized_vec/self.delta+lt_size/2, 0, lt_size-1).long()
            bins = torch.take(self.centroid_lookup_table[nbits], table_entries)
        else:
            bins = torch.bucketize(vec * (vec.numel()**0.5) / torch.norm(vec,2), self.boundries[nbits])
        
        if cscale:
            scale = torch.norm(vec,2) / (self.cscales[nbits] * np.sqrt(vec.numel()))
        else:
            scale = torch.norm(vec,2)**2 / torch.dot(torch.take(self.centroids[nbits], bins) , vec)
    
        return bins, scale  


    def stochastic_quantize(self, vec, nbits_low, nbits_high, p_high, seed):
        
        bins_low = torch.bucketize(vec * (vec.numel()**0.5) / torch.norm(vec,2), self.boundries[nbits_low])
        bins_high = torch.bucketize(vec * (vec.numel()**0.5) / torch.norm(vec,2), self.boundries[nbits_high])
        
        centroids_low = torch.take(self.centroids[nbits_low], bins_low)
        centroids_high = torch.take(self.centroids[nbits_high], bins_high)
        
        self.prng.manual_seed(seed)
        mask_high = torch.bernoulli(torch.ones(size=(vec.numel(),), device=self.device) * p_high, generator=self.prng).bool()
        
        bins_low.masked_scatter_(mask_high, torch.masked_select(bins_high, mask_high))        
        centroids_low.masked_scatter_(mask_high, torch.masked_select(centroids_high, mask_high))     
        
        scale = torch.norm(vec,2)**2 / torch.dot(centroids_low , vec)
    
        return bins_low, scale  


    def compress(self, data):

        dim = data['vec'].numel()
        
        if not dim & (dim - 1) == 0:
            
            padded_dim = int(2**(np.ceil(np.log2(dim))))
            padded_vec = torch.zeros(padded_dim, device=self.device)
            padded_vec[:dim] = data['vec'] 
            
            vec = self.randomized_hadamard_transform(padded_vec, data['seed'])
        
        else:
            
            vec = self.randomized_hadamard_transform(data['vec'], data['seed'])
        
        if data['nbits'] == round(data['nbits']) or data['nbits'] < 1:
            
            bins, scale = self.quantize(vec, np.ceil(data['nbits']), False)
            
            return {'bins': bins, 'scale': scale, 'seed': data['seed'], 'nbits': data['nbits'], 'dim': dim, 'pdrop': data.get('pdrop',0)}
        
        else:
            
            bits_h = int(np.ceil(data['nbits']))
            bits_l = int(np.floor(data['nbits']))
            
            p_high = data['nbits'] - bits_l
            
            bins, scale = self.stochastic_quantize(vec, bits_l, bits_h, p_high, data['seed']*7+13)
                        
            return {'bins': bins, 'scale': scale, 'seed': data['seed'], 'nbits': (bits_l, bits_h, p_high), 'dim': dim, 'pdrop': data.get('pdrop',0)}
        
##############################################################################
##############################################################################

class EdenReceiver(HadamardReceiver):

    
    def __init__(self, device='cpu'):
        
        HadamardReceiver.__init__(self, device=device)
        
        self.centroids, _, _ = gen_normal_centoirds_and_boundries(device)
        
            
    def decompress(self, data):
        
        if not isinstance(data['nbits'], tuple):
            
            vec = torch.take(self.centroids[np.ceil(data['nbits'])], data['bins'])
        
        else:

            self.prng.manual_seed(data['seed']*7+13)
            mask_high = torch.bernoulli(torch.ones(size=(data['bins'].numel(),), device=self.device) * data['nbits'][2], generator=self.prng).bool()
            
            vec = torch.take(self.centroids[int(data['nbits'][1])], data['bins'])                        
            vec.masked_scatter_(~mask_high, torch.take(self.centroids[int(data['nbits'][0])], torch.masked_select(data['bins'],~mask_high)))  
            
        pdrop = 0
        
        if not isinstance(data['nbits'], tuple):
            if data['nbits'] < 1:
               pdrop = 1 - data['nbits']
        if data['pdrop'] > 0:
           pdrop += (1 - pdrop) * data['pdrop'] 
        
        if pdrop > 0:
                            
            dim = vec.numel()
            
            ri = torch.randperm(dim)[:round(dim*pdrop)]
            vec[ri] = 0
            vec /= (1-pdrop)
        
        vec = self.randomized_inverse_hadamard_transform(vec, data['seed'])
    
        return (data['scale'] * vec)[:data['dim']]
               
##############################################################################
##############################################################################            

