#!/usr/bin/env python3

import pathlib
path = pathlib.Path(__file__).parent.resolve()

import sys
sys.path.insert(0, str(path) + "/../")

import torch
import xxhash
import numpy as np

from hadamard import HadamardSender, HadamardReceiver

##############################################################################
##############################################################################

class QuicFLSender(HadamardSender):
    
    def __init__(self, device='cpu', bits=[1,2,3,4], sr_bits=[6,5,4,4], prefix=str(path) + '/tables/'):
        
        HadamardSender.__init__(self, device=device)
        self.local_prng = torch.Generator(device=device)
        
        self.sender_table_X = {}
        self.sender_table_p = {}
        self.data = {}
        self.half_table_size = {} 
        
        for i in range(4):
                        
            fn = prefix + '{}_X_{}_h_256_q_'.format(bits[i],sr_bits[i])
                        
            self.sender_table_X[bits[i]], self.sender_table_p[bits[i]], self.data[bits[i]] = self.sender_table(fn, device)
            self.half_table_size[bits[i]] = ((self.sender_table_X[bits[i]].numel() // self.data[bits[i]]['h_len']) - 1) * self.data[bits[i]]['h_len'] // 2
        
    ##########################################################################
       
    def sender_table(self, prefix, device):
    
        sender_table_X = torch.load(prefix + 'sender_table_X.pt').to(device)
        sender_table_p = torch.load(prefix + 'sender_table_p.pt').to(device)
        
        data = eval(open(prefix + 'data.txt').read())
    
        return sender_table_X, sender_table_p, data
            
    ##########################################################################
        
    def compress(self, data):  
        
        dim = data['vec'].numel()
        
        prng_seed = xxhash.xxh64(str(data["seed"])).intdigest()%2**16
        self.local_prng.manual_seed(prng_seed)
        
        if not dim & (dim - 1) == 0:
            
            padded_dim = int(2**(np.ceil(np.log2(dim))))
            padded_vec = torch.zeros(padded_dim, device=data['vec'].device)
            padded_vec[:dim] = data['vec'] 
            
            vec = HadamardSender.randomized_hadamard_transform(self, padded_vec, data['rotation_seed'])
            h = torch.randint(0,self.data[data['nbits']]['h_len'],(padded_dim,), device=data['vec'].device, generator=self.local_prng).to(data['vec'].device)
        
            scale = np.sqrt(padded_dim) / vec.norm(2)
            
        else:
            
            vec = HadamardSender.randomized_hadamard_transform(self, data['vec'], data['rotation_seed'])
            h = torch.randint(0,self.data[data['nbits']]['h_len'],(dim,), device=data['vec'].device, generator=self.local_prng).to(data['vec'].device)
            
            scale = np.sqrt(dim) / vec.norm(2)
            
        rotated_and_scaled_vec = vec * scale
        
        exact = (rotated_and_scaled_vec > self.data[data['nbits']]['T']) + (rotated_and_scaled_vec < - self.data[data['nbits']]['T'])
        
        q_rotated_and_scaled_vec = rotated_and_scaled_vec/self.data[data['nbits']]['delta'] 
        q_rotated_and_scaled_vec[exact] = 0
        
        p = q_rotated_and_scaled_vec - q_rotated_and_scaled_vec.floor()
        int_q_rotated_and_scaled_vec = q_rotated_and_scaled_vec.floor() + torch.bernoulli(p, generator=self.local_prng)
        
        X = torch.take(self.sender_table_X[data['nbits']], (int_q_rotated_and_scaled_vec*self.data[data['nbits']]['h_len'] + h + self.half_table_size[data['nbits']]).long())
        p_X = torch.take(self.sender_table_p[data['nbits']], (int_q_rotated_and_scaled_vec*self.data[data['nbits']]['h_len'] + h + self.half_table_size[data['nbits']]).long())
        
        X += torch.bernoulli(p_X)
        X = X.long()
                
        return {'X': X, 
                'exact_values': rotated_and_scaled_vec[exact], 
                'exact_indeces': exact, 
                'seed': data['seed'],
                'prng_seed': prng_seed,
                'rotation_seed': data['rotation_seed'], 
                'dim': dim, 
                'scale': scale, 
                'nbits': data['nbits'], 
                'h_len': self.data[data['nbits']]['h_len']}
        
##############################################################################
##############################################################################

class QuicFLReceiver(HadamardReceiver):
    
    def __init__(self, device='cpu', bits=[1,2,3,4], sr_bits=[6,5,4,4], prefix=str(path) + '/tables/'):
        
        HadamardReceiver.__init__(self, device=device)
        self.local_prng = torch.Generator(device=device)
        
        self.recv_table = {}
         
        for i in range(4):
            
            fn = prefix + '{}_X_{}_h_256_q_'.format(bits[i],sr_bits[i])
            
            self.recv_table[bits[i]] = self.receiver_table(fn, device)
               
    ##########################################################################
       
    def receiver_table(self, prefix, device):
        
        recv_table = torch.load(prefix +'recv_table.pt').to(device)
        
        return recv_table

    ##########################################################################
                   
    def decompress(self, data):
        
        self.local_prng.manual_seed(data['prng_seed'])
        h = torch.randint(0,data['h_len'],(data['X'].numel(),), device=data['X'].device, generator=self.local_prng).to(data['X'].device)

        client_rotated_and_scaled_vec = torch.take(self.recv_table[data['nbits']], (data['X']*data['h_len'] + h))
        client_rotated_and_scaled_vec[data['exact_indeces']] = data['exact_values'] ### exact is a boolean maskl - improve...
        client_rotated_and_scaled_vec /= data['scale']
                    
        vec = HadamardReceiver.randomized_inverse_hadamard_transform(self, client_rotated_and_scaled_vec, data['rotation_seed'])
    
        return vec[:data['dim']]
               
##############################################################################
##############################################################################            
##############################################################################
##############################################################################

