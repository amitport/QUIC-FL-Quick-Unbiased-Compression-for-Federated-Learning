#!/usr/bin/env python3

import torch
import xxhash

##############################################################################
##############################################################################

class StochasticQuantizationSender:


    def __init__(self, device='cpu'):

        self.device = device
        self.prng = torch.Generator(device=device)


    def compress(self, data):

        self.prng.manual_seed(xxhash.xxh64(str(data["seed"])).intdigest()%2**16)

        if data.get('step','standard') == "standard":
            step = (data['vec'].max()-data['vec'].min())/(data['nlevels']-1) ### step size
        elif data["step"] == "norm":
            step = 1.4142135623730951*torch.norm(data['vec'])/(data['nlevels']-1) ### step size
        else:
            raise Exception("unknown step size")

        r = (data['vec']-data['vec'].min())/step ### number of steps from the min
        r = torch.floor(r) + torch.bernoulli(r-torch.floor(r), generator=self.prng) ### sq

        return {'bins': r, 'min': data['vec'].min(), 'step': step}

##############################################################################
##############################################################################

class StochasticQuantizationReceiver:


    def __init__(self, device='cpu'):

        self.device = device


    def decompress(self, data):

        return data['min']+data['bins']*data['step']

##############################################################################
##############################################################################

