#!/usr/bin/env python3

import numpy as np
import torch
import argparse
from countsketch import CountSketchSender, CountSketchReceiver
from eden import EdenSender, EdenReceiver
from hadamard import HadamardStochasticQuantizationSender, HadamardStochasticQuantizationReceiver
from kashin import KashinStochasticQuantizationSender, KashinStochasticQuantizationReceiver
from qsgd import QSGDSender, QSGDReceiver
from quic_fl import QuicFLSender, QuicFLReceiver

##############################################################################
##############################################################################

if __name__ == '__main__':
    ALGORITHMS = {'quic-fl': (QuicFLSender, QuicFLReceiver),
                  'eden': (EdenSender, EdenReceiver),
                  'kashin': (KashinStochasticQuantizationSender, KashinStochasticQuantizationReceiver),
                  'kashin-tf': (KashinStochasticQuantizationSender, KashinStochasticQuantizationReceiver),
                  'hadamard': (HadamardStochasticQuantizationSender, HadamardStochasticQuantizationReceiver),
                  'qsgd': (QSGDSender, QSGDReceiver),
                  'countsketch': (CountSketchSender, CountSketchReceiver)}

    parser = argparse.ArgumentParser(description='NMSE test.')
    parser.add_argument('--alg', choices=ALGORITHMS.keys(), default='quic-fl')
    parser.add_argument('--nbits', type=int, default=4,  help='number of integer bits to use')
    parser.add_argument('--d', type=int, default=2 ** 19, help='dimension of the \' vectors')
    parser.add_argument('--nclients', type=int, default=10, help='number of clients/senders (for vNMSE set nclient=1)')
    parser.add_argument('--ntrials', type=int, default=10, help='number of trials')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--rotation_seed', type=int, default=123, help='rotation seed')
    args = parser.parse_args()

    alg, nbits, d, nclients, ntrials = args.alg, args.nbits, args.d, args.nclients, args.ntrials

    seed, rotation_seed = args.seed, args.rotation_seed
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    ### random.seed(seed)
    
    ### device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    sender, receiver = ALGORITHMS[alg][0](device=device), ALGORITHMS[alg][1](device=device)

    ### distribution
    lognormal_distribution = torch.distributions.LogNormal(0, 1)

    ### Total NMSE
    NMSE = 0
    
    ### info print
    print("\n*** Running {} with {} bits per entry, dimension {}, {} trials and {} clients ***".format(alg,nbits,d,ntrials,nclients))
    print("\n*** Seed: {}, rotation seed: {}***\n".format(seed, rotation_seed))
    
    for trial in range(ntrials):
        
        print("Completed trial {}/{}".format(trial+1,ntrials))
                            
        rvec = torch.zeros(d).to(device)
        ovec = torch.zeros(d).to(device)
        sum_norm = 0
        
        for client in range(nclients):
                 
            vec = lognormal_distribution.sample([d]).to(device)
            
            ##################################################################
            
            data = {}
            data['vec'] = vec
            data['seed'] = trial * nclients + client
            data['nbits'] = nbits
            
            ### for hadamard, kashin and quic-fl 
            data['rotation_seed'] = rotation_seed
            
            ### for QSGD
            data['nlevels'] = 2 ** nbits

            ### for countsketch
            data["r"] = 3
            data["c"] = d // 32
            
            ### kashin-tf?
            if alg == 'kashin-tf':
                data['niters'] = 3

            ##################################################################
            
            data = sender.compress(data)
            client_rvec = receiver.decompress(data)
            
            rvec += client_rvec
            ovec += vec
            sum_norm += vec.norm() ** 2

        rvec /= nclients
        ovec /= nclients

        NMSE += torch.norm(ovec - rvec, 2) ** 2 / (sum_norm / nclients)

    print("\n *** Empirical NMSE: {} *** \n".format(NMSE / ntrials))
