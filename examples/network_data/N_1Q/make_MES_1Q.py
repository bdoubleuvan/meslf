"""Create the single-carrier networks consisting of 1 quarter, which consists of nS streets of n loads each"""
from examples.network_data.N_1Q import make_GN_1Q, make_EN_1Q, make_HN_1Q
import sys
import os 

def create_single_carrier_networks(n,m,nS,dir_path):
    """Create the pd dataframes for the single carrier networks.
    
    Parameters
    ----------
    n : int
        number of loads, for each street
    m : int
        number of junctions with two loads, for each street
    nS : int
        number of street networks
    """
    
    make_GN_1Q.main(n,m,nS,dir_path)
    make_EN_1Q.main(n,m,nS,dir_path)
    make_HN_1Q.main(n,m,nS,dir_path)
    
if __name__== '__main__':  
    if(len(sys.argv) > 1):
        n = int(sys.argv[1]) # numer of loads, for each street
        m = int(sys.argv[2]) # number of junctions with two loads, for each street
        nS = int(sys.argv[3]) # number of street networks
    else:
        n = 10#10#3#20 # numer of loads, for each street
        m = 5#5#1#10 # number of junctions with two loads, for each street
        nS = 4#3#2#10 # number of street networks
    dir_path = os.path.dirname(os.path.realpath(__file__))
    create_single_carrier_networks(n,m,nS,dir_path)
