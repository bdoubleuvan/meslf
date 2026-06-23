"""Create the single-carrier networks consisting of 1 city, which consists of (hierarchically) nD districts, nQ quarters, and nS streets of n loads each"""
from examples.network_data.N_1C import make_GN_1C, make_EN_1C, make_HN_1C
import sys
import os 
import argparse 

command_line_input = argparse.ArgumentParser()
command_line_input.add_argument(
    "--n", # number of loads, for each street
    type=int,
    default = 10, # default if nothing is provided
    )
command_line_input.add_argument(
    "--m", # number of junctions with two loads, for each street
    type=int,
    default = 5, # default if nothing is provided
    )
command_line_input.add_argument(
    "--nS", # number of street networks
    type=int,
    default = 4, # default if nothing is provided
    )
command_line_input.add_argument(
    "--nQ", # number of quarter networks
    type=int,
    default = 4, # default if nothing is provided
    )
command_line_input.add_argument(
    "--nD", # number of district networks
    type=int,
    default = 5, # default if nothing is provided
    )
command_line_input.add_argument(
    "--Ta", # Ambient temperature of the heat network
    type=int,
    default = 0, # default if nothing is provided
    )
command_line_input.add_argument(
    "--p_low", # Lowest value of fraction of reference pressure that is used in the initial linear pressure profile
    type=float,
    default = .98, # default if nothing is provided
    )
command_line_input.add_argument(
    "--p_high", # name on the command line Drop the `--` for positional/required parameters
    type=float,
    default = .99, # default if nothing is provided
    )

def create_single_carrier_networks(n,m,nS,nQ,nD,Ta,p_init_low,p_init_high,dir_path,heat_load='outflow',hydr_eq='dp_of_q'):
    """Create the pd dataframes for the single carrier networks.
    
    Parameters
    ----------
    n : int
        number of loads, for each street
    m : int
        number of junctions with two loads, for each street
    nS : int
        number of street networks
    nQ : int
        number of quarter networks
    nD : int
        number of district networks
    """
    
    make_GN_1C.main(n,m,nS,nQ,nD,dir_path,p_init_low,p_init_high,hydr_eq=hydr_eq)
    make_EN_1C.main(n,m,nS,nQ,nD,dir_path)
    make_HN_1C.main(n,m,nS,nQ,nD,Ta,p_init_low,p_init_high,dir_path,heat_load=heat_load)

if __name__== '__main__':  
    args = command_line_input.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))

    create_single_carrier_networks(args.n,args.m,args.nS,args.nQ,args.nD,args.Ta,args.p_low,args.p_high,dir_path)
