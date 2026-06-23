"""Network consisting of two 'streets', based on the GN_2streets data."""
from meslf.networks.gas_network import GasNetwork
from meslf.networks.read_write_network import from_pd_dataframes
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # create gas network from data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','GN_2streets')
    nodes = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_nodes.pkl'))
    links = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_links.pkl'))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_halflinks.pkl'))

    gas_net = from_pd_dataframes(nodes,links,halflinks)

    # plot network topology
    plt.figure('Network topology')
    ax = plt.gca()
    gas_net.draw_network(ax)
    plt.axis('equal')
    plt.axis('off')
    plt.plot()
    
    plt.show()
