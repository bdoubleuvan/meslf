"""Test the network classes and methods
"""
from meslf.networks.network import Network, Node, Link, HalfLink
import numpy as np
import pandas as pd

# Network and Node are abstract class, with abstract methods, which means they cannot be instantiated.
# The following functions creates a new class, which overrides the abstract methods.
def concreter(abclass):
    """Creates a new class, which is the same as abclass, but with the abstract methods overriden

    from: https://stackoverflow.com/questions/9757299/python-testing-an-abstract-base-class
    """
    if not "__abstractmethods__" in abclass.__dict__:
        return abclass
    new_dict = abclass.__dict__.copy()
    for abstractmethod in abclass.__abstractmethods__:
        #replace each abc method or property with an identity function:
        new_dict[abstractmethod] = lambda x, *args, **kw: (x, args, kw)
    #creates a new class, with the overriden ABCs:
    return type("dummy_concrete_%s" % abclass.__name__, (abclass,), new_dict)

# Test Network methods
def test_add_node():
    """Check that a node is added to a network correctly
    """
    # Given
    net = concreter(Network)('test network')
    node = concreter(Node)('test node')

    # When
    net.add_node(node)

    # Then
    assert node in net.nodes

def test_remove_node():
    """Check that a node is removed from a network correctly
    """
    # Given
    net = concreter(Network)('test network')
    node = concreter(Node)('test node')
    net.add_node(node)

    # When
    net.remove_node(node)

    # Then
    assert node not in net.nodes

def test_remove_node_with_link():
    """Check that a node, and the link connected to it, are removed from a network correctly
    """
    # Given
    net = concreter(Network)('test network')
    node0 = concreter(Node)('test node0')
    node1 = concreter(Node)('test node1')
    link = Link('test link',node0,node1)

    # When
    net.add_link(link)
    net.remove_node(node0)

    # Then
    assert link not in net.links

def test_remove_node_with_half_link():
    """Check that a node, and the half link connected to it, are removed from a network correctly
    """
    # Given
    net = concreter(Network)('test network')
    node = concreter(Node)('test node')
    halflink = HalfLink('test link',node)

    # When
    net.add_half_link(halflink)
    net.remove_node(node)
    print(net.half_links)

    # Then
    assert halflink not in net.half_links

def test_add_link():
    """Check that a link is added to a network correctly.
    """
    # Given
    net = concreter(Network)('test network')
    node0 = concreter(Node)('test node0')
    node1 = concreter(Node)('test node1')
    link = Link('test link',node0,node1)

    # When
    net.add_link(link)

    # Then
    assert (link in net.links and node0 in net.nodes and node1 in net.nodes)

def test_remove_link():
    """Check if a link is removed from a network correctly
    """
    # Given
    net = concreter(Network)('test network')
    node0 = concreter(Node)('test node0')
    node1 = concreter(Node)('test node1')
    link = Link('test link',node0,node1)
    net.add_link(link)

    # When
    net.remove_link(link)

    # Then
    assert link not in net.links

def test_add_half_link():
    """Check if a half link is added to a network correctly
    """
    # Given
    net = concreter(Network)('test network')
    node = concreter(Node)('test node')
    halflink = HalfLink('test link',node)

    # When
    net.add_half_link(halflink)

    # Then
    assert ((halflink in net.half_links) and (node in net.nodes))

def test_remove_half_link():
    """Check if a half link is removed from a network correctly
    """
    # Given
    net = concreter(Network)('test network')
    node = concreter(Node)('test node')
    halflink = HalfLink('test link',node)
    net.add_half_link(halflink)

    # When
    net.remove_half_link(halflink)

    # Then
    assert halflink not in net.half_links

def test_get_nodes():
    """Check if all the nodes of a network are returned
    """
    # Given
    net = concreter(Network)('test network')
    node0 = concreter(Node)('test node0')
    node1 = concreter(Node)('test node1')

    net.add_node(node0)
    net.add_node(node1)

    # When
    all_nodes = list(net.get_nodes())

    # Then
    assert net.nodes == all_nodes

def test_get_links():
    """Check if all the links of a network are returned
    """
    # Given
    net = concreter(Network)('test network')
    node0 = concreter(Node)('test node0')
    node1 = concreter(Node)('test node1')
    link0 = Link('test link0',node0,node1)
    link1 = Link('test link1',node1,node0)

    net.add_link(link0)
    net.add_link(link1)

    # When
    all_links = list(net.get_links())

    # Then
    assert net.links == all_links

def test_get_half_links():
    """Check if all the half links of a network are returned
    """
    # Given
    net = concreter(Network)('test network')
    node = concreter(Node)('test node')
    halflink0 = HalfLink('test link 0',node)
    halflink1 = HalfLink('test link 1',node)

    # When
    all_halflinks = list(net.get_half_links())

    # Then
    assert all_halflinks == net.half_links

# Test node methods
def test_node_get_out_links():
    """Check if all the outgoing links and half links connected to a node are returned
    """
    # Given
    node0 = concreter(Node)('test node 0')
    node1 = concreter(Node)('test node 1')
    halflink = HalfLink('test halflink',node0)
    link = Link('test link',node0,node1)

    # When
    all_out_links = list(node0.get_out_links())

    # Then
    assert all_out_links == node0.out_links

def test_node_get_in_links():
    """Check if all the incoming links and half links connected to a node are returned
    """
    # Given
    node0 = concreter(Node)('test node 0')
    node1 = concreter(Node)('test node 1')
    halflink = HalfLink('test halflink',node1)
    link = Link('test link',node0,node1)

    # When
    all_in_links = list(node1.get_in_links())

    # Then
    assert all_in_links == node1.in_links


def test_node_get_half_links():
    """Check if all the half links connected to a node are returned
    """
    # Given
    node = concreter(Node)('test node')
    halflink0 = HalfLink('test link 0',node)
    halflink1 = HalfLink('test link 1',node)

    # When
    all_half_links = list(node.get_half_links())

    # Then
    assert all_half_links == node.half_links


def test_node_get_all_links():
    """Check if all the links and half links connected to a node are returned
    """
    # Given
    node0 = concreter(Node)('test node 0')
    node1 = concreter(Node)('test node 1')
    halflink = HalfLink('test halflink',node0)
    link0 = Link('test link 0',node0,node1)
    link1 = Link('test link 1',node1,node0)

    # When
    all_links = list(node0.get_links())

    # Then
    assert all_links == node0.out_links + node0.in_links

def test_add_network_half_links():
    """Check that the half links of two network are added correctly
    """
    # Given
    net0 = concreter(Network)('test network 0')
    node0 = concreter(Node)('test node 0')
    halflink0 = HalfLink('test link 0',node0)
    net0.add_half_link(halflink0)
    net1 = concreter(Network)('test network 1')
    node1 = concreter(Node)('test node 1')
    halflink1 = HalfLink('test link 1',node1)
    net1.add_half_link(halflink1)

    net = concreter(Network)('test network combined')

    # When
    net.add_network(net0)
    net.add_network(net1)
    half_links_expected = [halflink0,halflink1]

    # Then
    assert net.half_links == half_links_expected
