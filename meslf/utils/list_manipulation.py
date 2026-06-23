"""Usefull list manipulations """

# Flatten lists
def flatten(list1):
    """Flattens a list of list.
    
    Parameters
    ----------
    list1 : lst
        List of lists to flatten
        
    Returns
    -------
    flat_list : lst
        The flattened list
    """
    return [item for sublist in list1 for item in sublist]
    
def merge_sorted(list1,list2,idfun = None):
    """Creates a new list, which merges list2 with list1, while keeping the order of list1.
    
    Parameters
    ----------
    list1 : list
        First list to be merged. The order of this list is kept
    list2 : list
        Second list to be merged.
    idfun : function, optional
        Function that operates on both list items before merging them (the default is the identity function, which means the list items aren't affected).
        
    Returns
    -------
    merged_list : list
        The order perserving merged list
    """
    # Based on f5 https://www.peterbe.com/plog/uniqifiers-benchmark
    if idfun is None:
       def idfun(x): return x
    seen = {}
    merged_list = []
    for item in list1 + list2:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        merged_list.append(item)
    return merged_list
