import numpy as np
from spyglass.shijiegu.singleUnit import find_spikes
import networkx as nx
from spyglass.shijiegu.fragmented import find_firing_cross_correlation
import matplotlib.pyplot as plt

def find_active_cells_intvl(intvl, nwb_units_all, cell_list):
    """return indices of cells that are active for each interval"""

    axis = np.array([intvl[0], intvl[1]])
    firing_matrix = find_spikes(nwb_units_all,cell_list,axis)
    index = np.argwhere(np.sum(firing_matrix, axis = 0)).ravel()

    return index

def find_active_cells(ripple_times, nwb_units_all, cell_list, cont = False):
    """return indices of cells that are active for each ripple interval"""
    indices = []

    for i in ripple_times.index:

        if cont:
            intvls = ripple_times.loc[i].cont_intvl
        else:
            intvls = ripple_times.loc[i].frag_intvl

        for intvl in intvls:
            axis = np.array([intvl[0], intvl[1]])
            firing_matrix = find_spikes(nwb_units_all,cell_list,axis)
            index = np.argwhere(np.sum(firing_matrix, axis = 0)).ravel()
            indices.append(index)

    return indices

def make_graph(cell_list,indices):
    """for each ripple interval, cells that are active together get an edge in the graph."""


    # initialize all possible edge weights to 0
    weights = {}
    for i in np.arange(len(cell_list)):
        for j in np.arange(len(cell_list)):
            weights[(i,j)] = 0

    # make weights indices are of length the number of ripple intervals
    for ri in np.arange(len(indices)):
        cells = indices[ri]
        for i in range(len(cells)):
            c1 = cells[i]
            weights[c1, c1] += 1
            for c2 in np.setdiff1d(cells,c1):
                weights[c1, c2] += 1

    # use weights to make graph
    G = nx.Graph()
    #all_weights = [weights[key] for key in weights.keys()]
    for key in weights.keys():
        if weights[key] > 0:
            G.add_edge(key[0],key[1])
            G[key[0]][key[1]]['weight'] = weights[key]

    return G, weights

def find_xcorr_col(cell,cell_list,pairs,firing_corr,ripple_ind):
    # find relevent columns in the xcorr result

    cell_ind = np.argwhere(np.sum(np.array(cell_list) == cell,axis = 1) == 2).ravel()[0]
    col_ind1 = np.argwhere(pairs[:,0] == cell_ind).ravel()
    col_ind2 = np.argwhere(pairs[:,1] == cell_ind).ravel()
    other_cell1 = np.array(cell_list)[pairs[col_ind1,1]]
    other_cell2 = np.array(cell_list)[pairs[col_ind2,0]]

    firing_corr_p1 = firing_corr[:,col_ind1]
    ripple_ind_p1 = ripple_ind[col_ind1]
    firing_corr_p2 = firing_corr[:,col_ind2]
    ripple_ind_p2 = ripple_ind[col_ind2]

    firing_corr_p = np.concatenate((firing_corr_p1,np.flipud(firing_corr_p2)),axis = 1)
    ripple_ind_p = np.concatenate((ripple_ind_p1,ripple_ind_p2))
    other_cell = np.concatenate((other_cell1,other_cell2))
    other_cell_ind = np.concatenate((pairs[col_ind1,1],pairs[col_ind2,0]))
    return firing_corr_p, ripple_ind_p, other_cell, other_cell_ind

def weight_from_xcorrelation(ripple_times, nwb_units_all, cell_list, cell_from, cell_to,
                                 cont = False, DELTA_T = 0.01, span = 1):
    """cross correlate during ripple all cells from cell_from (in names not indices)
    to other cells in cell_to (in names as well)"""

    firing_corr_frag, pairs_frag, ripple_ind_frag, lag_frag = find_firing_cross_correlation(
        ripple_times, nwb_units_all, cell_list, cont = cont, DELTA_T = DELTA_T)

    # initialize all possible edge weights to 0
    weights = {}
    for i in np.arange(len(cell_list)):
        for j in np.arange(len(cell_list)):
            weights[(i,j)] = 0

    for c0_ind in range(len(cell_from)):
        cell = cell_from[c0_ind]
        cell0_ind = np.argwhere(np.sum(np.array(cell_list) == np.array(cell), axis = 1) == 2).ravel()[0]

        # all the xcorrelation involving cell0
        (firing_corr_frag_p,
         ripple_ind_frag_p,
         other_cell,
         other_cell_ind) = find_xcorr_col(cell,cell_list,
                                          pairs_frag,firing_corr_frag,ripple_ind_frag)

        ind_center = np.argwhere(lag_frag == 0).ravel()[0]
        xcorr_center = np.max(firing_corr_frag_p.T[:,(ind_center-span):(ind_center+span)],axis = 1)
        xcorr_thresh = 0 #np.mean(xcorr_center) + 1*np.std(xcorr_center)
        cell1_ind = other_cell_ind[xcorr_center >= xcorr_thresh]

        #if len(cell1_ind) > 0:
        #    cell_name = np.array(cell_list)[cell0_ind]
        #    label = str(cell_name[0])+' '+str(cell_name[1])
        #    G.add_nodes_from([(cell0_ind, {"color": "blue"})])
        #    cell0_list[cell0_ind] = label

        for c1_ind in cell1_ind:
            #cell_name = np.array(cell_list)[c1_ind]
            #label = str(cell_name[0])+' '+str(cell_name[1])
            #G.add_nodes_from([(c1_ind, {"color": "orange"})])

            # the other cell need to be in cell_to
            c1_name = np.array(cell_list)[c1_ind]
            if np.sum(np.sum(np.array(cell_to) == c1_name, axis = 1) == 2) > 0:
                weights[(cell0_ind,c1_ind)] += 1

            #cell1_list[c1_ind] = label
    return weights

def make_graph_from_weights(weights, threshold = 2, weights_sub = None):
    """color edges in the weights_sub in red if weights_sub is offered"""
    G = nx.Graph()

    # add in all weights
    for key in weights.keys():
        if weights[key] >= threshold:
            (cell0_ind, cell1_ind) = key

            # add node
            G.add_nodes_from([(cell0_ind, {"color": "orange"})])
            G.add_nodes_from([(cell1_ind, {"color": "orange"})])

            # add weight
            G.add_edge(cell0_ind, cell1_ind, color = 'k', weight = weights[key])
            G[cell0_ind][cell1_ind]['weight'] = weights[key]

    if weights_sub is None:
        return G

    for key in weights_sub.keys():
        if weights_sub[key] >= threshold:
            (cell0_ind, cell1_ind) = key

            # add node
            G.add_nodes_from([(cell0_ind, {"color": "orange"})])

            # add weight
            G.add_edge(cell0_ind, cell1_ind, color='r', weight = weights_sub[key])
            G[cell0_ind][cell1_ind]['weight'] = weights_sub[key]
    return G

def cell_name_to_ind(cell_name,cell_list):
    cell_ind = np.argwhere(
        np.sum(cell_name == np.array(cell_list),axis = 1) == 2
        ).ravel()[0]
    return cell_ind

def plot_graph(G_prime, cell_list, labels, cell0_list = None, cell1_list = None,
               edge_color = 'k', alpha = 0.5,
               color_fieldPeak = None, color_ratio = None, version = 1):
    #pos = nx.spring_layout(G_prime,scale = 3)  # positions for all nodes
    fig, axes = plt.subplots(1,2, figsize = (20,8))
    # version 0: two plots, one colored by two groups of cells, one colored by place field location
    # version 1: two plots, one colored by spike count ratio, one colored by place field location

    pos = nx.kamada_kawai_layout(G_prime,scale = 3)

    # plot 1
    if version == 0:
        # two groups
        for cell0_ind in range(len(cell0_list)):
            cell = cell0_list[cell0_ind]
            cell_ind = cell_name_to_ind(cell,cell_list)
            #cell_ind = np.argwhere(np.sum(cell == np.array(cell_list),axis = 1) == 2).ravel()[0]
            try:
                nx.draw_networkx_nodes(G_prime, pos, ax = axes[0], nodelist=[cell_ind], node_color="tab:blue") #the order of plotting matters here
            except:
                pass
            #nx.draw_networkx_labels(G_prime, pos, ax = axes[0])
        for cell1_ind in range(len(cell1_list)):
            cell = cell1_list[cell1_ind]
            cell_ind = cell_name_to_ind(cell,cell_list)
            #cell_ind = np.argwhere(np.sum(cell == np.array(cell_list),axis = 1) == 2).ravel()[0]
            try:
                nx.draw_networkx_nodes(G_prime, pos, ax = axes[0], nodelist=[cell_ind], node_color="tab:orange") #the order of plotting matters here
            except:
                pass
        nx.draw_networkx_labels(G_prime, pos, labels, ax = axes[0])
        #nx.draw(G_prime, pos, ax = axes[0], with_labels=False, alpha = 0.5, node_color="white")

    elif version == 1:
        # ratio color
        for cell_ind in range(len(cell_list)):
            cell = cell_list[cell_ind]
            try:
                nx.draw_networkx_nodes(G_prime, pos, ax=axes[0],
                                       nodelist=[cell_ind], node_color=color_ratio[cell]) #the order of plotting matters here
                nx.draw_networkx_labels(G_prime,pos,[cell_ind], ax = axes[0])
            except:
                pass
        nx.draw_networkx_labels(G_prime, pos, labels, ax = axes[0])

    for edge in G_prime.edges(data='weight'):
        nx.draw_networkx_edges(G_prime, pos, ax=axes[0], edgelist=[edge], width=edge[2]/3, edge_color = edge_color, alpha = alpha)


    # plot 2:
    # color_fieldPeak
    for cell_ind in range(len(cell_list)):
        cell = cell_list[cell_ind]
        try:
            nx.draw_networkx_nodes(G_prime, pos, ax=axes[1],
                                   nodelist=[cell_ind], node_color=color_fieldPeak[cell]) #the order of plotting matters here
        except:
            pass
    nx.draw(G_prime, pos, ax = axes[1], with_labels=True, alpha = 0.5, node_color="white")
    nx.draw_networkx_labels(G_prime, pos, labels, ax = axes[1])

    #nx.draw_networkx_labels(G_prime,pos,cell0_list, ax = axes[1]);
    #nx.draw_networkx_labels(G_prime,pos,cell1_list, ax = axes[1]);

    for edge in G_prime.edges(data='weight'):
        nx.draw_networkx_edges(G_prime, pos, ax=axes[1], edgelist=[edge], width=edge[2]/5)
    return pos, axes

def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center

def multipartite_layout(G, subset_key="subset", align="vertical", scale=1, center=None):
    if align not in ("vertical", "horizontal"):
        msg = "align must be either vertical or horizontal."
        raise ValueError(msg)

    G, center = _process_params(G, center=center, dim=2)
    if len(G) == 0:
        return {}

    try:
        # check if subset_key is dict-like
        if len(G) != sum(len(nodes) for nodes in subset_key.values()):
            raise nx.NetworkXError(
                "all nodes must be in one subset of `subset_key` dict"
            )
    except AttributeError:
        # subset_key is not a dict, hence a string
        node_to_subset = nx.get_node_attributes(G, subset_key)
        if len(node_to_subset) != len(G):
            raise nx.NetworkXError(
                f"all nodes need a subset_key attribute: {subset_key}"
            )
        subset_key = nx.utils.groups(node_to_subset)
    # Sort by layer, if possible
    try:
        layers = dict(sorted(subset_key.items()))
    except TypeError:
        layers = subset_key

    pos = None
    nodes = []
    width = len(layers)
    for i, layer in enumerate(layers.values()):
        height = len(layer)
        xs = np.repeat(i, height)
        ys = np.arange(0, height, dtype=float)
        offset = ((width - 1) / 2, (height - 1) / 2)
        layer_pos = np.column_stack([xs, ys]) - offset
        if pos is None:
            pos = layer_pos
        else:
            pos = np.concatenate([pos, layer_pos])
        nodes.extend(layer)
    pos = rescale_layout(pos, scale=scale) + center
    if align == "horizontal":
        pos = pos[:, ::-1]  # swap x and y coords
    pos = dict(zip(nodes, pos))
    return pos

def rescale_layout(pos, scale=1):
    # Find max length over all dimensions
    pos -= pos.mean(axis=0)
    lim = np.abs(pos).max()  # max coordinate for all axes
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        pos *= scale / lim
    return pos

def bfs_layout(G, start, *, align="vertical", scale=1, center=None):
    """Position nodes according to breadth-first search algorithm.

    Parameters
    ----------
    G : NetworkX graph
        A position will be assigned to every node in G.

    start : node in `G`
        Starting node for bfs

    center : array-like or None
        Coordinate pair around which to center the layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.bfs_layout(G, 0)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """
    G, center = _process_params(G, center, 2)

    # Compute layers with BFS
    layers = dict(enumerate(nx.bfs_layers(G, start)))

    if len(G) != sum(len(nodes) for nodes in layers.values()):
        raise nx.NetworkXError(
            "bfs_layout didn't include all nodes. Perhaps use input graph:\n"
            "        G.subgraph(nx.node_connected_component(G, start))"
        )

    # Compute node positions with multipartite_layout
    return multipartite_layout(
        G, subset_key=layers, align=align, scale=scale, center=center
    )