import pickle
import numpy as np

from spyglass.common.common_interval import IntervalList
from spyglass.utils import logger


def load_model_with_networkx_compatibility(detector_class, file_path):
    """Load a pickled model with NetworkX compatibility fixes.
    
    This function handles compatibility issues between different versions of
    NetworkX when loading pickled models. Older models may have NetworkX graphs
    with different internal structure than expected by newer versions.
    
    Parameters
    ----------
    detector_class : class
        The detector class (SortedSpikesDetector or ClusterlessDetector)
    file_path : str or Path
        Path to the pickled model file
        
    Returns
    -------
    model
        The loaded model object
        
    Raises
    ------
    Exception
        If the model cannot be loaded even with compatibility fixes
    """
    try:
        # Try the normal load first
        return detector_class.load_model(file_path)
    except AttributeError as e:
        if "'Graph' object has no attribute '_adj'" in str(e):
            # This is a NetworkX compatibility issue, try to fix it
            logger.warning(
                f"NetworkX compatibility issue detected when loading {file_path}. "
                "Attempting to fix graph attributes during loading."
            )
            
            # Load the pickled data directly and fix NetworkX graphs
            with open(file_path, "rb") as f:
                try:
                    # Use a custom unpickler that fixes NetworkX graphs
                    unpickler = _NetworkXCompatibilityUnpickler(f)
                    return unpickler.load()
                except Exception as fix_error:
                    logger.error(
                        f"Failed to fix NetworkX compatibility issue: {fix_error}"
                    )
                    # Re-raise the original error if our fix didn't work
                    raise e
        else:
            # Re-raise if it's a different AttributeError
            raise e


class _NetworkXCompatibilityUnpickler(pickle.Unpickler):
    """Custom unpickler that fixes NetworkX compatibility issues."""
    
    def load(self):
        """Load and fix NetworkX graph compatibility."""
        try:
            obj = super().load()
            return self._fix_networkx_graphs(obj)
        except AttributeError as e:
            if "'Graph' object has no attribute '_adj'" in str(e):
                # Reset file position and try a different approach
                self.file.seek(0)
                return self._load_with_graph_fix()
            raise e
    
    def _fix_networkx_graphs(self, obj):
        """Recursively fix NetworkX graphs in an object."""
        try:
            import networkx as nx
        except ImportError:
            # If NetworkX is not available, return object as-is
            return obj
        
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                if isinstance(attr_value, nx.Graph):
                    # Fix the graph by ensuring it has the required attributes
                    self._fix_single_graph(attr_value)
                elif hasattr(attr_value, '__dict__') or isinstance(attr_value, (list, tuple)):
                    setattr(obj, attr_name, self._fix_networkx_graphs(attr_value))
        elif isinstance(obj, (list, tuple)):
            obj_type = type(obj)
            return obj_type(self._fix_networkx_graphs(item) for item in obj)
        elif hasattr(obj, 'environments') and hasattr(obj.environments, '__iter__'):
            # Handle the specific case of classifier objects with environments
            for env in obj.environments:
                if hasattr(env, 'track_graph') and env.track_graph is not None:
                    self._fix_single_graph(env.track_graph)
        
        return obj
    
    def _fix_single_graph(self, graph):
        """Fix a single NetworkX graph object."""
        try:
            import networkx as nx
        except ImportError:
            return
        
        # If the graph doesn't have _adj but has _succ, create _adj pointing to _succ
        if not hasattr(graph, '_adj') and hasattr(graph, '_succ'):
            graph._adj = graph._succ
        # If it has neither, this might be a different issue
        elif not hasattr(graph, '_adj') and not hasattr(graph, '_succ'):
            # Try to recreate the adjacency structure
            if hasattr(graph, 'nodes') and hasattr(graph, 'edges'):
                # Create a new graph with the same structure
                new_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
                new_graph.add_nodes_from(graph.nodes(data=True))
                new_graph.add_edges_from(graph.edges(data=True))
                # Copy the fixed attributes back
                graph._adj = new_graph._adj
                if hasattr(new_graph, '_succ'):
                    graph._succ = new_graph._succ
    
    def _load_with_graph_fix(self):
        """Alternative loading method that pre-patches NetworkX."""
        try:
            import networkx as nx
        except ImportError:
            # If NetworkX is not available, fall back to regular loading
            return super().load()
        
        # Store original __setstate__ method
        orig_setstate = getattr(nx.Graph, '__setstate__', None)
        
        def patched_setstate(self, state):
            """Patched __setstate__ that handles missing _adj attribute."""
            try:
                if orig_setstate:
                    orig_setstate(self, state)
                else:
                    self.__dict__.update(state)
            except AttributeError as e:
                if "'Graph' object has no attribute '_adj'" in str(e):
                    # Create _adj if it doesn't exist
                    if '_succ' in state and '_adj' not in state:
                        state['_adj'] = state['_succ']
                    self.__dict__.update(state)
                else:
                    raise e
        
        # Temporarily patch the method
        try:
            nx.Graph.__setstate__ = patched_setstate
            # Also patch subclasses
            if hasattr(nx, 'DiGraph'):
                nx.DiGraph.__setstate__ = patched_setstate
            if hasattr(nx, 'MultiGraph'):
                nx.MultiGraph.__setstate__ = patched_setstate
            if hasattr(nx, 'MultiDiGraph'):
                nx.MultiDiGraph.__setstate__ = patched_setstate
            
            # Now load the object
            obj = super().load()
            return obj
        finally:
            # Restore original method
            if orig_setstate:
                nx.Graph.__setstate__ = orig_setstate
            else:
                if hasattr(nx.Graph, '__setstate__'):
                    delattr(nx.Graph, '__setstate__')


def _get_interval_range(key):
    """Return maximum range of model times in encoding/decoding intervals

    Parameters
    ----------
    key : dict
        The decoding selection key

    Returns
    -------
    Tuple[float, float]
        The minimum and maximum times for the model
    """
    encoding_interval = (
        IntervalList
        & {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["encoding_interval"],
        }
    ).fetch1("valid_times")

    decoding_interval = (
        IntervalList
        & {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["decoding_interval"],
        }
    ).fetch1("valid_times")

    return (
        min(
            np.asarray(encoding_interval).min(),
            np.asarray(decoding_interval).min(),
        ),
        max(
            np.asarray(encoding_interval).max(),
            np.asarray(decoding_interval).max(),
        ),
    )
