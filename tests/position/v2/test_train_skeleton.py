"""Tests for skeleton table (train.py)."""

import datajoint as dj
import networkx as nx
import pytest

INSERT_KWARGS = dict(skip_duplicates=True, accept_default=True)


class TestSkeletonInsert:
    """Test skeleton.insert1() functionality."""

    def test_insert_from_config_dict(self, skeleton):
        """Test inserting skeleton from config dictionary."""
        config = {
            "bodyparts": ["nose", "earL", "earR"],
            "skeleton": [("nose", "earL"), ("nose", "earR")],
        }

        skeleton_key = skeleton.insert1(config, **INSERT_KWARGS)

        assert "skeleton_id" in skeleton_key
        skeleton = (skeleton & skeleton_key).fetch1()
        assert set(skeleton["bodyparts"]) == set(config["bodyparts"])

    def test_insert_with_custom_skeleton_id(self, skeleton):
        """Test inserting skeleton with custom skeleton_id."""
        expected_id = "test-skel-001"
        config = {
            "skeleton_id": expected_id,
            "bodyparts": ["nose", "head"],
            "skeleton": [("nose", "head")],
        }

        skeleton_key = skeleton.insert1(config, **INSERT_KWARGS)

        assert skeleton_key["skeleton_id"] == expected_id

    def test_insert_duplicate_skeleton_skips(self, skeleton):
        """Test inserting duplicate skeleton with skip_duplicates."""
        config = {
            "bodyparts": ["nose", "head"],
            "skeleton": [("nose", "head")],
        }

        # Insert once
        key1 = skeleton.insert1(config, **INSERT_KWARGS)
        skeleton_count_before = len(skeleton)

        # Insert again
        key2 = skeleton.insert1(config, **INSERT_KWARGS)
        skeleton_count_after = len(skeleton)

        # Should not create duplicate
        assert skeleton_count_after == skeleton_count_before

        # Should return same skeleton_id (if not None)
        assert key1["skeleton_id"] == key2["skeleton_id"]

    def test_insert_validates_bodyparts(self, skeleton):
        """Test insert validates bodyparts exist in BodyPart table."""

        config = {
            "bodyparts": ["invalid_bodypart_xyz"],
            "skeleton": ["invalid_bodypart_xyz"],
        }

        with pytest.raises(dj.DataJointError, match="Unknown bodypart"):
            skeleton.insert1(config, accept_default=True)

    def test_insert_requires_bodyparts(self, skeleton):
        """Test insert fails without bodyparts field."""
        config = {"skeleton": [("nose", "head")]}

        with pytest.raises(dj.DataJointError, match="bodyparts"):
            skeleton.insert1(config, accept_default=True)

    def test_insert_no_edges(self, skeleton):
        """Test insert ok without edges field."""
        config = {"bodyparts": ["nose", "head"]}
        skeleton.insert1(config, accept_default=True)


class TestSkeletonTopology:
    """Test skeleton topology hash computation."""

    def test_shape_hash_same_for_equivalent_skeletons(self, skeleton):
        """Test topology hash is same for equivalent skeletons."""

        bodyparts = ["nose", "earL", "earR"]
        edges1 = [("nose", "earL"), ("nose", "earR")]
        edges2 = [("nose", "earR"), ("nose", "earL")]  # Different order

        hash1 = skeleton._shape_hash_from_edges(bodyparts, edges1)
        hash2 = skeleton._shape_hash_from_edges(bodyparts, edges2)

        # Same topology should give same hash (order-independent)
        assert hash1 == hash2

    def test_shape_hash_different_for_different_skeletons(self, skeleton):
        """Test topology hash differs for different skeletons."""

        bodyparts = ["nose", "earL", "earR", "tailbase"]
        edges1 = [("nose", "earL"), ("nose", "earR")]
        edges2 = [
            ("nose", "tailbase"),
            ("tailbase", "earL"),
            ("tailbase", "earR"),
        ]

        hash1 = skeleton._shape_hash_from_edges(bodyparts, edges1)
        hash2 = skeleton._shape_hash_from_edges(bodyparts, edges2)

        # Different topology should give different hash
        assert hash1 != hash2

    def test_build_labeled_graph_has_correct_parts(self, skeleton):
        """Test graph has correct number of nodes/edges."""
        bodyparts = ["nose", "earL", "earR"]
        edges = [("nose", "earL"), ("nose", "earR")]

        graph = skeleton._build_labeled_graph(bodyparts, edges)

        assert isinstance(graph, nx.Graph)
        assert len(graph.nodes) == len(bodyparts)
        assert len(graph.edges) == len(edges)


class TestSkeletonDuplicateDetection:
    """Test skeleton duplicate detection with graph isomorphism."""

    def test_fuzzy_equal_match(self, skeleton):
        """Test fuzzy matching for matches."""
        assert skeleton._fuzzy_equal("nose", "nose", threshold=0.85)
        assert skeleton._fuzzy_equal("Nose", "nose", threshold=0.85)
        assert skeleton._fuzzy_equal("LEFT_EAR", "left_ear", threshold=0.85)
        assert skeleton._fuzzy_equal("leftear", "left_ear", threshold=0.85)
        assert not skeleton._fuzzy_equal("nose", "tailbase", threshold=0.85)


class TestSkeletonEdgeCases:
    """Test edge cases and error handling for skeleton."""

    def test_single_bodypart_single_edge(self, skeleton):
        """Test skeleton with self-edge."""
        config = {
            "bodyparts": ["nose"],
            "skeleton": [("nose", "nose")],
        }
        try:
            skeleton.insert1(config, **INSERT_KWARGS)
        except Exception as e:
            # If it fails, should be a clear DataJoint error
            assert "DataJointError" in str(
                type(e).__name__
            ) or "ValueError" in str(type(e).__name__)

    def test_disconnected_graph(self, skeleton):
        """Test skeleton with disconnected components."""
        config = {
            "bodyparts": ["nose", "earL", "earR", "tailBase"],
            "skeleton": [
                ("nose", "earL"),  # Component 1
                ("earR", "tailBase"),  # Component 2 (disconnected)
            ],
        }

        # Should handle disconnected graphs (may be valid for some animals)
        skeleton_key = skeleton.insert1(config, **INSERT_KWARGS)

        if skeleton_key:
            skeleton = (skeleton & skeleton_key).fetch1()
            assert len(skeleton["edges"]) == 2
