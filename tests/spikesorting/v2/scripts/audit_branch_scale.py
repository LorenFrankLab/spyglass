"""Opt-in curation-history scaling with real persisted branches."""

import json
from time import perf_counter


def test_branch_discovery_scaling(planted_two_unit_sort, monkeypatch):
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef

    root = CurationRef.from_key(
        CurationV2.insert_curation(planted_two_unit_sort, reuse_existing=True)
    )
    records = []
    queries = []
    query = CurationV2.connection.query

    def tracked(*args, **kwargs):
        queries.append(args[0])
        return query(*args, **kwargs)

    for count in (8, 32, 64):
        while len(root.children) < count:
            CurationV2.insert_curation(
                planted_two_unit_sort,
                parent_curation_id=root.curation_id,
                labels={0: ["accept"]},
                description=f"Audit branch {len(root.children)}",
            )
        root.visualize_lineage()  # warm schema imports and DataJoint headings
        queries.clear()
        with monkeypatch.context() as patch:
            patch.setattr(CurationV2.connection, "query", tracked)
            start = perf_counter()
            children = root.children
            tree = root.visualize_lineage()
            seconds = perf_counter() - start
        assert len(children) == count
        assert len(tree.splitlines()) == count + 1
        records.append(
            {
                "branches": count,
                "seconds": seconds,
                "sql_queries": len(queries),
                "tree_bytes": len(tree.encode()),
            }
        )
    print("BRANCH_SCALE", json.dumps(records))
    assert records[-1]["sql_queries"] <= records[0]["sql_queries"] + 5, (
        "Lineage rendering must batch its reads as the history grows"
    )
