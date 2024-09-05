def test_metric_curation(spike_v1, pop_curation_metric):
    ret = spike_v1.CurationV1 & pop_curation_metric & "description LIKE 'a%'"
    assert len(ret) == 1, "CurationV1.insert_curation failed to insert a record"
