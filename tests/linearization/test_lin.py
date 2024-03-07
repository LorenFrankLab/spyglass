from datajoint.hash import key_hash


def test_fetch1_dataframe(lin_v1, lin_merge, lin_merge_key):
    hash_df = key_hash(
        (lin_merge & lin_merge_key).fetch1_dataframe().round(3).to_dict()
    )
    hash_exp = "883a7b8aa47931ae7b265660ca27b462"
    assert hash_df == hash_exp, "Dataframe differs from expected"


## Todo: Add more tests of this pipeline, not just the fetch1_dataframe method
