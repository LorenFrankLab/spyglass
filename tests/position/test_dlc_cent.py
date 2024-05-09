import pandas as pd


def test_centroid_fetch1_dataframe(sgp, populate_centroid, centroid_key):
    fetched_df = (sgp.v1.DLCCentroid & centroid_key).fetch1_dataframe()
    assert isinstance(fetched_df, pd.DataFrame)
    raise NotImplementedError
