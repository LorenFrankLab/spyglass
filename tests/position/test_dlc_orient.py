import pandas as pd
import pytest


@pytest.mark.skip(reason="Needs labeled data")
def test_orient_fetch1_dataframe(sgp, orient_key, populate_orient):
    fetched_df = (sgp.v1.DLCOrientation & orient_key).fetch1_dataframe()
    assert isinstance(fetched_df, pd.DataFrame)
    raise NotImplementedError
