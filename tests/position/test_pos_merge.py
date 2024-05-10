import pandas as pd
import pytest


def test_pos_merge(sgp, pos_merge, populate_dlc, dlc_key):
    fetched_df = (sgp.v1.PositionOutput.DLCPosV1() & dlc_key).fetch1_dataframe()
    assert isinstance(fetched_df, pd.DataFrame)
    raise NotImplementedError
