def test_session_validation_functions_from_targeted():
    """Basic instantiation path for Session."""
    from spyglass.common.common_session import Session

    session_table = Session()
    assert session_table is not None
