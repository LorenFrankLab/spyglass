import pytest
from datajoint.utils import to_camel_case


@pytest.fixture(scope="session")
def TableChain():
    from spyglass.utils.dj_graph import TableChain

    return TableChain


def full_to_camel(t):
    return to_camel_case(t.split(".")[-1].strip("`"))


def test_chain_str(chain):
    """Test that the str of a TableChain object is as expected."""
    chain = chain

    str_got = str(chain)
    str_exp = (
        full_to_camel(chain.parent)
        + chain._link_symbol
        + full_to_camel(chain.child)
    )

    assert str_got == str_exp, "Unexpected str of TableChain object."


def test_chain_repr(chain):
    """Test that the repr of a TableChain object is as expected."""
    repr_got = repr(chain)
    repr_ext = "Chain: " + chain._link_symbol.join(
        [full_to_camel(t) for t in chain.path]
    )
    assert repr_got == repr_ext, "Unexpected repr of TableChain object."


def test_chain_len(chain):
    """Test that the len of a TableChain object is as expected."""
    assert len(chain) == len(chain.path), "Unexpected len of TableChain."


def test_chain_getitem(chain):
    """Test getitem of TableChain object."""
    by_int = str(chain[0])
    by_str = str(chain[chain.restr_ft[0].full_table_name])
    assert by_int == by_str, "Getitem by int and str not equal."


def test_nolink_join(no_link_chain):
    assert no_link_chain.cascade() is None, "Unexpected join of no link chain."


def test_chain_str_no_link(no_link_chain):
    """Test that the str of a TableChain object with no link is as expected."""
    assert str(no_link_chain) == "No link", "Unexpected str of no link chain."
    assert repr(no_link_chain) == "No link", "Unexpected repr of no link chain."


def test_invalid_chain(TableChain):
    with pytest.raises(ValueError):
        TableChain()
