from inspect import isclass as inspect_isclass

import datajoint as dj

from spyglass.utils import SpyglassMixin

# Ranges are offset from one another to create unique list of entries for each
# table while respecting the foreign key constraints.

parent_id = range(10)
parent_attr = [i + 10 for i in range(2, 12)]

other_id = range(9)
other_attr = [i + 10 for i in range(3, 12)]

intermediate_id = range(2, 10)
intermediate_attr = [i + 10 for i in range(4, 12)]

pk_id = range(3, 10)
pk_attr = [i + 10 for i in range(5, 12)]

sk_id = range(6)
sk_attr = [i + 10 for i in range(6, 12)]

pk_sk_id = range(5)
pk_sk_attr = [i + 10 for i in range(7, 12)]

pk_alias_id = range(4)
pk_alias_attr = [i + 10 for i in range(8, 12)]

sk_alias_id = range(3)
sk_alias_attr = [i + 10 for i in range(9, 12)]


def offset(gen, offset):
    return list(gen)[offset:]


class ParentNode(SpyglassMixin, dj.Lookup):
    definition = """
    parent_id: int
    ---
    parent_attr : int
    """
    contents = [(i, j) for i, j in zip(parent_id, parent_attr)]


class OtherParentNode(SpyglassMixin, dj.Lookup):
    definition = """
    other_id: int
    ---
    other_attr : int
    """
    contents = [(i, j) for i, j in zip(other_id, other_attr)]


class IntermediateNode(SpyglassMixin, dj.Lookup):
    definition = """
    intermediate_id: int
    ---
    -> ParentNode
    intermediate_attr : int
    """
    contents = [
        (i, j, k)
        for i, j, k in zip(
            intermediate_id, offset(parent_id, 1), intermediate_attr
        )
    ]


class PkNode(SpyglassMixin, dj.Lookup):
    definition = """
    pk_id: int
    -> IntermediateNode
    ---
    pk_attr : int
    """
    contents = [
        (i, j, k) for i, j, k in zip(pk_id, offset(intermediate_id, 2), pk_attr)
    ]


class SkNode(SpyglassMixin, dj.Lookup):
    definition = """
    sk_id: int
    ---
    -> IntermediateNode
    sk_attr : int
    """
    contents = [
        (i, j, k) for i, j, k in zip(sk_id, offset(intermediate_id, 3), sk_attr)
    ]


class PkSkNode(SpyglassMixin, dj.Lookup):
    definition = """
    pk_sk_id: int
    -> IntermediateNode
    ---
    -> OtherParentNode
    pk_sk_attr : int
    """
    contents = [
        (i, j, k, m)
        for i, j, k, m in zip(
            pk_sk_id, offset(intermediate_id, 4), other_id, pk_sk_attr
        )
    ]


class PkAliasNode(SpyglassMixin, dj.Lookup):
    definition = """
    pk_alias_id: int
    -> PkNode.proj(fk_pk_id='pk_id')
    ---
    pk_alias_attr : int
    """
    contents = [
        (i, j, k, m)
        for i, j, k, m in zip(
            pk_alias_id,
            offset(pk_id, 1),
            offset(intermediate_id, 3),
            pk_alias_attr,
        )
    ]


class SkAliasNode(SpyglassMixin, dj.Lookup):
    definition = """
    sk_alias_id: int
    ---
    -> SkNode.proj(fk_sk_id='sk_id')
    -> PkSkNode
    sk_alias_attr : int
    """
    contents = [
        (i, j, k, m, n)
        for i, j, k, m, n in zip(
            sk_alias_id,
            offset(sk_id, 2),
            offset(pk_sk_id, 1),
            offset(intermediate_id, 5),
            sk_alias_attr,
        )
    ]


LOCALS_GRAPH = {
    k: v
    for k, v in locals().items()
    if inspect_isclass(v) and k != "SpyglassMixin"
}
__all__ = list(LOCALS_GRAPH)
