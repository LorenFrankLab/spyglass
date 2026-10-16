import datajoint as dj

from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("common_region")


@schema
class BrainRegion(SpyglassMixin, dj.Lookup):
    definition = """
    region_id: smallint auto_increment
    ---
    region_name: varchar(200)             # the name of the brain region
    subregion_name=NULL: varchar(200)     # subregion name
    subsubregion_name=NULL: varchar(200)  # subregion within subregion
    """

    @classmethod
    def names_key(
        cls,
        region_name: str,
        subregion_name: str = None,
        subsubregion_name: str = None,
    ) -> dict:
        """Return the name triple that identifies a region.

        `region_id` is an `auto_increment` surrogate, so the triple is what
        callers actually know a region by.

        Parameters
        ----------
        region_name : str
            The name of the brain region.
        subregion_name : str, optional
            The name of the subregion within the brain region.
        subsubregion_name : str, optional
            The name of the subregion within the subregion.

        Returns
        -------
        dict
        """
        return dict(
            region_name=region_name,
            subregion_name=subregion_name,
            subsubregion_name=subsubregion_name,
        )

    @classmethod
    def fetch_id(cls, **names) -> int:
        """Return the stored id for a name triple, or None. Writes nothing.

        Parameters
        ----------
        **names
            As `names_key` returns.

        Returns
        -------
        int or None
            None when no such region is stored.
        """
        found = (cls & cls.names_key(**names)).fetch("region_id")
        return int(found[0]) if len(found) else None

    @classmethod
    def next_id(cls) -> int:
        """Return the id an insert would assign next. Writes nothing.

        A prediction, not a reservation: nothing stops a concurrent session
        taking it. `resolve_on_insert` is what reconciles that.

        Returns
        -------
        int
        """
        return cls()._auto_increment(dict(), pk="region_id")["region_id"]

    @classmethod
    def clear_reservations(cls) -> None:
        """Forget the ids reserved for regions not yet written.

        Called once per plan. Reservations are process-level because several
        tables reserve from one pool: two tables keeping separate pools would
        predict the same id for different regions.
        """
        cls._reserved = dict()

    @classmethod
    def reserve_id(cls, **names) -> int:
        """Return the id for a name triple, reserving one if it is new.

        Reads to find a stored region, and otherwise predicts the id an insert
        would assign. Writes nothing: planning must not insert.

        Parameters
        ----------
        **names
            As `names_key` accepts.

        Returns
        -------
        int
        """
        key = cls.names_key(**names)
        if (found := cls.fetch_id(**key)) is not None:
            return found

        reserved = getattr(cls, "_reserved", None)
        if reserved is None:
            reserved = cls._reserved = dict()

        signature = tuple(sorted(key.items()))
        if signature in reserved:
            return reserved[signature]

        # One past the highest this run has seen, stored or reserved, so two
        # new regions do not land on the same id.
        region_id = max([cls.next_id(), *(v + 1 for v in reserved.values())])
        reserved[signature] = region_id
        return region_id

    @classmethod
    def reserved_rows(cls) -> list:
        """Return the rows for every region reserved and not yet written."""
        return [
            dict(signature) | {"region_id": region_id}
            for signature, region_id in getattr(
                cls, "_reserved", dict()
            ).items()
            if cls.fetch_id(**dict(signature)) is None
        ]

    @classmethod
    def resolve_on_insert(cls, rows) -> dict:
        """Insert planned regions, returning `{planned_id: actual_id}`.

        Called inside the insert transaction, so the lookup it does cannot be
        raced by the time the row lands. A planned id already taken by another
        session is dropped in favour of whatever `auto_increment` assigns;
        skipping an id is the accepted cost.

        Parameters
        ----------
        rows : sequence of dict
            Planned `BrainRegion` rows, each carrying a `region_id` the plan
            chose and the name triple.

        Returns
        -------
        dict
            Planned id to actual id, for every row whose id moved. Empty when
            every planned id survived.
        """
        remap = {}
        for row in rows:
            names = cls.names_key(
                region_name=row["region_name"],
                subregion_name=row.get("subregion_name"),
                subsubregion_name=row.get("subsubregion_name"),
            )
            actual = cls.fetch_id(**names)
            if actual is None:
                # Without `region_id`, so `auto_increment` assigns a free one
                # whatever happened since the plan was made.
                cls.insert1(names)
                actual = cls.fetch_id(**names)
            if actual != row["region_id"]:
                remap[row["region_id"]] = actual
        return remap

    @classmethod
    def fetch_add(cls, *args, **kwargs) -> int:
        """Return the region id for a name triple, adding it if absent.

        Parameters
        ----------
        See `names_key`.

        Returns
        -------
        int
        """
        names = cls.names_key(*args, **kwargs)
        if (found := cls.fetch_id(**names)) is not None:
            return found
        cls.insert1(names)
        return cls.fetch_id(**names)
