"""Temporary table to track impact of `AutomaticCuration.get_labels`

File to be deleted before merge.
"""

import json
from copy import deepcopy
from datetime import datetime

import datajoint as dj

from spyglass.spikesorting.v0.spikesorting_curation import (
    AutomaticCuration,
    AutomaticCurationParameters,
    CuratedSpikeSorting,
    CuratedSpikeSortingSelection,
    Curation,
    QualityMetrics,
)

schema = dj.schema("cbroz_bugs")


@schema
class Bug1281(dj.Computed):
    definition = """
    -> AutomaticCuration
    ---
    is_impacted: bool   # This key is impacted by bug 1281
    has_downstream: bool  # A CuratedSpikeSorting entry depends on this
    missing_metrics=null: blob  # List of metrics not applied
    """

    _impact_date = datetime(2025, 4, 22)

    # -- Helpers ------------------------------------------------------

    @staticmethod
    def _normalize_labels(labels):
        """Return labels dict with all keys cast to strings.

        Quality metrics loaded from JSON always have string keys, but
        DataJoint blob serialization may store them as ints.  Normalize
        to strings so comparisons are consistent.

        Parameters
        ----------
        labels : dict
            ``{unit_id: [label, ...]}`` with str or int keys.

        Returns
        -------
        dict
            Same structure with all keys as ``str``.
        """
        return {str(k): v for k, v in labels.items()}

    @staticmethod
    def _fetch_auto_curation_key(key):
        """Return ``auto_curation_key`` blob for an AutomaticCuration key."""
        return (AutomaticCuration & key).fetch1("auto_curation_key")

    @staticmethod
    def _fetch_label_params(key):
        """Return ``label_params`` dict for an AutomaticCuration key."""
        params = (AutomaticCuration & key) * AutomaticCurationParameters
        return params.fetch1("label_params")

    @staticmethod
    def _fetch_quality_metrics(key):
        """Load quality metrics JSON for a key, or None if missing."""
        metrics_path = (QualityMetrics & key).fetch1("quality_metrics_path")
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    @classmethod
    def _compute_expected(cls, key, label_params, quality_metrics):
        """Recompute labels using the fixed ``get_labels``.

        Parameters
        ----------
        key : dict
            Primary key to the parent Curation (same as AutomaticCuration
            primary key minus the auto_curation_key).
        label_params : dict
            Label parameter rules.
        quality_metrics : dict
            Quality metrics loaded from JSON.

        Returns
        -------
        expected_labels : dict
            Recomputed labels with normalized (string) keys.
        """
        parent_curation = (Curation & key).fetch(as_dict=True)[0]
        parent_labels = parent_curation["curation_labels"]
        expected = AutomaticCuration.get_labels(
            sorting=None,
            parent_labels=deepcopy(parent_labels),
            quality_metrics=quality_metrics,
            label_params=label_params,
        )
        return cls._normalize_labels(expected)

    @classmethod
    def _compare_labels(
        cls, stored_labels, expected_labels, label_params, quality_metrics
    ):
        """Compare stored vs expected labels, return diffs and metrics.

        Both dicts are normalized to string keys before comparison.

        Parameters
        ----------
        stored_labels : dict
            Labels fetched from the Curation row.
        expected_labels : dict
            Labels recomputed by the fixed ``get_labels``.
        label_params : dict
            Label parameter rules (metric -> [op, thresh, tags]).
        quality_metrics : dict
            Quality metrics loaded from JSON.

        Returns
        -------
        diffs : dict
            ``{unit_id: {"stored": [...], "expected": [...]}}`` for
            units whose labels differ.
        missing_metrics : list[str]
            Metrics whose labels are absent or wrong in stored data.
        """
        stored = cls._normalize_labels(stored_labels)
        expected = cls._normalize_labels(expected_labels)

        all_uids = set(stored.keys()) | set(expected.keys())

        diffs = {}
        for uid in all_uids:
            s = stored.get(uid)
            e = expected.get(uid)
            if s != e:
                diffs[uid] = {"stored": s, "expected": e}

        missing_metrics = []
        for metric in label_params:
            if metric not in quality_metrics:
                continue
            for unit_id in quality_metrics[metric]:
                uid = str(unit_id)
                if uid in all_uids and stored.get(uid) != expected.get(uid):
                    missing_metrics.append(metric)
                    break

        return diffs, missing_metrics

    @staticmethod
    def _has_downstream(auto_curation_key):
        """Check if a CuratedSpikeSortingSelection entry exists."""
        return len(CuratedSpikeSortingSelection & auto_curation_key) > 0

    # -- Core methods -------------------------------------------------

    def _insert(
        self,
        key,
        is_impacted=False,
        has_downstream=False,
        missing_metrics=None,
    ):
        self.insert1(
            {
                **key,
                "is_impacted": is_impacted,
                "has_downstream": has_downstream,
                "missing_metrics": missing_metrics,
            }
        )

    def make(self, key):
        # --- Early return 1: Empty label_params ---
        # No labels to compute, nothing to break.
        label_params = self._fetch_label_params(key)
        if not label_params:
            self._insert(key)
            return

        # --- Early return 2: Curation created before bug date ---
        # All three bugs (indentation, aliasing, duplicate comparison)
        # were introduced in PR #1281. Entries created before that
        # date used different code and are not affected.
        auto_curation_key = self._fetch_auto_curation_key(key)
        time_of_creation = (Curation & auto_curation_key).fetch1(
            "time_of_creation"
        )
        if time_of_creation < self._impact_date.timestamp():
            self._insert(key)
            return

        # --- Early return 3: Missing quality metrics file ---
        quality_metrics = self._fetch_quality_metrics(key)
        if quality_metrics is None:
            self._insert(key)
            return

        # --- Full check: Recompute labels and compare ---
        has_downstream = self._has_downstream(auto_curation_key)
        expected_labels = self._compute_expected(
            key, label_params, quality_metrics
        )
        stored_labels = (Curation & auto_curation_key).fetch1("curation_labels")

        _, missing_metrics = self._compare_labels(
            stored_labels, expected_labels, label_params, quality_metrics
        )

        is_impacted = len(missing_metrics) > 0
        self.insert1(
            {
                **key,
                "is_impacted": is_impacted,
                "has_downstream": has_downstream,
                "missing_metrics": (missing_metrics if is_impacted else None),
            }
        )

    def inspect(self, key):
        """Print detailed diagnostics for one AutomaticCuration entry.

        Parameters
        ----------
        key : dict
            Primary key to AutomaticCuration (and thus Bug1281).
        """
        # --- Fetch Bug1281 row if populated ---
        row = (self & key).fetch(as_dict=True)
        if row:
            row = row[0]
            print("=== Bug1281 record ===")
            print(f"  is_impacted:    {row['is_impacted']}")
            print(f"  has_downstream: {row['has_downstream']}")
            print(f"  missing_metrics: {row['missing_metrics']}")
        else:
            print("=== Bug1281 record: not yet populated ===")
            return

        # --- label_params ---
        label_params = self._fetch_label_params(key)
        print(f"\n=== label_params ({len(label_params)} metric(s)) ===")
        for metric, rule in label_params.items():
            print(f"  {metric}: {rule[0]} {rule[1]} -> {rule[2]}")

        # --- auto_curation_key & Curation timestamps ---
        auto_curation_key = self._fetch_auto_curation_key(key)
        time_of_creation = (Curation & auto_curation_key).fetch1(
            "time_of_creation"
        )
        created = datetime.fromtimestamp(time_of_creation)
        bug_date = self._impact_date
        print("\n=== Curation created ===")
        print(f"  {created}  (bug introduced {bug_date.date()})")
        print(
            f"  after bug date: " f"{time_of_creation >= bug_date.timestamp()}"
        )

        # --- quality_metrics overlap ---
        quality_metrics = self._fetch_quality_metrics(key)
        if quality_metrics is None:
            print("\n=== Quality metrics: FILE NOT FOUND ===")
            return

        overlap = sorted(set(label_params.keys()) & set(quality_metrics.keys()))
        missing_from_qm = sorted(
            set(label_params.keys()) - set(quality_metrics.keys())
        )
        print("\n=== Metric overlap ===")
        print(f"  label_params metrics:  {sorted(label_params.keys())}")
        print(f"  quality_metrics keys:  {sorted(quality_metrics.keys())}")
        print(f"  overlap ({len(overlap)}):  {overlap}")
        if missing_from_qm:
            print(f"  skipped (not in qm):   {missing_from_qm}")

        # --- Stored vs expected labels ---
        expected_labels = self._compute_expected(
            key, label_params, quality_metrics
        )
        stored_labels = (Curation & auto_curation_key).fetch1("curation_labels")

        diffs, _ = self._compare_labels(
            stored_labels, expected_labels, label_params, quality_metrics
        )

        stored_norm = self._normalize_labels(stored_labels)
        expected_norm = self._normalize_labels(expected_labels)

        print("\n=== Label comparison ===")
        print(f"  total units (stored):   {len(stored_norm)}")
        print(f"  total units (expected): {len(expected_norm)}")
        print(f"  units with differences: {len(diffs)}")
        if diffs:
            print(f"\n  {'unit':>8}  {'stored':<30} {'expected':<30}")
            print(f"  {'----':>8}  {'------':<30} {'--------':<30}")
            for uid in sorted(
                diffs,
                key=lambda x: int(x) if x.isdigit() else x,
            ):
                d = diffs[uid]
                s_str = str(d["stored"]) if d["stored"] is not None else "---"
                e_str = (
                    str(d["expected"]) if d["expected"] is not None else "---"
                )
                print(f"  {uid:>8}  {s_str:<30} {e_str:<30}")

        # --- Downstream impact ---
        has_downstream = self._has_downstream(auto_curation_key)
        print("\n=== Downstream ===")
        print(f"  CuratedSpikeSortingSelection entry: {has_downstream}")
        if has_downstream:
            n_units = len(CuratedSpikeSorting.Unit & auto_curation_key)
            print(f"  CuratedSpikeSorting.Unit rows:      {n_units}")

            # Show which units would change accept/reject status
            if diffs:
                reject_changes = []
                for uid, d in diffs.items():
                    was_reject = (
                        d["stored"] is not None and "reject" in d["stored"]
                    )
                    should_reject = (
                        d["expected"] is not None and "reject" in d["expected"]
                    )
                    if was_reject != should_reject:
                        reject_changes.append(
                            {
                                "unit": uid,
                                "was_rejected": was_reject,
                                "should_reject": should_reject,
                            }
                        )
                if reject_changes:
                    print(
                        f"\n  Units with changed accept/reject "
                        f"status: {len(reject_changes)}"
                    )
                    for rc in reject_changes:
                        status = (
                            "SHOULD BE REJECTED (was accepted)"
                            if rc["should_reject"]
                            else "SHOULD BE ACCEPTED (was rejected)"
                        )
                        print(f"    unit {rc['unit']}: {status}")
                else:
                    print(
                        "\n  No units change accept/reject status "
                        "(label differences are non-reject labels "
                        "only)"
                    )
