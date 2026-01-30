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
    _comparison_to_function,
)

schema = dj.schema("cbroz_bugs")


@schema
class Bug1281(dj.Computed):
    definition = """
    -> AutomaticCuration
    ---
    return_bug: bool     # Bug A: early return after first metric
    list_bug: bool       # Bug B: list aliasing across units
    dupe_bug: bool       # Bug C: duplicate label comparison
    has_downstream: bool # CuratedSpikeSorting depends on this
    """

    _return_bug_impact_date = datetime(2025, 4, 22)

    # -- Normalization ------------------------------------------------

    @staticmethod
    def _normalize_labels(labels):
        """Return labels dict with all keys cast to int.

        Quality metrics loaded from JSON have string keys, while
        the fixed ``get_labels`` uses ``int(unit_id)``.  Normalize
        to int so comparisons are consistent regardless of source.

        Parameters
        ----------
        labels : dict
            ``{unit_id: [label, ...]}`` with str or int keys.

        Returns
        -------
        dict
            Same structure with all keys as ``int``.
        """
        return {int(k): v for k, v in labels.items()}

    # -- Fetch helpers ------------------------------------------------

    @staticmethod
    def _fetch_auto_curation_key(key):
        """Return ``auto_curation_key`` blob."""
        return (AutomaticCuration & key).fetch1("auto_curation_key")

    @staticmethod
    def _fetch_label_params(key):
        """Return ``label_params`` dict."""
        params = (AutomaticCuration & key) * AutomaticCurationParameters
        return params.fetch1("label_params")

    @staticmethod
    def _fetch_quality_metrics(key):
        """Load quality metrics JSON, or None if missing."""
        metrics_path = (QualityMetrics & key).fetch1("quality_metrics_path")
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    @staticmethod
    def _has_downstream(auto_curation_key):
        """Check if a CuratedSpikeSortingSelection entry exists."""
        return len(CuratedSpikeSortingSelection & auto_curation_key) > 0

    # -- Label computation --------------------------------------------

    @classmethod
    def _compute_expected(cls, parent_labels, label_params, quality_metrics):
        """Compute fully-fixed labels, normalized to int keys.

        Parameters
        ----------
        parent_labels : dict
            Labels from the parent Curation (int-normalized).
        label_params : dict
            Label parameter rules.
        quality_metrics : dict
            Quality metrics loaded from JSON.

        Returns
        -------
        dict
            Recomputed labels with int keys.
        """
        expected = AutomaticCuration.get_labels(
            sorting=None,
            parent_labels=deepcopy(parent_labels),
            quality_metrics=quality_metrics,
            label_params=label_params,
        )
        return cls._normalize_labels(expected)

    @staticmethod
    def _get_labels_buggy(
        parent_labels,
        quality_metrics,
        label_params,
        bug_a=False,
        bug_b=False,
        bug_c=False,
    ):
        """Run labeling logic with specified bugs enabled.

        Each flag re-introduces one historical bug while leaving the
        rest of the logic fixed.  Caller must ``deepcopy`` both
        *parent_labels* and *label_params* before calling, because
        Bug B mutates label_params through aliased references.

        All keys are normalized to ``int`` so that bug detection is
        not confounded by key-type mismatches.

        Parameters
        ----------
        parent_labels : dict
            Starting labels (int keys). Will be mutated.
        quality_metrics : dict
            Quality metrics (string keys from JSON).
        label_params : dict
            Label parameter rules.
        bug_a : bool
            If True, return inside the ``for metric`` loop.
        bug_b : bool
            If True, skip ``.copy()`` on ``label[2]``.
        bug_c : bool
            If True, use list-in-list comparison + ``.extend()``.

        Returns
        -------
        dict
            Labels dict with int keys.
        """
        if not label_params:
            return parent_labels

        for metric in label_params:
            if metric not in quality_metrics:
                continue

            compare = _comparison_to_function[label_params[metric][0]]

            for unit_id in quality_metrics[metric]:
                label = label_params[metric]
                uid = int(unit_id)

                if not compare(quality_metrics[metric][unit_id], label[1]):
                    continue

                if uid not in parent_labels:
                    if bug_b:
                        parent_labels[uid] = label[2]
                    else:
                        parent_labels[uid] = label[2].copy()
                else:
                    if bug_c:
                        if label[2] not in parent_labels[uid]:
                            parent_labels[uid].extend(label[2])
                    else:
                        if "accept" in parent_labels[uid]:
                            parent_labels[uid].remove("accept")
                        for element in label[2].copy():
                            if element not in parent_labels[uid]:
                                parent_labels[uid].append(element)

            if bug_a:
                return parent_labels

        return parent_labels

    # -- Per-bug detection --------------------------------------------

    @classmethod
    def _detect_return_bug(
        cls,
        auto_curation_key,
        parent_labels,
        label_params,
        quality_metrics,
        expected,
    ):
        """Bug A: would early return produce different labels?

        Only possible when >1 metric overlaps with quality_metrics
        AND the Curation was created on or after the date Bug A was
        introduced (PR #1281, 2025-04-22).
        """
        time_of_creation = (Curation & auto_curation_key).fetch1(
            "time_of_creation"
        )
        if time_of_creation < cls._return_bug_impact_date.timestamp():
            return False
        overlap = set(label_params) & set(quality_metrics)
        if len(overlap) <= 1:
            return False
        buggy = cls._get_labels_buggy(
            deepcopy(parent_labels),
            quality_metrics,
            deepcopy(label_params),
            bug_a=True,
        )
        return cls._normalize_labels(buggy) != expected

    @classmethod
    def _detect_list_bug(
        cls, parent_labels, label_params, quality_metrics, expected
    ):
        """Bug B: would list aliasing cause cross-unit leakage?

        Aliasing manifests when multiple units match the same
        metric (sharing a list object) and at least one of those
        units also matches a subsequent metric, causing the
        append to propagate to all aliased units.
        """
        buggy = cls._get_labels_buggy(
            deepcopy(parent_labels),
            quality_metrics,
            deepcopy(label_params),
            bug_b=True,
        )
        return cls._normalize_labels(buggy) != expected

    @classmethod
    def _detect_dupe_bug(
        cls, parent_labels, label_params, quality_metrics, expected
    ):
        """Bug C: would list-in-list comparison create duplicates?

        The old code checked ``label[2] not in parent_labels[uid]``
        (always True for flat string lists), so ``.extend()``
        always ran, producing duplicate labels.
        """
        buggy = cls._get_labels_buggy(
            deepcopy(parent_labels),
            quality_metrics,
            deepcopy(label_params),
            bug_c=True,
        )
        return cls._normalize_labels(buggy) != expected

    # -- Comparison helper --------------------------------------------

    @classmethod
    def _compare_labels(cls, stored_labels, expected_labels):
        """Return per-unit diffs between stored and expected.

        Both dicts are normalized to int keys before comparison.

        Parameters
        ----------
        stored_labels : dict
            Labels from the Curation row.
        expected_labels : dict
            Labels from the fixed ``get_labels``.

        Returns
        -------
        dict
            ``{uid: {"stored": [...], "expected": [...]}}`` for
            units whose labels differ.
        """
        stored = cls._normalize_labels(stored_labels)
        expected = cls._normalize_labels(expected_labels)
        all_uids = set(stored) | set(expected)
        diffs = {}
        for uid in all_uids:
            s = stored.get(uid)
            e = expected.get(uid)
            if s != e:
                diffs[uid] = {"stored": s, "expected": e}
        return diffs

    # -- Core methods -------------------------------------------------

    def _insert_clean(self, key, has_downstream=False):
        """Insert an unaffected entry."""
        self.insert1(
            {
                **key,
                "return_bug": False,
                "list_bug": False,
                "dupe_bug": False,
                "has_downstream": has_downstream,
            }
        )

    def make(self, key):
        # --- Early return: empty label_params ---
        label_params = self._fetch_label_params(key)
        if not label_params:
            self._insert_clean(key)
            return

        auto_curation_key = self._fetch_auto_curation_key(key)

        # --- Early return: missing quality metrics file ---
        quality_metrics = self._fetch_quality_metrics(key)
        if quality_metrics is None:
            self._insert_clean(key)
            return

        # --- Normalize parent labels to int keys ---
        parent_curation = (Curation & key).fetch(as_dict=True)[0]
        parent_labels = self._normalize_labels(
            parent_curation["curation_labels"]
        )

        # --- Compute fully-fixed expected labels ---
        expected = self._compute_expected(
            parent_labels, label_params, quality_metrics
        )

        # --- Detect each bug independently ---
        return_bug = self._detect_return_bug(
            auto_curation_key,
            parent_labels,
            label_params,
            quality_metrics,
            expected,
        )
        list_bug = self._detect_list_bug(
            parent_labels, label_params, quality_metrics, expected
        )
        dupe_bug = self._detect_dupe_bug(
            parent_labels, label_params, quality_metrics, expected
        )
        has_downstream = self._has_downstream(auto_curation_key)

        self.insert1(
            {
                **key,
                "return_bug": return_bug,
                "list_bug": list_bug,
                "dupe_bug": dupe_bug,
                "has_downstream": has_downstream,
            }
        )

    def inspect(self, key):
        """Print detailed diagnostics for one AutomaticCuration entry.

        Parameters
        ----------
        key : dict
            Primary key to AutomaticCuration (and thus Bug1281).
        """
        # --- Bug1281 record ---
        row = (self & key).fetch(as_dict=True)
        if row:
            row = row[0]
            print("=== Bug1281 record ===")
            print(f"  return_bug (A): {row['return_bug']}")
            print(f"  list_bug   (B): {row['list_bug']}")
            print(f"  dupe_bug   (C): {row['dupe_bug']}")
            print(f"  has_downstream: {row['has_downstream']}")
        else:
            print("=== Bug1281 record: not yet populated ===")
            return

        # --- label_params ---
        label_params = self._fetch_label_params(key)
        print(f"\n=== label_params " f"({len(label_params)} metric(s)) ===")
        for metric, rule in label_params.items():
            print(f"  {metric}: {rule[0]} {rule[1]} -> {rule[2]}")

        # --- quality_metrics overlap ---
        quality_metrics = self._fetch_quality_metrics(key)
        if quality_metrics is None:
            print("\n=== Quality metrics: FILE NOT FOUND ===")
            return

        overlap = sorted(set(label_params) & set(quality_metrics))
        missing_from_qm = sorted(set(label_params) - set(quality_metrics))
        print("\n=== Metric overlap ===")
        print(f"  label_params:  {sorted(label_params.keys())}")
        print(f"  quality_metrics: " f"{sorted(quality_metrics.keys())}")
        print(f"  overlap ({len(overlap)}): {overlap}")
        if missing_from_qm:
            print(f"  skipped (not in qm): {missing_from_qm}")

        # --- Stored vs expected labels ---
        parent_curation = (Curation & key).fetch(as_dict=True)[0]
        parent_labels = self._normalize_labels(
            parent_curation["curation_labels"]
        )
        expected = self._compute_expected(
            parent_labels, label_params, quality_metrics
        )
        auto_curation_key = self._fetch_auto_curation_key(key)
        stored_labels = (Curation & auto_curation_key).fetch1("curation_labels")
        diffs = self._compare_labels(stored_labels, expected)

        stored_norm = self._normalize_labels(stored_labels)
        print("\n=== Label comparison ===")
        print(f"  total units (stored):   {len(stored_norm)}")
        print(f"  total units (expected): {len(expected)}")
        print(f"  units with differences: {len(diffs)}")
        if diffs:
            print(f"\n  {'unit':>8}  " f"{'stored':<30} {'expected':<30}")
            print(f"  {'----':>8}  " f"{'------':<30} {'--------':<30}")
            for uid in sorted(diffs):
                d = diffs[uid]
                s_str = str(d["stored"]) if d["stored"] is not None else "---"
                e_str = (
                    str(d["expected"]) if d["expected"] is not None else "---"
                )
                print(f"  {uid:>8}  {s_str:<30} {e_str:<30}")

        # --- Downstream impact ---
        has_downstream = self._has_downstream(auto_curation_key)
        print("\n=== Downstream ===")
        print(f"  CuratedSpikeSortingSelection: {has_downstream}")
        if has_downstream:
            n_units = len(CuratedSpikeSorting.Unit & auto_curation_key)
            print(f"  CuratedSpikeSorting.Unit rows: {n_units}")

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
                        f"\n  Units with changed "
                        f"accept/reject status: "
                        f"{len(reject_changes)}"
                    )
                    for rc in reject_changes:
                        status = (
                            "SHOULD BE REJECTED " "(was accepted)"
                            if rc["should_reject"]
                            else "SHOULD BE ACCEPTED " "(was rejected)"
                        )
                        print(f"    unit {rc['unit']}: {status}")
                else:
                    print("\n  No units change accept/reject " "status")
