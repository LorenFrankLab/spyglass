"""Pose-data read-path helpers: canonicalization and DataFrame conversion.

Tool-agnostic utilities that reconcile tool-native body-part names with the
curated ``BodyPart`` namespace and build the shared 3-level column MultiIndex
``(scorer, bodyparts, coords)`` used across the pipeline.
"""

from difflib import get_close_matches

import datajoint as dj
import pandas as pd

from spyglass.position.v2.utils.skeleton import canonicalize, normalize_label


def canonicalize_pose_columns(pose_df, bodyparts, canon_map):
    """Map tool-native body-part names to the canonical Spyglass namespace.

    The single boundary where names emitted by a pose tool (from its model
    config / inference output) are reconciled with the curated ``BodyPart``
    spelling, so every downstream stage speaks one namespace. Only the column
    labels and the bodyparts list are remapped; the underlying pose data is
    untouched. Tool-agnostic: operates on the shared 3-level MultiIndex used by
    DLC and SLEAP alike.

    Parameters
    ----------
    pose_df : pandas.DataFrame
        Pose data with a 3-level column MultiIndex
        ``(scorer, bodyparts, coords)``.
    bodyparts : list[str]
        Tool-native body-part names, in column order.
    canon_map : dict[str, str]
        Mapping from :func:`~spyglass.position.v2.utils.skeleton.build_canonical_map`,
        built from the ``BodyPart`` table.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        The relabeled DataFrame and the canonical bodyparts list (column
        order preserved).

    Raises
    ------
    datajoint.errors.DataJointError
        If any name has no canonical match, or two surface forms collapse to
        the same canonical spelling (which would create a duplicate column).
    """
    mapping = {bp: canonicalize(bp, canon_map) for bp in bodyparts}

    unresolved = [bp for bp, canon in mapping.items() if canon is None]
    if unresolved:
        hints = "; ".join(
            f"{bp!r} (did you mean "
            f"{[canon_map[m] for m in get_close_matches(normalize_label(bp), list(canon_map), n=3)] or 'no close match'})"
            for bp in unresolved
        )
        raise dj.DataJointError(
            f"Body part name(s) not in BodyPart: {hints}. Correct the model "
            "config, ask an admin to add them, or re-run the import with "
            "normalize_names=True to rewrite the project to canonical names."
        )

    canonical = [mapping[bp] for bp in bodyparts]
    dupes = sorted({c for c in canonical if canonical.count(c) > 1})
    if dupes:
        raise dj.DataJointError(
            f"Body part names collapse to the same canonical spelling: "
            f"{dupes}. The model config has multiple surface forms of one "
            "part; rename them so each maps to a distinct body part."
        )

    return pose_df.rename(columns=mapping, level="bodyparts"), canonical


def pose_estimation_to_dataframe(pose_estimation, scorer, is_3d, canon_map):
    """Build a pose DataFrame from an ndx-pose ``PoseEstimation`` object.

    Reconstructs the 3-level column MultiIndex ``(scorer, bodyparts, coords)``
    used across the pipeline, resolving each series' body-part name to its
    canonical spelling on read. This is the back-compat counterpart to the
    write-side ingest boundary: NWB written with tool-native (or older)
    spellings reads back canonical wherever a match exists. Names with **no**
    canonical match are left unchanged, so reading never fails on a legacy or
    unknown part.

    Parameters
    ----------
    pose_estimation : ndx_pose.PoseEstimation
        Object whose ``pose_estimation_series`` provides per-bodypart data,
        confidence, and timestamps.
    scorer : str
        Scorer label for the first MultiIndex level.
    is_3d : bool
        When True, a ``z`` coordinate is read from series with >= 3 columns.
    canon_map : dict[str, str]
        Mapping from
        :func:`~spyglass.position.v2.utils.skeleton.build_canonical_map`.

    Returns
    -------
    pandas.DataFrame
        Pose data indexed by real per-frame timestamps (seconds).
    """
    data_dict = {}
    for series in pose_estimation.pose_estimation_series.values():
        # Strip only the trailing "_pose" suffix (write-side uses
        # f"{bodypart}_pose"); replace() would corrupt an internal "_pose".
        bodypart = series.name.removesuffix("_pose")
        bodypart = canonicalize(bodypart, canon_map, default=bodypart)

        pose_data = series.data[:]
        data_dict[(scorer, bodypart, "x")] = pose_data[:, 0]
        data_dict[(scorer, bodypart, "y")] = pose_data[:, 1]
        if is_3d and pose_data.shape[1] >= 3:
            data_dict[(scorer, bodypart, "z")] = pose_data[:, 2]

        data_dict[(scorer, bodypart, "likelihood")] = series.confidence[:]

    df = pd.DataFrame(data_dict)
    df.columns = pd.MultiIndex.from_tuples(
        df.columns, names=["scorer", "bodyparts", "coords"]
    )

    # Use real per-frame timestamps (seconds) as the index so that
    # compute_pose_outputs derives the correct sampling_rate and computes
    # velocity in cm/s rather than cm/frame.
    first_series = list(pose_estimation.pose_estimation_series.values())[0]
    df.index = pd.Index(first_series.timestamps[:], name="time")
    return df
