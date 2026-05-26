"""Generate MEArec ground-truth fixtures and convert them to Spyglass NWB files.

This script is the manual one-shot that populates the cached simulated
recordings used by the modern-spike-sorting validation suite. It is **not** a
pytest test (no ``test_`` prefix) and is not run during normal test collection.

For each fixture it:

1. Registers the probe geometry with MEAutility (so MEArec can simulate it).
2. Generates a biophysical template library with NEURON (cached on disk).
3. Generates a ground-truth recording with planted spike trains.
4. Converts the recording to a Spyglass-ingestible NWB file.
5. Round-trips the NWB through Spyglass session ingestion to prove it works.
6. Records fixture provenance (versions, seeds, hashes) in a manifest.

It always runs against the isolated Docker test database via
``bootstrap_v2_test_environment`` and writes only under a temporary
``SPYGLASS_BASE_DIR``; it never touches production.

Usage
-----
    python tests/spikesorting/v2/fixtures/generate_mearec.py \\
        --base-dir tests/_data/spikesorting_v2 --database-prefix pytests

Pass ``--smoke`` for a fast, tiny pipeline check before a full generation run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import ProbeLayout

# The bootstrap must run before importing Spyglass, so resolve and import the
# standalone test-environment helper by path rather than as a package.
_THIS_DIR = Path(__file__).resolve().parent
_V2_TEST_DIR = _THIS_DIR.parent
if str(_V2_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_V2_TEST_DIR))

from test_env import bootstrap_v2_test_environment  # noqa: E402

# Spyglass and its `_fixtures` helpers are imported lazily inside functions,
# always after `bootstrap_v2_test_environment` has repointed `dj.config` at the
# isolated test database -- never at module import time.

_DEFAULT_BASE_DIR = "tests/_data/spikesorting_v2"
_DEFAULT_PREFIX = "pytests"


@dataclass(frozen=True)
class FixtureSpec:
    """One recording fixture to generate.

    Attributes
    ----------
    name : str
        Fixture name; also the output file stem and NWB ``session_id``.
    layout : ProbeLayout
        Probe geometry.
    duration_s : float
        Recording duration in seconds.
    n_exc, n_inh : int
        Number of excitatory / inhibitory ground-truth units.
    drifting : bool
        Whether to simulate slow electrode drift.
    drift_um_per_min : float
        Slow-drift speed when ``drifting`` is True.
    seed : int
        Deterministic seed shared by templates, spike trains, and convolution.
    """

    name: str
    layout: ProbeLayout
    duration_s: float
    n_exc: int
    n_inh: int
    drifting: bool
    drift_um_per_min: float
    seed: int


@dataclass
class GenProfile:
    """Tunable cost parameters for a generation run.

    The ``smoke`` profile is a cheap end-to-end pipeline check; the ``full``
    profile produces the fixtures the validation suite actually uses. Both use
    every bundled cell model -- the smoke speedup comes from generating far
    fewer templates per model, not from dropping models (the bundled cell
    models share one compiled-mechanism folder).
    """

    name: str
    n_templates_per_model: int
    template_n_jobs: int


def _profiles() -> dict[str, tuple[GenProfile, tuple[FixtureSpec, ...]]]:
    """Return ``{profile_name: (GenProfile, (FixtureSpec, ...))}`` mapping.

    Defined as a function rather than a module-level constant because building
    a ``ProbeLayout`` imports from ``spyglass.spikesorting.v2._fixtures``,
    which must happen only after ``bootstrap_v2_test_environment`` has run.
    """
    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        neuropixels_probe_layout,
        polymer_probe_layout,
    )

    polymer = polymer_probe_layout()
    neuropixels = neuropixels_probe_layout()
    return {
        "smoke": (
            GenProfile(
                name="smoke", n_templates_per_model=2, template_n_jobs=-1
            ),
            (
                FixtureSpec(
                    name="mearec_polymer_smoke",
                    layout=polymer,
                    duration_s=4.0,
                    n_exc=4,
                    n_inh=2,
                    drifting=False,
                    drift_um_per_min=0.0,
                    seed=0,
                ),
            ),
        ),
        "full": (
            GenProfile(
                name="full", n_templates_per_model=30, template_n_jobs=-1
            ),
            (
                FixtureSpec(
                    name="mearec_polymer_128ch_60s",
                    layout=polymer,
                    duration_s=60.0,
                    n_exc=17,
                    n_inh=7,
                    drifting=False,
                    drift_um_per_min=0.0,
                    seed=0,
                ),
                FixtureSpec(
                    name="mearec_neuropixels_60s",
                    layout=neuropixels,
                    duration_s=60.0,
                    n_exc=14,
                    n_inh=6,
                    drifting=False,
                    drift_um_per_min=0.0,
                    seed=1,
                ),
                FixtureSpec(
                    name="mearec_polymer_128ch_drift_120s",
                    layout=polymer,
                    duration_s=120.0,
                    n_exc=17,
                    n_inh=7,
                    drifting=True,
                    drift_um_per_min=5.0,
                    seed=2,
                ),
            ),
        ),
    }


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _meautility_probe_name(layout: ProbeLayout) -> str:
    """Return the MEAutility electrode name used for a probe layout."""
    return f"spyglass_v2_{layout.probe_type}".replace("-", "_")


def _register_meautility_probe(layout: ProbeLayout) -> str:
    """Register the probe geometry with MEAutility so MEArec can simulate it.

    MEArec resolves probes by name through MEAutility. The contact positions
    are mapped into MEAutility's ``yz`` probe plane: the probe's in-plane ``x``
    becomes the MEAutility ``z`` axis and the in-plane ``y`` becomes the
    MEAutility ``y`` axis, with depth ``x = 0``.

    Parameters
    ----------
    layout : ProbeLayout
        Probe geometry.

    Returns
    -------
    str
        The registered MEAutility electrode name.
    """
    import MEAutility as mu
    import yaml

    name = _meautility_probe_name(layout)
    if name in mu.return_mea_list():
        return name

    positions = [
        [0.0, float(contact.rel_y), float(contact.rel_x)]
        for contact in layout.contacts
    ]
    info = {
        "electrode_name": name,
        "description": f"Spyglass v2 validation probe: {layout.description}",
        "pos": positions,
        "center": False,
        "plane": "yz",
        "shape": "square",
        "size": layout.contact_size_um / 2.0,
        "type": "mea",
        "sortlist": None,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as handle:
        yaml.safe_dump(info, handle)
        yaml_path = handle.name
    mu.add_mea(yaml_path)
    Path(yaml_path).unlink(missing_ok=True)
    return name


def _cell_models_folder() -> Path:
    """Return the bundled MEArec cell-model folder.

    Returns
    -------
    pathlib.Path
    """
    import MEArec

    return Path(MEArec.get_default_cell_models_folder())


def _generate_templates(
    layout: ProbeLayout,
    probe_name: str,
    profile: GenProfile,
    templates_h5: Path,
    *,
    drifting: bool,
    seed: int,
) -> None:
    """Generate (or reuse) a biophysical template library for a probe.

    Parameters
    ----------
    layout : ProbeLayout
        Probe geometry (used only for logging).
    probe_name : str
        Registered MEAutility electrode name.
    profile : GenProfile
        Generation profile controlling template count and parallelism.
    templates_h5 : pathlib.Path
        Destination ``.h5``; reused if it already exists.
    drifting : bool
        Whether to simulate drifting templates.
    seed : int
        Deterministic seed for template positions and rotations.
    """
    import MEArec

    if templates_h5.exists():
        print(f"  templates: reusing cached {templates_h5.name}")
        return

    params = MEArec.get_default_templates_params()
    params["probe"] = probe_name
    params["n"] = profile.n_templates_per_model
    params["seed"] = seed
    params["drifting"] = drifting

    cell_models_folder = _cell_models_folder()
    # The bundled cell models ship as .mod source; compile them into a shared
    # `mods/` folder on the first run, then reuse the compiled mechanisms.
    recompile = not (cell_models_folder / "mods").is_dir()
    print(
        f"  templates: simulating {layout.probe_type} "
        f"(n={params['n']}/model, drifting={drifting}, recompile={recompile})"
    )
    started = time.time()
    tempgen = MEArec.gen_templates(
        cell_models_folder=str(cell_models_folder),
        params=params,
        parallel=True,
        n_jobs=profile.template_n_jobs,
        recompile=recompile,
        # Keep the per-cell EAP scratch: MEArec's delete_tmp cleanup does a
        # bare rmtree that fails on NFS ghost (.nfs*) files. The templates are
        # already loaded into memory by then, so skipping the cleanup is safe.
        delete_tmp=False,
        verbose=False,
    )
    templates_h5.parent.mkdir(parents=True, exist_ok=True)
    MEArec.save_template_generator(
        tempgen, filename=str(templates_h5), verbose=False
    )
    print(
        f"  templates: {len(tempgen.templates)} EAPs in "
        f"{time.time() - started:.1f}s -> {templates_h5.name}"
    )


def _generate_recording(
    spec: FixtureSpec,
    templates_h5: Path,
    recording_h5: Path,
) -> None:
    """Generate a ground-truth recording from a template library.

    Parameters
    ----------
    spec : FixtureSpec
        Fixture specification.
    templates_h5 : pathlib.Path
        Template library produced by :func:`_generate_templates`.
    recording_h5 : pathlib.Path
        Destination recording ``.h5``; reused if it already exists.
    """
    import MEArec

    if recording_h5.exists():
        print(f"  recording: reusing cached {recording_h5.name}")
        return

    params = MEArec.get_default_recordings_params()
    params["spiketrains"]["n_exc"] = spec.n_exc
    params["spiketrains"]["n_inh"] = spec.n_inh
    params["spiketrains"]["duration"] = spec.duration_s
    params["seeds"]["spiketrains"] = spec.seed
    params["seeds"]["templates"] = spec.seed
    params["seeds"]["convolution"] = spec.seed
    params["seeds"]["noise"] = spec.seed
    params["recordings"]["drifting"] = spec.drifting
    if spec.drifting:
        # MEArec slow-drift speed is in um/min.
        params["recordings"]["slow_drift_velocity"] = spec.drift_um_per_min

    print(
        f"  recording: {spec.duration_s:.0f}s, "
        f"{spec.n_exc + spec.n_inh} units, drifting={spec.drifting}"
    )
    started = time.time()
    recgen = MEArec.gen_recordings(
        params=params,
        templates=str(templates_h5),
        verbose=False,
    )
    recording_h5.parent.mkdir(parents=True, exist_ok=True)
    MEArec.save_recording_generator(
        recgen, filename=str(recording_h5), verbose=False
    )
    print(
        f"  recording: {len(recgen.spiketrains)} units in "
        f"{time.time() - started:.1f}s -> {recording_h5.name}"
    )


def _write_probeinterface_json(layout: ProbeLayout, json_path: Path) -> str:
    """Write the probe geometry as a probeinterface JSON and return its hash.

    Parameters
    ----------
    layout : ProbeLayout
        Probe geometry.
    json_path : pathlib.Path
        Destination JSON path.

    Returns
    -------
    str
        SHA-256 digest of the written JSON.
    """
    from probeinterface import Probe, ProbeGroup, write_probeinterface

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(
        positions=layout.positions_um(),
        shapes="square",
        shape_params={"width": layout.contact_size_um},
        shank_ids=[str(c.shank_id) for c in layout.contacts],
    )
    probe.set_device_channel_indices(list(range(layout.n_contacts)))
    group = ProbeGroup()
    group.add_probe(probe)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    write_probeinterface(json_path, group)
    return _sha256(json_path)


def _verify_ingestion(nwb_path: Path, spec: FixtureSpec) -> dict:
    """Round-trip the fixture NWB through Spyglass session ingestion.

    Proves a freshly generated fixture ingests end to end: a ``Session``,
    ``Raw``, non-empty ``Electrode`` table, ``IntervalList`` rows, the probe
    tables, and the sidecar ``ground_truth/units`` processing-module
    table all appear.

    Parameters
    ----------
    nwb_path : pathlib.Path
        Fixture NWB file.
    spec : FixtureSpec
        Fixture specification (probe geometry, unit count).

    Returns
    -------
    dict
        Counts of the ingested rows, for the manifest.

    Raises
    ------
    AssertionError
        If any expected ingestion result is missing.
    """
    from spyglass.common import Electrode, IntervalList, Raw, Session
    from spyglass.common.common_device import Probe, ProbeType

    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

    nwb_file_name = copy_and_insert_nwb(nwb_path)
    session_key = {"nwb_file_name": nwb_file_name}

    assert len(Session & session_key) == 1, "Session row missing"
    assert len(Raw & session_key) == 1, "Raw row missing"
    n_electrodes = len(Electrode & session_key)
    assert n_electrodes == spec.layout.n_contacts, (
        f"Electrode count {n_electrodes} != {spec.layout.n_contacts}"
    )
    n_intervals = len(IntervalList & session_key)
    assert n_intervals > 0, "no IntervalList rows"

    probe_key = {"probe_id": spec.layout.probe_type}
    assert len(Probe & probe_key) == 1, "Probe row missing"
    probe_electrodes = len(
        Probe.Electrode & {"probe_id": spec.layout.probe_type}
    )
    assert probe_electrodes == spec.layout.n_contacts, (
        f"Probe.Electrode count {probe_electrodes} "
        f"!= {spec.layout.n_contacts}"
    )
    num_shanks = (ProbeType & {"probe_type": spec.layout.probe_type}).fetch1(
        "num_shanks"
    )
    assert num_shanks == spec.layout.n_shanks, (
        f"ProbeType.num_shanks {num_shanks} != {spec.layout.n_shanks}"
    )

    # Planted ground-truth units live in a sidecar
    # ``ProcessingModule("ground_truth")["units"]`` table (NOT
    # ``nwbfile.units``), so ``ImportedSpikeSorting`` -- which only
    # reads ``nwbfile.units`` -- has nothing to ingest from these
    # fixtures and is intentionally NOT exercised here. Verify
    # instead that the sidecar table is present and non-empty.
    import pynwb

    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )

    with pynwb.NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        assert nwb.units is None, (
            "Source NWB leaked planted units into nwbfile.units; "
            "they belong in the sidecar processing module so the v1 "
            "baseline-capture path can write to nwbfile.units freely."
        )
        gt_table = get_ground_truth_units_table(nwb)
        assert gt_table is not None, (
            "Source NWB missing sidecar ground-truth units table."
        )
        n_planted = int(len(gt_table.id[:]))

    return {
        "nwb_file_name": nwb_file_name,
        "n_electrodes": int(n_electrodes),
        "n_interval_lists": int(n_intervals),
        "n_probe_electrodes": int(probe_electrodes),
        "num_shanks": int(num_shanks),
        "ground_truth_units": n_planted,
    }


def generate_fixtures(
    base_dir: Path,
    *,
    profile_name: str,
    skip_ingestion: bool,
) -> Path:
    """Generate every fixture for a run and write the provenance manifest.

    Parameters
    ----------
    base_dir : pathlib.Path
        Resolved Spyglass base directory.
    profile_name : str
        Key into ``_profiles()`` -- ``"smoke"`` or ``"full"``.
    skip_ingestion : bool
        Skip the Spyglass ingestion round-trip (geometry/conversion only).

    Returns
    -------
    pathlib.Path
        The written manifest path.
    """
    from importlib.metadata import version as _pkg_version

    profile, specs = _profiles()[profile_name]
    fixtures_dir = _THIS_DIR
    work_dir = base_dir / "mearec_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "profile": profile.name,
        "mearec_version": _pkg_version("MEArec"),
        "neuroconv_version": _pkg_version("neuroconv"),
        "spikeinterface_version": _pkg_version("spikeinterface"),
        "fixtures": {},
    }

    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        mearec_to_spyglass_nwb,
    )

    for spec in specs:
        print(f"[{spec.name}]")
        probe_name = _register_meautility_probe(spec.layout)
        templates_tag = "drift" if spec.drifting else "static"
        templates_h5 = (
            work_dir
            / f"templates_{spec.layout.probe_type}_{profile.name}_{templates_tag}.h5"
        )
        recording_h5 = work_dir / f"{spec.name}.h5"
        nwb_path = fixtures_dir / f"{spec.name}.nwb"
        probe_json = work_dir / f"probe_{spec.layout.probe_type}.json"

        _generate_templates(
            spec.layout,
            probe_name,
            profile,
            templates_h5,
            drifting=spec.drifting,
            seed=spec.seed,
        )
        _generate_recording(spec, templates_h5, recording_h5)

        mearec_to_spyglass_nwb(
            recording_h5,
            nwb_path,
            fixture_name=spec.name,
            probe_layout=spec.layout,
        )
        print(f"  nwb: wrote {nwb_path.name}")

        probe_json_hash = _write_probeinterface_json(spec.layout, probe_json)

        fixture_record: dict = {
            "n_units": spec.n_exc + spec.n_inh,
            "duration_s": spec.duration_s,
            "n_channels": spec.layout.n_contacts,
            "probe_type": spec.layout.probe_type,
            "drifting": spec.drifting,
            "seed": spec.seed,
            "probe_json_sha256": probe_json_hash,
            "recording_h5_sha256": _sha256(recording_h5),
            "nwb_sha256": _sha256(nwb_path),
        }
        if not skip_ingestion:
            fixture_record["ingestion"] = _verify_ingestion(nwb_path, spec)
            print("  ingestion: round-trip OK")
        manifest["fixtures"][spec.name] = fixture_record

    manifest_path = fixtures_dir / "fixtures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest -> {manifest_path}")
    return manifest_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate MEArec ground-truth fixtures for v2 validation."
    )
    parser.add_argument(
        "--base-dir",
        default=_DEFAULT_BASE_DIR,
        help="Spyglass base directory (must be under tests/_data/ or a "
        "temporary directory).",
    )
    parser.add_argument(
        "--database-prefix",
        default=_DEFAULT_PREFIX,
        help="DataJoint schema prefix (default: pytests).",
    )
    parser.add_argument(
        "--dj-config",
        default=None,
        help="Optional path to an existing DataJoint config file.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast, tiny end-to-end pipeline check instead of the full "
        "fixture set.",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip the Spyglass ingestion round-trip (geometry/conversion "
        "only; no database needed).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: bootstrap the test environment and generate fixtures."""
    args = _parse_args(argv)
    bootstrap_v2_test_environment(
        base_dir=args.base_dir,
        database_prefix=args.database_prefix,
        dj_config=args.dj_config,
    )
    from spyglass.settings import base_dir as resolved_base_dir

    generate_fixtures(
        Path(resolved_base_dir),
        profile_name="smoke" if args.smoke else "full",
        skip_ingestion=args.skip_ingestion,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
