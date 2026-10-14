"""Exercise container scripts and portable output with a controlled client.

Only the container transport and package installer are stubbed. Recording
serialization, the generated script, SI save/load, dispatch cleanup and runtime
provenance use their real implementations; no Docker or database is needed.
"""

import runpy
import sys
import uuid

import numpy as np
import pytest
import spikeinterface as si
import spikeinterface.sorters as sis
from spikeinterface.sorters import container_tools
from spikeinterface.sorters.sorterlist import sorter_dict

from spyglass.spikesorting.v2._sorting.container import (
    RUNTIME_ANNOTATION,
    run_sorter_container,
)
from spyglass.spikesorting.v2._sorting.dispatch import (
    run_si_sorter,
    sort_runtime_versions,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from spyglass import settings

    monkeypatch.setattr(settings, "temp_dir", str(tmp_path))
    recording = si.NumpyRecording(
        [np.zeros((100, 4), dtype=np.float32)], 30_000
    )
    recording = recording.save(folder=tmp_path / "input", progress_bar=False)

    class FakeSorter:
        gpu_capability = "not-supported"

        @staticmethod
        def use_gpu(params):
            return False

        @staticmethod
        def get_sorter_version():
            return "1.0.5"

    class Client:
        version = "0.103.0"
        install_succeeds = True
        fail_sort = False
        change_version_at_sort = False
        started = False
        stopped = False
        script_executed = False

        def __init__(self, mode, image, volumes, user_base, gpu):
            self.mode, self.image = mode, image
            self.volumes, self.user_base, self.gpu = volumes, user_base, gpu
            self.installs = []

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def run_command(self, command):
            if command[0] != "python":
                return ""
            if command[1] == "-c":
                return (
                    f"SPYGLASS_SI_VERSION={self.version}\n"
                    if self.version is not None
                    else "ModuleNotFoundError: No module named 'spikeinterface'"
                )
            if command[1] == "-m":
                return "pip upgraded"
            self.script_executed = True
            if self.change_version_at_sort:
                self.version = "0.103.0"
            before_jobs = si.get_global_job_kwargs().copy()
            before_inf = hasattr(np, "Inf")
            try:
                with monkeypatch.context() as context:
                    context.setattr(si, "__version__", self.version)
                    context.setattr(sys, "argv", command[1:])
                    runpy.run_path(command[1], run_name="__main__")
            except Exception as exc:  # noqa: BLE001 - simulate client output
                return repr(exc)  # SI's client returns output, not exit codes.
            finally:
                si.reset_global_job_kwargs()
                si.set_global_job_kwargs(**before_jobs)
                if not before_inf and hasattr(np, "Inf"):
                    del np.Inf
            return "sort completed"

    clients = []

    def client_factory(*args):
        client = Client(*args)
        clients.append(client)
        return client

    def installer(client, name, **kwargs):
        client.installs.append((name, kwargs))
        if name == "spikeinterface" and client.install_succeeds:
            client.version = kwargs["version"]
        elif name == "conflicting-package":
            client.version = "0.103.0"

    def sort(sorter_name, recording, folder, **kwargs):
        if clients[-1].fail_sort:
            raise RuntimeError("sort failed")
        assert hasattr(
            np, "Inf"
        ), "MS4 must also work with NumPy 2 in containers"
        result = si.NumpySorting.from_unit_dict({7: np.array([10, 20])}, 30_000)
        result.set_property("quality", np.array(["good"]))
        return result

    monkeypatch.setattr(container_tools, "ContainerClient", client_factory)
    monkeypatch.setattr(
        container_tools, "install_package_in_container", installer
    )
    monkeypatch.setitem(sorter_dict, "mountainsort4", FakeSorter)
    monkeypatch.setattr(sis, "run_sorter_local", sort)
    return recording, Client, clients


def _execution(backend, **overrides):
    return dict(
        backend=backend,
        container_image="example/ms4:1.0.5",
        installation_mode="pypi",
        spikeinterface_version="0.104.3",
        **overrides,
    )


@pytest.mark.parametrize("backend", ["docker", "singularity"])
@pytest.mark.parametrize("mode", ["pypi", "github"])
def test_container_pin_and_provenance_survive_dispatch_cleanup(
    runtime, backend, mode, tmp_path
):
    recording, _client_class, clients = runtime
    execution = _execution(backend)
    execution["installation_mode"] = mode
    result = run_si_sorter(
        "mountainsort4",
        {"whiten": False},
        recording,
        uuid.uuid4(),
        {},
        execution,
    )
    client = clients[-1]
    assert client.started and client.stopped
    assert client.script_executed
    assert client.version == "0.104.3"
    assert client.installs[0][1]["installation_mode"] == mode
    assert sort_runtime_versions(result, "mountainsort4", execution) == (
        "0.104.3",
        "1.0.5",
    )
    assert (
        result.get_annotation(RUNTIME_ANNOTATION)["sorter_version"] == "1.0.5"
    )
    np.testing.assert_array_equal(result.get_unit_spike_train(7), [10, 20])
    np.testing.assert_array_equal(result.get_property("quality"), ["good"])
    assert not list(tmp_path.glob("sort_*"))


@pytest.mark.parametrize("installed", [None, "0.104.3"])
def test_missing_or_matching_runtime(runtime, installed):
    recording, client_class, clients = runtime
    client_class.version = installed
    result = run_si_sorter(
        "mountainsort4", {}, recording, uuid.uuid4(), {}, _execution("docker")
    )
    assert len(clients[-1].installs) == (1 if installed is None else 0)
    assert result.get_unit_ids().tolist() == [7]


@pytest.mark.parametrize(
    "failure",
    ["install", "no-install", "extra-requirements", "late-change", "sort"],
)
def test_container_failures_stop_and_clean_up(runtime, failure, tmp_path):
    recording, client_class, clients = runtime
    execution = _execution("docker")
    if failure == "install":
        client_class.install_succeeds = False
    elif failure == "no-install":
        execution["installation_mode"] = "no-install"
    elif failure == "extra-requirements":
        execution["extra_requirements"] = ["conflicting-package"]
    elif failure == "late-change":
        client_class.change_version_at_sort = True
    else:
        client_class.fail_sort = True
    with pytest.raises(RuntimeError, match="runtime|receipt"):
        run_si_sorter(
            "mountainsort4", {}, recording, uuid.uuid4(), {}, execution
        )
    assert clients[-1].stopped
    assert not list(tmp_path.glob("sort_*"))
    if failure in ("install", "no-install", "extra-requirements"):
        assert not clients[-1].script_executed


def test_unpinned_baked_image_reports_its_actual_versions(runtime, tmp_path):
    recording, _client_class, clients = runtime
    result = run_sorter_container(
        "mountainsort4",
        recording,
        tmp_path / "output",
        docker_image="example/ms4:1.0.5",
        installation_mode="no-install",
    )
    assert sort_runtime_versions(
        result, "mountainsort4", {"backend": "docker"}
    ) == ("0.103.0", "1.0.5")
    assert not clients[-1].installs


def test_container_provenance_never_falls_back_to_host():
    sorting = si.NumpySorting.from_unit_dict({0: np.array([1])}, 30_000)
    with pytest.raises(RuntimeError, match="provenance"):
        sort_runtime_versions(sorting, "mountainsort4", {"backend": "docker"})
