"""Verified Docker/Singularity sorter execution using SI's container client.

SI 0.104.3's container runner installs SpikeInterface only when it is absent,
even with an explicit version pin. This service owns the in-container script
so version checks and producing-runtime provenance run in the same process as
the sorter. All files live in the dispatcher's per-attempt scratch directory.
No database access occurs here.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNTIME_ANNOTATION = "spyglass_sort_runtime"
_VERSION_MARKER = "SPYGLASS_SI_VERSION="


def _installed_si_version(client):
    """Probe a fresh interpreter; container clients do not expose exit codes."""
    output = client.run_command(
        [
            "python",
            "-c",
            (
                "import spikeinterface; from spikeinterface.sorters import "
                "run_sorter_local; "
                f"print({_VERSION_MARKER!r} + spikeinterface.__version__)"
            ),
        ]
    )
    for line in str(output).splitlines():
        if line.startswith(_VERSION_MARKER):
            return line[len(_VERSION_MARKER) :].strip()
    return None


def _prepare_runtime(client, installation_mode, requested_version):
    """Install when necessary and fail closed if the resulting pin differs."""
    import spikeinterface as si
    from packaging.version import Version
    from spikeinterface.sorters.container_tools import (
        install_package_in_container,
    )

    installed = _installed_si_version(client)
    target = requested_version or si.__version__
    matches = installed is not None and Version(installed) == Version(target)
    if installation_mode != "no-install" and (
        installed is None or (requested_version is not None and not matches)
    ):
        mode = "github" if installation_mode == "auto" else installation_mode
        client.run_command(
            ["python", "-m", "pip", "install", "--user", "--upgrade", "pip"]
        )
        install_package_in_container(
            client,
            "spikeinterface",
            installation_mode=mode,
            extra="[full]",
            version=target,
            **(
                {
                    "github_url": "https://github.com/SpikeInterface/spikeinterface"
                }
                if mode == "github"
                else {}
            ),
        )
        installed = _installed_si_version(client)
    if installed is None or (
        requested_version is not None
        and Version(installed) != Version(requested_version)
    ):
        raise RuntimeError(
            "Container SpikeInterface runtime does not match the execution "
            f"recipe: requested={requested_version!r}, installed={installed!r}, "
            f"installation_mode={installation_mode!r}."
        )


def run_sorter_container(
    sorter_name,
    recording,
    folder,
    *,
    docker_image=None,
    singularity_image=None,
    installation_mode="auto",
    spikeinterface_version=None,
    extra_requirements=None,
    delete_container_files=True,
    remove_existing_folder=True,
    **sorter_params,
):
    """Return SI's portable saved sorting with verified runtime annotations.

    Uses the same recording serialization, bind discovery, GPU policy and
    isolated Singularity user base as SI. A completion receipt is written only
    after sorting and saving succeed; a failed pip command cannot silently run
    the wrong runtime. The pin is checked again after extra requirements.
    """
    import os
    import pickle
    import platform

    import spikeinterface as si
    from spikeinterface.core.core_tools import check_json
    from spikeinterface.sorters.container_tools import (
        ContainerClient,
        find_recording_folders,
        install_package_in_container,
        path_to_unix,
        windows_extractor_dict_to_unix,
    )
    from spikeinterface.sorters.sorterlist import sorter_dict
    from spikeinterface.sorters.utils import has_nvidia

    from spyglass.utils import logger

    folder = Path(folder).resolve()
    scratch = folder.parent
    scratch.mkdir(parents=True, exist_ok=True)
    mode = "docker" if docker_image is not None else "singularity"
    image = docker_image if mode == "docker" else singularity_image
    rec_dict = recording.to_dict(recursive=True)
    input_folders = find_recording_folders(rec_dict)
    if platform.system() == "Windows":
        rec_dict = windows_extractor_dict_to_unix(rec_dict)
    if recording.check_serializability("json"):
        rec_file = scratch / "in_container_recording.json"
        rec_file.write_text(json.dumps(check_json(rec_dict)), encoding="utf8")
    elif recording.check_serializability("pickle"):
        rec_file = scratch / "in_container_recording.pickle"
        rec_file.write_bytes(pickle.dumps(rec_dict))
    else:
        raise RuntimeError(
            "Container execution requires a serializable recording."
        )

    config_file = scratch / "in_container_params.json"
    receipt = scratch / "in_container_runtime.json"
    saved_sorting = folder / "in_container_sorting"
    config_file.write_text(
        json.dumps(
            check_json(
                {
                    "sorter": sorter_name,
                    "params": sorter_params,
                    "job_kwargs": si.get_global_job_kwargs(),
                    "requested_version": spikeinterface_version,
                    "recording": path_to_unix(rec_file),
                    "output": path_to_unix(folder),
                    "sorting": path_to_unix(saved_sorting),
                    "receipt": path_to_unix(receipt),
                    "remove_existing_folder": remove_existing_folder,
                }
            )
        ),
        encoding="utf8",
    )
    script = scratch / "in_container_sorter_script.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import spikeinterface as si\n"
        "from packaging.version import Version\n"
        "from spikeinterface.sorters import run_sorter_local\n"
        "from spikeinterface.sorters.sorterlist import sorter_dict\n"
        "if __name__ == '__main__':\n"
        "    config = json.loads(Path(sys.argv[1]).read_text())\n"
        "    requested = config['requested_version']\n"
        "    if requested and Version(si.__version__) != Version(requested):\n"
        "        raise RuntimeError(f'Container SI pin mismatch: {requested} != {si.__version__}')\n"
        "    if config['sorter'] == 'mountainsort4' and not hasattr(np, 'Inf'):\n"
        "        np.Inf = np.inf\n"
        "    version = sorter_dict[config['sorter']].get_sorter_version()\n"
        "    runtime = {'spikeinterface_version': si.__version__,\n"
        "               'sorter_version': str(version) if version is not None else None}\n"
        "    si.set_global_job_kwargs(**config['job_kwargs'])\n"
        "    recording = si.load(config['recording'])\n"
        "    sorting = run_sorter_local(config['sorter'], recording, folder=config['output'],\n"
        "        remove_existing_folder=config['remove_existing_folder'], **config['params'])\n"
        "    sorting.save(folder=config['sorting'])\n"
        "    Path(config['receipt']).write_text(json.dumps(runtime))\n",
        encoding="utf8",
    )
    volumes = {
        str(path): {"bind": path_to_unix(path), "mode": "ro"}
        for path in input_folders
    }
    volumes[str(scratch)] = {"bind": path_to_unix(scratch), "mode": "rw"}
    user_base = None
    if mode == "singularity":
        user_base = scratch / "in_container_python_base"
        user_base.mkdir()
    sorter_class = sorter_dict[sorter_name]
    gpu_kwargs = {}
    if sorter_class.use_gpu(sorter_params):
        capability = sorter_class.gpu_capability
        if capability == "nvidia-required" and not has_nvidia():
            raise RuntimeError("The container sorter requires an NVIDIA GPU.")
        if capability not in ("nvidia-required", "nvidia-optional"):
            raise NotImplementedError("Container GPU support requires NVIDIA.")
        if has_nvidia():
            gpu_kwargs["container_requires_gpu"] = True
    client = ContainerClient(
        mode,
        image,
        volumes,
        path_to_unix(user_base) if user_base is not None else None,
        gpu_kwargs,
    )
    try:
        client.start()
        _prepare_runtime(client, installation_mode, spikeinterface_version)
        for requirement in list(extra_requirements or []) + list(
            getattr(recording, "extra_requirements", [])
        ):
            install_package_in_container(
                client, requirement, installation_mode="pypi"
            )
        # Extra installs can upgrade/downgrade SI. Check without repairing so
        # incompatible requirements fail instead of oscillating pip installs.
        _prepare_runtime(client, "no-install", spikeinterface_version)
        output = client.run_command(
            ["python", path_to_unix(script), path_to_unix(config_file)]
        )
        if not receipt.is_file():
            raise RuntimeError(
                f"Container sorting failed without a completion receipt:\n{output}"
            )
        runtime = json.loads(receipt.read_text(encoding="utf8"))
        sorting = si.load(saved_sorting)
        sorting.annotate(**{RUNTIME_ANNOTATION: runtime})
        return sorting
    finally:
        # Keep the original failure if chown/stop fails. The dispatcher's
        # TemporaryDirectory owns all input, output and user-base cleanup.
        if platform.system() != "Windows":
            try:
                client.run_command(
                    ["chown", str(os.getuid()), "-R", path_to_unix(scratch)]
                )
            except Exception as exc:  # noqa: BLE001 - preserve sort failure
                logger.warning(
                    f"Container scratch ownership cleanup failed: {exc!r}"
                )
        try:
            client.stop()
        except Exception as exc:  # noqa: BLE001 - preserve sort failure
            logger.warning(f"Container stop failed: {exc!r}")
        if delete_container_files:
            for path in (rec_file, config_file, script, receipt):
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(f"Container input cleanup failed: {exc!r}")
