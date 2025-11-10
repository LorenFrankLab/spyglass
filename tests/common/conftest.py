from pathlib import Path

import ndx_optogenetics as ndxo
import numpy as np
import pynwb
import pytest
from hdmf.common.table import DynamicTable, VectorData
from ndx_franklab_novela import CameraDevice, FrankLabOptogeneticEpochsTable
from pynwb import NWBHDF5IO, TimeSeries
from pynwb.behavior import BehavioralEvents
from pynwb.testing.mock.behavior import mock_TimeSeries
from pynwb.testing.mock.file import mock_NWBFile, mock_Subject


@pytest.fixture(scope="session")
def interval_list(common):
    yield common.IntervalList()


@pytest.fixture(scope="session")
def mini_devices(mini_content):
    yield mini_content.devices


@pytest.fixture(scope="session")
def mini_behavior(mini_content):
    yield mini_content.processing.get("behavior")


@pytest.fixture(scope="session")
def mini_pos(mini_behavior):
    yield mini_behavior.get_data_interface("position").spatial_series


@pytest.fixture(scope="session")
def mini_pos_series(mini_pos):
    yield next(iter(mini_pos))


@pytest.fixture(scope="session")
def mini_beh_events(mini_behavior):
    yield mini_behavior.get_data_interface("behavioral_events")


@pytest.fixture(scope="session")
def mini_pos_interval_dict(mini_insert, common):
    yield {"interval_list_name": common.PositionSource.get_pos_interval_name(0)}


@pytest.fixture(scope="session")
def mini_pos_tbl(common, mini_pos_series):
    yield common.PositionSource.SpatialSeries * common.RawPosition.PosObject & {
        "name": mini_pos_series
    }


@pytest.fixture(scope="session")
def pos_src(common):
    yield common.PositionSource()


@pytest.fixture(scope="session")
def pos_interval_01(pos_src):
    yield [pos_src.get_pos_interval_name(x) for x in range(1)]


@pytest.fixture(scope="session")
def common_ephys(common):
    yield common.common_ephys


@pytest.fixture(scope="session")
def pop_common_electrode_group(common_ephys):
    common_ephys.ElectrodeGroup.populate()
    yield common_ephys.ElectrodeGroup()


@pytest.fixture(scope="session")
def dio_only_nwb(raw_dir, common):
    nwbfile = mock_NWBFile(
        identifier="my_identifier",
        session_description="my_session_description",
        lab="My Lab",
        institution="My Institution",
        experimenter=["Dr. A", "Dr. B"],
    )
    nwbfile.subject = mock_Subject()
    time_series = mock_TimeSeries(
        name="my_time_series", timestamps=np.arange(20), data=np.ones((20, 1))
    )
    behavioral_events = BehavioralEvents(
        name="behavioral_events", time_series=time_series
    )
    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="behavior module"
    )
    behavior_module.add(behavioral_events)

    from spyglass.settings import raw_dir

    file_name = "mock_behavior.nwb"
    nwbfile_path = Path(raw_dir) / file_name
    with NWBHDF5IO(nwbfile_path, "w") as io:
        io.write(nwbfile)

    yield file_name

    # Cleanup
    (common.Nwbfile & {"nwb_file_name": "mock_behavior_.nwb"}).delete(
        safemode=False
    )


@pytest.fixture(scope="function")
def virus_dict():
    return dict(
        name="test_virus_1",
        construct_name="construct_1",
        manufacturer="Test Manufacturer",
        titer_in_vg_per_ml=1e12,
        description="Test virus",
    )


@pytest.fixture(scope="function")
def virus_injection_dict(virus_dict):
    return dict(
        name="injection_1",
        description="Test virus injection",
        hemisphere="left",
        location="CA1",
        ap_in_mm=0.5,
        ml_in_mm=1.0,
        dv_in_mm=1.5,
        roll_in_deg=0.0,
        pitch_in_deg=0.0,
        yaw_in_deg=0.0,
        reference="Bregma",
        volume_in_uL=1.0,
    )


@pytest.fixture(scope="function")
def excitation_source_model_dict():
    return dict(
        name="test_source_model",
        description="Test description",
        manufacturer="Test manufacturer",
        illumination_type="Test illumination type",
        wavelength_range_in_nm=(400, 700),
    )


@pytest.fixture(scope="function")
def excitation_source_dict():
    return dict(
        name="test_source",
        wavelength_in_nm=450.0,
        power_in_W=1.0,
        intensity_in_W_per_m2=100.0,
    )


@pytest.fixture(scope="function")
def fiber_model_dict():
    return dict(
        name="test_fiber_model",
        description="Test fiber model",
        fiber_name="Test Fiber",
        fiber_model="Test Model",
        manufacturer="Test Manufacturer",
        numerical_aperture=0.5,
        core_diameter_in_um=200.0,
        active_length_in_mm=1.0,
        ferrule_name="Test Ferrule",
        ferrule_diameter_in_mm=2.0,
    )


@pytest.fixture(scope="function")
def fiber_implant_dict():
    return dict(
        implanted_fiber_description="Test fiber implant",
        location="CA1",
        hemisphere="left",
        ap_in_mm=0.5,
        ml_in_mm=1.0,
        dv_in_mm=1.5,
        roll_in_deg=0.0,
        pitch_in_deg=0.0,
        yaw_in_deg=0.0,
    )


@pytest.fixture(scope="function")
def opto_epoch_dict():
    return dict(
        start_time=0.0,
        stop_time=10000.0,
        stimulation_on=True,
        pulse_length_in_ms=10.0,
        period_in_ms=100.0,
        number_pulses_per_pulse_train=10,
        number_trains=1,
        intertrain_interval_in_ms=100,
        power_in_mW=77,
        epoch_name="epoch_01",
        epoch_number=1,
        convenience_code="test",
        epoch_type="run",
        theta_filter_on=True,
        theta_filter_lockout_period_in_samples=100,
        theta_filter_phase_in_deg=90.0,
        theta_filter_reference_ntrode=1,
        spatial_filter_on=True,
        spatial_filter_lockout_period_in_samples=100,
        spatial_filter_region_node_coordinates_in_pixels=np.random.uniform(
            0, 100, (10, 4, 2)
        ),
        ripple_filter_on=True,
        ripple_filter_lockout_period_in_samples=100,
        ripple_filter_threshold_sd=3.0,
        ripple_filter_num_above_threshold=5,
        speed_filter_on=True,
        speed_filter_threshold_in_cm_per_s=5.0,
        speed_filter_on_above_threshold=True,
    )


@pytest.fixture(scope="module")
def custom_prefix():
    """Custom database prefix for testing custom AnalysisNwbfile tables."""
    yield "testcustom"


@pytest.fixture(scope="module")
def custom_config(dj_conn, custom_prefix):
    """Set up custom config with database prefix."""
    import datajoint as dj

    original_prefix = dj.config.get("custom", {}).get("database.prefix")

    if "custom" not in dj.config:
        dj.config["custom"] = {}
    dj.config["custom"]["database.prefix"] = custom_prefix

    yield custom_prefix

    # Restore original config
    if original_prefix:
        dj.config["custom"]["database.prefix"] = original_prefix
    elif "database.prefix" in dj.config.get("custom", {}):
        del dj.config["custom"]["database.prefix"]


@pytest.fixture(scope="module")
def common_nwbfile(common):
    """Return common nwbfile module."""
    return common.common_nwbfile


@pytest.fixture(scope="module")
def analysis_registry(common_nwbfile):
    """Return AnalysisRegistry table."""
    return common_nwbfile.AnalysisRegistry()


@pytest.fixture(scope="module")
def master_analysis_table(common_nwbfile):
    """Return master AnalysisNwbfile table for comparison."""
    return common_nwbfile.AnalysisNwbfile()


@pytest.fixture(scope="module")
def custom_analysis_table(custom_config, dj_conn, common_nwbfile):
    """Create and return a custom AnalysisNwbfile table.

    This fixture dynamically creates a table following the factory pattern.
    """
    import datajoint as dj

    from spyglass.utils.dj_mixin import SpyglassAnalysis

    prefix = custom_config
    schema = dj.schema(f"{prefix}_nwbfile")

    # Make Nwbfile available in the schema context for foreign key resolution
    Nwbfile = common_nwbfile.Nwbfile  # noqa F401
    _ = common_nwbfile.AnalysisRegistry().unblock_new_inserts()

    @schema
    class AnalysisNwbfile(SpyglassAnalysis, dj.Manual):
        definition = """This definition is managed by SpyglassAnalysis"""

    yield AnalysisNwbfile()


@pytest.fixture
def mock_create(monkeypatch):
    """Fixture to mock create() method for faster testing.

    Replaces the full NWB file copy with a simple text file write.
    This speeds up tests by ~10x without affecting test logic.

    Usage:
        def test_something(custom_analysis_table, mock_create):
            mock_create(custom_analysis_table)
            # Now create() will use the fast mock
            file = custom_analysis_table.create(nwb_file_name)
    """

    def _mock_create(table):
        """Apply mock to a given AnalysisNwbfile table."""

        def mock_create_impl(nwb_file_name, **kwargs):
            # Get the new file name using private method
            new_file_name = table._AnalysisMixin__get_new_file_name(
                nwb_file_name
            )
            # Just write a simple file instead of copying full NWB
            file_path = Path(table.get_abs_path(new_file_name))
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("test")
            return new_file_name

        monkeypatch.setattr(table, "create", mock_create_impl)
        return table.create  # Return original in case test needs to restore

    return _mock_create


@pytest.fixture(scope="function")
def opto_only_nwb(
    raw_dir,
    common,
    opto_epoch_dict,
    virus_dict,
    virus_injection_dict,
    excitation_source_model_dict,
    excitation_source_dict,
    fiber_model_dict,
    fiber_implant_dict,
    data_import,
):
    dummy_name = "mock_optogenetics.nwb"
    # Create a mock NWBFile with optogenetic objects
    nwb = mock_NWBFile()
    nwb.subject = mock_Subject()

    nwb.add_processing_module(
        pynwb.ProcessingModule(
            name="tasks", description="Contains all tasks information"
        )
    )
    # stimulus dio
    nwb.create_processing_module(
        name="behavior", description="Contains all behavior-related data"
    )
    beh_events = BehavioralEvents(name="behavioral_events")
    stimulus = TimeSeries(
        name="stimulus_channel",
        comments="",
        description="stimulus channel data",
        data=[0, 1, 0, 1, 0],  # Example data
        unit="N/A",
        timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],  # Example timestamps
    )
    beh_events.add_timeseries(stimulus)
    nwb.processing["behavior"].add(beh_events)

    # camera
    camera_dict = dict(
        name="camera 1",
        meters_per_pixel=0.0001,
        manufacturer="Test Camera Manufacturer",
        model="Test Camera Model",
        lens="Test Lens",
        camera_name="camera 1",
    )
    camera = CameraDevice(**camera_dict)
    nwb.add_device(camera)

    # task info
    epoch = opto_epoch_dict["epoch_name"]
    nwb.add_epoch(0.0, 10000.0, epoch)
    task_name = VectorData(
        name="task_name",
        description="the name of the task",
        data=[epoch],
    )
    task_description = VectorData(
        name="task_description",
        description="a description of the task",
        data=[
            "Test task description"
        ],  # This should be replaced with the actual task description
    )
    camera_id = VectorData(
        name="camera_id",
        description="the ID number of the camera used for video",
        data=[[1]],
    )
    task_epochs = VectorData(
        name="task_epochs",
        description="the temporal epochs where the animal was exposed to this task",
        data=[[1]],
    )
    task_environment = VectorData(
        name="task_environment",
        description="the environment in which the animal performed the task",
        data=["home"],
    )
    task = DynamicTable(
        name="task_1",  # NOTE: Do we want this name to match the descriptive name entered?
        description="",
        columns=[
            task_name,
            task_description,
            camera_id,
            task_epochs,
            task_environment,
        ],
    )
    nwb.processing["tasks"].add(task)

    # add the optogenetic objects
    virus = ndxo.OptogeneticVirus(**virus_dict)

    virus_injection = ndxo.OptogeneticVirusInjection(
        **virus_injection_dict, virus=virus
    )

    optogenetic_viruses = ndxo.OptogeneticViruses(optogenetic_virus=[virus])
    optogenetic_virus_injections = ndxo.OptogeneticVirusInjections(
        optogenetic_virus_injections=[virus_injection]
    )

    excitation_source_model = ndxo.ExcitationSourceModel(
        **excitation_source_model_dict
    )

    excitation_source = ndxo.ExcitationSource(
        **excitation_source_dict, model=excitation_source_model
    )

    # make the fiber objects
    optical_fiber_model = ndxo.OpticalFiberModel(**fiber_model_dict)

    # make the fiber object
    optical_fiber = ndxo.OpticalFiber(
        name="test_fiber",
        model=optical_fiber_model,
    )

    optical_fiber_locations_table = ndxo.OpticalFiberLocationsTable(
        description="Information about implanted optical fiber locations",
        reference="bregma",
    )
    optical_fiber_locations_table.add_row(
        **fiber_implant_dict,
        excitation_source=excitation_source,
        optical_fiber=optical_fiber,
    )

    nwb.add_device(excitation_source_model)
    nwb.add_device(excitation_source)
    nwb.add_device(optical_fiber_model)
    nwb.add_device(optical_fiber)
    optogenetic_experiment_metadata = ndxo.OptogeneticExperimentMetadata(
        optical_fiber_locations_table=optical_fiber_locations_table,
        optogenetic_viruses=optogenetic_viruses,
        optogenetic_virus_injections=optogenetic_virus_injections,
        stimulation_software="fsgui",
    )
    nwb.add_lab_meta_data(optogenetic_experiment_metadata)

    # opto epochs
    opto_epochs_table = FrankLabOptogeneticEpochsTable(
        name="optogenetic_epochs",
        description="Metadata about optogenetic stimulation parameters per epoch",
    )
    opto_epochs_table.add_row(
        **opto_epoch_dict,
        spatial_filter_cameras=[camera],
        spatial_filter_cameras_cm_per_pixel=[
            camera_dict["meters_per_pixel"] * 100
        ],
        stimulus_signal=stimulus,
    )
    nwb.add_time_intervals(opto_epochs_table)

    path = Path(raw_dir) / dummy_name
    with pynwb.NWBHDF5IO(path, "w") as io:
        io.write(nwb)

    data_import.insert_sessions(dummy_name, raise_err=True)
    yield dummy_name

    # Cleanup
    (common.Nwbfile & {"nwb_file_name": "mock_optogenetics_.nwb"}).delete(
        safemode=False
    )
