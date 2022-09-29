from math import ceil, floor
from typing import Callable, Dict, Tuple, cast

import h5py
import kachery_cloud as kcl
import matplotlib.animation as animation
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sortingview.views as vv
import sortingview.views.franklab as vvf
import xarray as xr
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from replay_trajectory_classification.environments import (
    get_bin_ind,
    get_grid,
    get_track_interior,
)
from ripple_detection import get_multiunit_population_firing_rate
from sortingview.SpikeSortingView import (
    MultiTimeseries,
    create_live_position_pdf_plot,
    create_position_plot,
    create_spike_raster_plot,
)
from tqdm.auto import tqdm


def make_single_environment_movie(
    time_slice,
    classifier,
    results,
    position_info,
    marks,
    movie_name="video_name.mp4",
    sampling_frequency=500,
    video_slowdown=8,
    position_name=["head_position_x", "head_position_y"],
    direction_name="head_orientation",
    vmax=0.07,
):
    if marks.ndim > 2:
        multiunit_spikes = (np.any(~np.isnan(marks), axis=1)).astype(float)
    else:
        multiunit_spikes = np.asarray(marks, dtype=float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(multiunit_spikes, sampling_frequency),
        index=position_info.index,
        columns=["firing_rate"],
    )

    # Set up formatting for the movie files
    Writer = animation.writers["ffmpeg"]
    fps = sampling_frequency // video_slowdown
    writer = Writer(fps=fps, bitrate=-1)

    # Set up data
    is_track_interior = classifier.environments[0].is_track_interior_
    posterior = (
        results.acausal_posterior.isel(time=time_slice)
        .sum("state")
        .where(is_track_interior)
    )
    map_position_ind = posterior.argmax(["x_position", "y_position"])
    map_position = np.stack(
        (
            posterior.x_position[map_position_ind["x_position"]],
            posterior.y_position[map_position_ind["y_position"]],
        ),
        axis=1,
    )

    position = np.asarray(position_info.iloc[time_slice][position_name])
    direction = np.asarray(position_info.iloc[time_slice][direction_name])

    window_size = 501

    window_ind = np.arange(window_size) - window_size // 2
    rate = multiunit_firing_rate.iloc[
        slice(time_slice.start + window_ind[0], time_slice.stop + window_ind[-1])
    ]

    with plt.style.context("dark_background"):
        # Set up plots
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(6, 6),
            gridspec_kw={"height_ratios": [5, 1]},
            constrained_layout=False,
        )

        axes[0].tick_params(colors="white", which="both")
        axes[0].spines["bottom"].set_color("white")
        axes[0].spines["left"].set_color("white")

        position_dot = axes[0].scatter(
            [],
            [],
            s=80,
            zorder=102,
            color="magenta",
            label="actual position",
            animated=True,
        )
        (position_line,) = axes[0].plot(
            [], [], color="magenta", linewidth=5, animated=True
        )
        map_dot = axes[0].scatter(
            [],
            [],
            s=80,
            zorder=102,
            color="green",
            label="decoded position",
            animated=True,
        )
        (map_line,) = axes[0].plot([], [], "green", linewidth=3, animated=True)

        mesh = posterior.isel(time=0).plot(
            x="x_position",
            y="y_position",
            vmin=0.0,
            vmax=vmax,
            ax=axes[0],
            animated=True,
            add_colorbar=False,
        )
        axes[0].set_xlabel("")
        axes[0].set_ylabel("")
        axes[0].set_xlim(
            position_info[position_name[0]].min() - 10,
            position_info[position_name[0]].max() + 10,
        )
        axes[0].set_ylim(
            position_info[position_name[1]].min() - 10,
            position_info[position_name[1]].max() + 10,
        )
        axes[0].spines["top"].set_color("black")
        axes[0].spines["right"].set_color("black")
        title = axes[0].set_title(
            f"time = {posterior.isel(time=0).time.values:0.2f}",
        )
        fontprops = fm.FontProperties(size=16)
        scalebar = AnchoredSizeBar(
            axes[0].transData,
            20,
            "20 cm",
            "lower right",
            pad=0.1,
            color="white",
            frameon=False,
            size_vertical=1,
            fontproperties=fontprops,
        )

        axes[0].add_artist(scalebar)
        axes[0].axis("off")

        (multiunit_firing_line,) = axes[1].plot(
            [], [], color="white", linewidth=2, animated=True, clip_on=False
        )
        axes[1].set_ylim((0.0, np.asarray(rate.max())))
        axes[1].set_xlim(
            (window_ind[0] / sampling_frequency, window_ind[-1] / sampling_frequency)
        )
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Multiunit\n[spikes/s]")
        axes[1].set_facecolor("black")
        axes[1].spines["top"].set_color("black")
        axes[1].spines["right"].set_color("black")

        n_frames = posterior.shape[0]
        progress_bar = tqdm()
        progress_bar.reset(total=n_frames)

        def _update_plot(time_ind):
            start_ind = max(0, time_ind - 5)
            time_slice = slice(start_ind, time_ind)

            position_dot.set_offsets(position[time_ind])

            r = 4
            position_line.set_data(
                [
                    position[time_ind, 0],
                    position[time_ind, 0] + r * np.cos(direction[time_ind]),
                ],
                [
                    position[time_ind, 1],
                    position[time_ind, 1] + r * np.sin(direction[time_ind]),
                ],
            )

            map_dot.set_offsets(map_position[time_ind])
            map_line.set_data(map_position[time_slice, 0], map_position[time_slice, 1])

            mesh.set_array(posterior.isel(time=time_ind).values.ravel(order="F"))

            title.set_text(f"time = {posterior.isel(time=time_ind).time.values:0.2f}")

            try:
                multiunit_firing_line.set_data(
                    window_ind / sampling_frequency,
                    np.asarray(rate.iloc[time_ind + (window_size // 2) + window_ind]),
                )
            except IndexError:
                pass

            progress_bar.update()

            return (
                position_dot,
                position_line,
                map_dot,
                map_line,
                mesh,
                title,
                multiunit_firing_line,
            )

        movie = animation.FuncAnimation(
            fig, _update_plot, frames=n_frames, interval=1000 / fps, blit=True
        )
        if movie_name is not None:
            movie.save(movie_name, writer=writer, dpi=200)

        return fig, movie


def setup_subplots(classifier, window_ind=None, rate=None, sampling_frequency=None):
    env_names = [env.environment_name for env in classifier.environments]

    mosaic = []
    for env_ind, env_name in enumerate(env_names):
        if len(mosaic) == 0:
            mosaic.append([])
            mosaic[-1].append(env_name)
        else:
            mosaic[-1].append(env_name)
        print("\n")

    mosaic.append(["multiunit"] * len(env_names))

    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [5, 1]},
        constrained_layout=False,
    )

    for env_name in env_names:
        ax = axes[env_name]
        env_ind = classifier.environments.index(env_name)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(
            classifier.environments[env_ind].edges_[0].min() - 10,
            classifier.environments[env_ind].edges_[0].max() + 10,
        )
        ax.set_ylim(
            classifier.environments[env_ind].edges_[1].min() - 10,
            classifier.environments[env_ind].edges_[1].max() + 10,
        )
        ax.set_title(env_name)
        ax.axis("off")
        sns.despine(ax=axes[env_name])

    ax = axes["multiunit"]
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Multiunit\n[spikes/s]")

    if rate is not None:
        ax.set_ylim((0.0, np.asarray(rate.max())))

    if window_ind is not None and sampling_frequency is not None:
        ax.set_xlim(
            (window_ind[0] / sampling_frequency, window_ind[-1] / sampling_frequency)
        )
    sns.despine(ax=ax)

    return fig, axes


def make_multi_environment_movie(
    time_slice,
    classifier,
    results,
    position_info,
    marks,
    current_environment="",
    movie_name="video_name.mp4",
    sampling_frequency=500,
    video_slowdown=8,
    position_name=["head_position_x", "head_position_y"],
    direction_name="head_orientation",
    vmax=0.07,
):

    # Set up formatting for the movie files
    Writer = animation.writers["ffmpeg"]
    fps = sampling_frequency // video_slowdown
    writer = Writer(fps=fps, bitrate=-1)

    # Set up neural data
    probability = results.isel(time=time_slice).acausal_posterior.sum("position")
    most_prob_env = probability.idxmax("state")
    env_names = [env.environment_name for env in classifier.environments]

    env_posteriors = {}
    for env_name in env_names:
        env = classifier.environments[classifier.environments.index(env_name)]
        n_position_bins = env.place_bin_centers_.shape[0]
        position_index = pd.MultiIndex.from_arrays(
            (env.place_bin_centers_[:, 0], env.place_bin_centers_[:, 1]),
            names=["x_position", "y_position"],
        )
        env_posteriors[env_name] = (
            results.sel(state=env_name, position=slice(0, n_position_bins))
            .isel(time=time_slice)
            .acausal_posterior.assign_coords(position=position_index)
            .unstack("position")
            .where(env.is_track_interior_)
        )

    if marks.ndim > 2:
        multiunit_spikes = (np.any(~np.isnan(marks), axis=1)).astype(float)
    else:
        multiunit_spikes = np.asarray(marks, dtype=float)

    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(multiunit_spikes, sampling_frequency),
        index=position_info.index,
        columns=["firing_rate"],
    )

    window_size = 501

    window_ind = np.arange(window_size) - window_size // 2
    rate = multiunit_firing_rate.iloc[
        slice(time_slice.start + window_ind[0], time_slice.stop + window_ind[-1])
    ]

    # Set up behavioral data
    position = np.asarray(position_info.iloc[time_slice][position_name])
    direction = np.asarray(position_info.iloc[time_slice][direction_name])

    with plt.style.context("dark_background"):
        fig, axes = setup_subplots(
            classifier,
            window_ind=window_ind,
            rate=rate,
            sampling_frequency=sampling_frequency,
        )

        position_dot = axes[current_environment].scatter(
            [],
            [],
            s=80,
            zorder=102,
            color="magenta",
            label="actual position",
            animated=True,
        )
        (position_line,) = axes[current_environment].plot(
            [], [], color="magenta", linewidth=5, animated=True
        )

        meshes = {}
        titles = {}
        map_dots = {}
        for env_name, posterior in env_posteriors.items():
            meshes[env_name] = posterior.isel(time=0).plot(
                x="x_position",
                y="y_position",
                vmin=0.0,
                vmax=vmax,
                ax=axes[env_name],
                animated=True,
                add_colorbar=False,
            )
            prob = float(probability.isel(time=0).sel(state=env_name))
            titles[env_name] = axes[env_name].set_title(
                f"environment = {env_name}\nprob. = {prob:0.2f}"
            )
            map_dots[env_name] = axes[env_name].scatter(
                [], [], s=80, zorder=102, color="green", animated=True
            )

        (multiunit_firing_line,) = axes["multiunit"].plot(
            [], [], color="white", linewidth=2, animated=True, clip_on=False
        )

        n_frames = posterior.shape[0]
        progress_bar = tqdm()
        progress_bar.reset(total=n_frames)

        def _update_plot(time_ind):
            position_dot.set_offsets(position[time_ind])

            r = 4
            position_line.set_data(
                [
                    position[time_ind, 0],
                    position[time_ind, 0] + r * np.cos(direction[time_ind]),
                ],
                [
                    position[time_ind, 1],
                    position[time_ind, 1] + r * np.sin(direction[time_ind]),
                ],
            )

            for env_name, mesh in meshes.items():
                posterior = (
                    env_posteriors[env_name].isel(time=time_ind).values.ravel(order="F")
                )
                mesh.set_array(posterior)
                prob = float(probability.isel(time=time_ind).sel(state=env_name))
                titles[env_name].set_text(
                    f"environment = {env_name}\nprob. = {prob:0.2f}"
                )
                if env_name == most_prob_env.isel(time=time_ind):
                    env = classifier.environments[
                        classifier.environments.index(env_name)
                    ]
                    map_dots[env_name].set_offsets(
                        env.place_bin_centers_[np.nanargmax(posterior)]
                    )
                    map_dots[env_name].set_alpha(1.0)
                else:
                    map_dots[env_name].set_alpha(0.0)

            multiunit_firing_line.set_data(
                window_ind / sampling_frequency,
                np.asarray(rate.iloc[time_ind + (window_size // 2) + window_ind]),
            )

            progress_bar.update()

            return (
                position_dot,
                position_line,
                *meshes.values(),
                *map_dots.values(),
                multiunit_firing_line,
            )

        movie = animation.FuncAnimation(
            fig, _update_plot, frames=n_frames, interval=1000 / fps, blit=True
        )
        if movie_name is not None:
            movie.save(movie_name, writer=writer, dpi=200)

        return fig, movie


def create_live_position_pdf_plot_h5(
    *, data: np.ndarray, segment_size: int, multiscale_factor: int
):
    data_uri = kcl.store_npy(data)
    key = {
        "type": "live_position_plot_h5",
        "version": 5,
        "data_uri": data_uri,
        "segment_size": segment_size,
        "multiscale_factor": multiscale_factor,
    }
    key_str = f"@create_live_position_pdf_plot_h5/{kcl.sha1_of_dict(key)}"
    a = kcl.get_mutable_local(key_str)
    if a and kcl.load_file(a):
        return a

    num_times = data.shape[0]
    num_positions = data.shape[1]

    def fetch_segment(istart: int, iend: int):
        return np.nan_to_num(data[istart:iend])

    with kcl.TemporaryDirectory() as tmpdir:
        print(tmpdir)
        output_file_name = tmpdir + "/live_position_pdf_plot.h5"
        with h5py.File(output_file_name, "w") as f:
            downsample_factor = 1
            while downsample_factor < num_times:
                num_segments = ceil(floor(num_times / downsample_factor) / segment_size)
                for iseg in range(num_segments):
                    i1 = iseg * segment_size
                    i2 = min(i1 + segment_size, floor(num_times / downsample_factor))
                    if downsample_factor == 1:
                        A = fetch_segment(istart=i1, iend=i2)
                        B = A / np.reshape(
                            np.repeat(np.max(A, axis=1), A.shape[1]), A.shape
                        )
                        B = (B * 100).astype(np.uint8)
                    else:
                        prev_downsample_factor = floor(
                            downsample_factor / multiscale_factor
                        )
                        B_prev_list = [
                            np.array(
                                f.get(
                                    f"segment/{prev_downsample_factor}/{iseg * multiscale_factor + offset}"
                                )
                            )
                            for offset in range(multiscale_factor)
                            if (iseg * multiscale_factor + offset)
                            * segment_size
                            * prev_downsample_factor
                            < num_times
                        ]
                        B_prev = np.concatenate(B_prev_list, axis=0).astype(np.float32)
                        N_prev = B_prev.shape[0]
                        if N_prev % multiscale_factor != 0:
                            N_prev = (
                                floor(N_prev / multiscale_factor) * multiscale_factor
                            )
                            B_prev = B_prev[:N_prev]
                        B: np.ndarray = np.mean(
                            np.reshape(
                                B_prev,
                                (
                                    floor(N_prev / multiscale_factor),
                                    multiscale_factor,
                                    num_positions,
                                ),
                            ),
                            axis=1,
                        )
                        B = B / np.reshape(
                            np.repeat(np.max(B, axis=1), B.shape[1]), B.shape
                        )
                        B = np.floor(B).astype(np.uint8)
                    print("Creating", f"segment/{downsample_factor}/{iseg}")
                    f.create_dataset(f"segment/{downsample_factor}/{iseg}", data=B)
                downsample_factor *= multiscale_factor

        h5_uri = kcl.store_file(output_file_name)
        kcl.set_mutable_local(key_str, h5_uri)

        return h5_uri


def create_interactive_1D_decoding_figurl(
    position_info,
    linear_position_info,
    classifier,
    results,
    marks=None,
    spikes=None,
    visualization_name="test",
    segment_size=100000,
    multiscale_factor=3,
    sampling_frequency=500,
):

    layout = MultiTimeseries(label=visualization_name)

    # spikes panel
    if spikes is not None:
        cell_ids, spike_ind = np.nonzero(spikes)
        time = np.asarray(results.time)
        spike_times = np.asarray(time[spike_ind] - time[0], dtype=np.float32)
        layout.add_panel(
            create_spike_raster_plot(
                times=np.asarray(spike_times - results.time[0], dtype=np.float32),
                labels=cell_ids.astype(np.int32),
                label="Spikes",
            )
        )
    if marks is not None:
        multiunit_spikes = (np.any(~np.isnan(marks.values), axis=1)).astype(float)
        multiunit_firing_rate = get_multiunit_population_firing_rate(
            multiunit_spikes, sampling_frequency
        )

        layout.add_panel(
            create_position_plot(
                timestamps=np.asarray(marks.time.values - marks.time.values[0]),
                positions=np.asarray(multiunit_firing_rate, dtype=np.float32),
                dimension_labels=["firing rate"],
                label="Firing Rate",
                discontinuous=False,
            ),
            relative_height=1,
        )

    # speed panel
    layout.add_panel(
        create_position_plot(
            timestamps=np.asarray(
                position_info.index - position_info.index[0], dtype=np.float32
            ),
            positions=np.asarray(position_info.head_speed, dtype=np.float32),
            dimension_labels=["Speed"],
            label="Speed",
            discontinuous=False,
        ),
        relative_height=1,
    )
    try:
        posterior = np.asarray(
            results.acausal_posterior.sum("state").where(
                classifier.environments[0].is_track_interior_
            ),
            dtype=np.float32,
        )
    except AttributeError:
        posterior = np.asarray(
            results.acausal_posterior.sum("state").where(classifier.is_track_interior_),
            dtype=np.float32,
        )
    time = np.asarray(results.time - results.time[0], dtype=np.float32)

    h5_uri = create_live_position_pdf_plot_h5(
        data=posterior, segment_size=segment_size, multiscale_factor=multiscale_factor
    )

    try:
        edges = classifier.environments[0].edges_
        is_track_interior = classifier.environments[0].is_track_interior_
    except AttributeError:
        edges = classifier.edges_
        is_track_interior = classifier.is_track_interior_

    binned_linear_position = (
        get_bin_ind(linear_position_info.linear_position, edges)[0] - 1
    )
    not_track = ~np.isin(binned_linear_position, np.nonzero(is_track_interior))
    binned_linear_position[not_track] -= 1

    panel = create_live_position_pdf_plot(
        linear_positions=binned_linear_position.astype(np.int32),
        start_time_sec=time[0],
        end_time_sec=time[-1],
        sampling_frequency=(len(time) - 1) / (time[-1] - time[0]),
        num_positions=posterior.shape[1],
        pdf_object={"format": "position_pdf_h5_v1", "uri": h5_uri},
        segment_size=segment_size,
        multiscale_factor=multiscale_factor,
        label="Position probability",
    )
    layout.add_panel(panel, relative_height=3)

    return layout.get_composite_figure().url()


def get_base_track_information(base_probabilities: xr.Dataset):
    x_count = len(base_probabilities.x_position)
    y_count = len(base_probabilities.y_position)
    x_min = np.min(base_probabilities.x_position).item()
    y_min = np.min(base_probabilities.y_position).item()
    x_width = round(
        (np.max(base_probabilities.x_position).item() - x_min) / (x_count - 1), 6
    )
    y_width = round(
        (np.max(base_probabilities.y_position).item() - y_min) / (y_count - 1), 6
    )
    return (x_count, x_min, x_width, y_count, y_min, y_width)


def generate_linearization_function(
    location_lookup: Dict[Tuple[float, float], int],
    x_count: int,
    x_min: float,
    x_width: float,
    y_min: float,
    y_width: float,
):

    args = {
        "location_lookup": location_lookup,
        "x_count": x_count,
        "x_min": x_min,
        "x_width": x_width,
        "y_min": y_min,
        "y_width": y_width,
    }

    def inner(t: Tuple[float, float, float]):
        return memo_linearize(t, **args)

    return inner


def memo_linearize(
    t: Tuple[float, float, float],
    /,
    location_lookup: Dict[Tuple[float, float], int],
    x_count: int,
    x_min: float,
    x_width: float,
    y_min: float,
    y_width: float,
):
    (_, y, x) = t
    my_tuple = (x, y)
    if my_tuple not in location_lookup:
        lin = x_count * round((y - y_min) / y_width) + round((x - x_min) / x_width)
        location_lookup[my_tuple] = lin
    return location_lookup[my_tuple]


def extract_slice_data(
    base_slice: xr.Dataset, location_fn: Callable[[Tuple[float, float]], int]
):
    i_trim = discretize_and_trim(base_slice)
    observations = i_trim.acausal_posterior.values
    positions = get_positions(i_trim, location_fn)
    observations_per_frame = get_observations_per_frame(i_trim, base_slice)
    return (observations, positions, observations_per_frame)


def discretize_and_trim(base_slice: xr.Dataset):
    i = np.multiply(base_slice, 255).astype("int8")
    i_stack = i.stack(unified_index=["time", "y_position", "x_position"])
    i_trim = i_stack.where(i_stack.acausal_posterior > 0, drop=True).astype("int8")
    return i_trim


def get_positions(
    i_trim: xr.Dataset, linearization_fn: Callable[[Tuple[float, float]], int]
):
    linearizer_map = map(linearization_fn, i_trim.unified_index.data)
    positions = np.array(list(linearizer_map), dtype="int16")
    return positions


def get_observations_per_frame(i_trim: xr.Dataset, base_slice: xr.Dataset):
    (times, time_counts_np) = np.unique(i_trim.time.data, return_counts=True)
    time_counts = xr.DataArray(time_counts_np, coords={"time": times})
    raw_times = base_slice.time
    (_, good_counts) = xr.align(raw_times, time_counts, join="left", fill_value=0)
    observations_per_frame = good_counts.data.astype("int8")
    return observations_per_frame


def process_decoded_data(results):
    frame_step_size = 100_000
    location_lookup = {}
    base_probabilities = cast(xr.Dataset, results)

    (x_count, x_min, x_width, y_count, y_min, y_width) = get_base_track_information(
        base_probabilities
    )
    location_fn = generate_linearization_function(
        location_lookup, x_count, x_min, x_width, y_min, y_width
    )

    total_frame_count = len(base_probabilities.time)
    final_frame_bounds = np.zeros(total_frame_count, dtype="int8")
    # intentionally oversized preallocation--will trim later
    # Note: By definition there can't be more than 255 observations per frame (since we drop any observation
    # lower than 1/255 and the probabilities for any frame sum to 1). However, this preallocation may be way
    # too big for memory for long recordings. We could use a smaller one, but would need to include logic
    # to expand the length of the array if its actual allocated bounds are exceeded.
    final_values = np.zeros(total_frame_count * 255, dtype="int8")
    final_locations = np.zeros(total_frame_count * 255, dtype="int16")

    frames_done = 0
    total_observations = 0
    while frames_done <= total_frame_count:
        base_slice = base_probabilities.isel(
            time=slice(frames_done, frames_done + frame_step_size)
        )
        (observations, positions, observations_per_frame) = extract_slice_data(
            base_slice, location_fn
        )
        final_frame_bounds[
            frames_done : frames_done + len(observations_per_frame)
        ] = observations_per_frame
        final_values[
            total_observations : total_observations + len(observations)
        ] = observations
        final_locations[
            total_observations : total_observations + len(observations)
        ] = positions
        total_observations += len(observations)
        frames_done += frame_step_size
    # These were intentionally oversized in preallocation; trim to the number of actual values.
    final_values.resize(total_observations)
    final_locations.resize(total_observations)

    return {
        "type": "DecodedPositionData",
        "xmin": x_min,
        "binWidth": x_width,
        "xcount": x_count,
        "ymin": y_min,
        "binHeight": y_width,
        "ycount": y_count,
        "uniqueLocations": np.unique(final_locations),
        "values": final_values,
        "locations": final_locations,
        "frameBounds": final_frame_bounds,
    }


def make_track(positions, bin_size: float = 1.0):
    (edges, _, place_bin_centers, _) = get_grid(positions, bin_size)
    is_track_interior = get_track_interior(positions, edges)

    # bin dimensions are the difference between bin centers in the x and y directions.
    bin_width = np.max(np.diff(place_bin_centers, axis=0)[:, 0])
    bin_height = np.max(np.diff(place_bin_centers, axis=0)[:, 1])

    # so we can represent the track as a collection of rectangles of width bin_width and height bin_height,
    # centered on the values of place_bin_centers where track_interior = true.
    # Note, the original code uses Fortran ordering.
    true_ctrs = place_bin_centers[is_track_interior.ravel(order="F")]

    return (bin_width, bin_height, get_ul_corners(bin_width, bin_height, true_ctrs))


def get_ul_corners(width: float, height: float, centers):
    ul = np.subtract(centers, (width / 2, -height / 2))

    # Reshape so we have an x array and a y array
    return np.transpose(ul)


def create_static_track_animation(
    *,
    track_rect_width: float,
    track_rect_height: float,
    ul_corners,
    timestamps,
    positions,
    compute_real_time_rate: bool = False,
    head_dir=None,
):
    # float32 gives about 7 digits of decimal precision; we want 3 digits right of the decimal.
    # So need to compress-store the timestamp if the start is greater than say 5000.
    first_timestamp = 0
    if timestamps[0] > 5000:
        first_timestamp = timestamps[0]
        timestamps -= first_timestamp
    data = {
        "type": "TrackAnimation",
        "trackBinWidth": track_rect_width,
        "trackBinHeight": track_rect_height,
        "trackBinULCorners": ul_corners.astype("float32"),
        "totalRecordingFrameLength": len(timestamps),
        "timestamps": timestamps.astype("float32"),
        "positions": positions.astype("float32"),
        "xmin": np.min(ul_corners[0]),
        "xmax": np.max(ul_corners[0]) + track_rect_width,
        "ymin": np.min(ul_corners[1]),
        "ymax": np.max(ul_corners[1]) + track_rect_height
        # Speed: should this be displayed?
        # TODO: Better approach for accommodating further data streams
    }
    if head_dir is not None:
        # print(f'Loading head direction: {head_dir}')
        data["headDirection"] = head_dir.astype("float32")
    if compute_real_time_rate:
        median_delta_t = np.median(np.diff(timestamps))
        sampling_frequency_Hz = 1 / median_delta_t
        data["samplingFrequencyHz"] = sampling_frequency_Hz
    if first_timestamp > 0:
        data["timestampStart"] = first_timestamp

    return data


def create_track_animation_object(*, static_track_animation: any):
    if "decodedData" in static_track_animation:
        decoded_data = static_track_animation["decodedData"]
        decoded_data_obj = vvf.DecodedPositionData(
            x_min=decoded_data["xmin"],
            x_count=decoded_data["xcount"],
            y_min=decoded_data["ymin"],
            y_count=decoded_data["ycount"],
            bin_width=decoded_data["binWidth"],
            bin_height=decoded_data["binHeight"],
            values=decoded_data["values"].astype("int16"),
            locations=decoded_data["locations"],
            frame_bounds=decoded_data["frameBounds"].astype("int16"),
        )
    else:
        decoded_data_obj = None

    view = vvf.TrackPositionAnimationV1(
        track_bin_width=static_track_animation["trackBinWidth"],
        track_bin_height=static_track_animation["trackBinHeight"],
        track_bin_ul_corners=static_track_animation["trackBinULCorners"],
        total_recording_frame_length=static_track_animation[
            "totalRecordingFrameLength"
        ],
        timestamp_start=static_track_animation["timestampStart"]
        if "timestampStart" in static_track_animation
        else None,
        timestamps=static_track_animation["timestamps"],
        positions=static_track_animation["positions"],
        x_min=static_track_animation["xmin"],
        x_max=static_track_animation["xmax"],
        y_min=static_track_animation["ymin"],
        y_max=static_track_animation["ymax"],
        sampling_frequency_hz=static_track_animation["samplingFrequencyHz"],
        head_direction=static_track_animation["headDirection"]
        if "headDirection" in static_track_animation
        else None,
        decoded_data=decoded_data_obj,
    )
    return view


def create_interactive_2D_decoding_figurl(
    position_info,
    marks,
    results,
    bin_size,
    position_names=["head_position_x", "head_position_y"],
    head_direction_name="head_orientation",
    sampling_frequency=500,
    view_height=800,
):

    positions = np.asarray(position_info[position_names])
    (track_width, track_height, upper_left_points) = make_track(
        positions, bin_size=bin_size
    )
    timestamps = np.squeeze(np.asarray(position_info.index)).copy()

    head_dir = np.squeeze(np.asarray(position_info[head_direction_name]))

    data = create_static_track_animation(
        ul_corners=upper_left_points,
        track_rect_height=track_height,
        track_rect_width=track_width,
        timestamps=timestamps,
        positions=positions.T,
        head_dir=head_dir,
        compute_real_time_rate=True,
    )
    data["decodedData"] = process_decoded_data(results.sum("state"))

    decode_view = create_track_animation_object(static_track_animation=data)

    probability_view = vv.TimeseriesGraph()
    COLOR_CYCLE = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    for state, color in zip(results.state.values, COLOR_CYCLE):
        probability_view.add_line_series(
            name=state,
            t=np.asarray(results.time),
            y=np.asarray(
                results.acausal_posterior.sel(state=state).sum(
                    ["x_position", "y_position"]
                ),
                dtype=np.float32,
            ),
            color=color,
            width=1,
        )

    speed_view = vv.TimeseriesGraph().add_line_series(
        name="Speed [cm/s]",
        t=np.asarray(position_info.index),
        y=np.asarray(position_info.head_speed, dtype=np.float32),
        color="black",
        width=1,
    )

    multiunit_spikes = (np.any(~np.isnan(marks.values), axis=1)).astype(float)
    multiunit_firing_rate = get_multiunit_population_firing_rate(
        multiunit_spikes, sampling_frequency
    )

    multiunit_firing_rate_view = vv.TimeseriesGraph().add_line_series(
        name="Multiunit Rate [spikes/s]",
        t=np.asarray(marks.time.values),
        y=np.asarray(multiunit_firing_rate, dtype=np.float32),
        color="black",
        width=1,
    )

    view = vv.Box(
        direction="horizontal",
        show_titles=True,
        height=view_height,
        items=[
            vv.LayoutItem(
                vv.Box(
                    direction="horizontal",
                    show_titles=True,
                    items=[
                        vv.LayoutItem(decode_view, stretch=1, title="Decode"),
                    ],
                )
            ),
            vv.LayoutItem(
                vv.Box(
                    direction="vertical",
                    show_titles=True,
                    items=[
                        vv.LayoutItem(
                            probability_view, stretch=1, title="Probability of State"
                        ),
                        vv.LayoutItem(speed_view, stretch=1, title="Speed"),
                        vv.LayoutItem(
                            multiunit_firing_rate_view, stretch=1, title="Multiunit"
                        ),
                    ],
                )
            ),
        ],
    )

    return view
