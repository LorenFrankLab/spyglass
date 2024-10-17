# Convenience functions
# some DLC-utils copied from datajoint element-interface utils.py
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cached_property
from os import system as os_system
from pathlib import Path
from random import choices as random_choices
from string import ascii_letters

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from spyglass.settings import temp_dir
from spyglass.utils import logger
from spyglass.utils.position import convert_to_pixels as _to_px

RGB_PINK = (234, 82, 111)
RGB_YELLOW = (253, 231, 76)
RGB_BLUE = (30, 144, 255)
RGB_ORANGE = (255, 127, 80)
RGB_WHITE = (255, 255, 255)
COLOR_SWATCH = [
    "#29ff3e",
    "#ff0073",
    "#ff291a",
    "#1e2cff",
    "#b045f3",
    "#ffe91a",
]


class VideoMaker:
    def __init__(
        self,
        video_filename,
        position_mean,
        orientation_mean,
        centroids,
        position_time,
        video_frame_inds=None,
        likelihoods=None,
        processor="matplotlib",
        frames=None,
        percent_frames=1,
        output_video_filename="output.mp4",
        cm_to_pixels=1.0,
        disable_progressbar=False,
        crop=None,
        *args,
        **kwargs,
    ):
        if processor != "matplotlib":
            raise ValueError(
                "open-cv processors are no longer supported. \n"
                + "Use matplotlib or submit a feature request via GitHub."
            )

        # self.output_temp_dir = Path(temp_dir) / "vid_frames"
        self.output_temp_dir = Path(".")
        # while self.output_temp_dir.exists():  # multi-user
        # suffix = "".join(random_choices(ascii_letters, k=2))
        # self.output_temp_dir = Path(temp_dir) / f"vid_frames_{suffix}"
        self.output_temp_dir.mkdir(parents=True, exist_ok=True)

        if not Path(video_filename).exists():
            raise FileNotFoundError(f"Video not found: {video_filename}")

        self.video_filename = video_filename
        self.video_frame_inds = video_frame_inds
        self.position_mean = position_mean["DLC"]
        self.orientation_mean = orientation_mean["DLC"]
        self.centroids = centroids
        self.likelihoods = likelihoods
        self.position_time = position_time
        self.percent_frames = percent_frames
        self.frames = frames
        self.output_video_filename = output_video_filename
        self.cm_to_pixels = cm_to_pixels
        self.crop = crop

        _ = self._set_frame_info()
        _ = self._set_plot_bases()

        logger.info(f"Making video: {self.output_video_filename}")
        self.generate_frames_multithreaded()
        self.stitch_video_from_frames()
        logger.info(f"Finished video: {self.output_video_filename}")
        plt.close(self.fig)

        # shutil.rmtree(self.output_temp_dir)

    def _set_frame_info(self):
        """Set the frame information for the video."""
        if self.frames is None:
            self.n_frames = int(
                len(self.video_frame_inds) * self.percent_frames
            )
            self.frames = np.arange(0, self.n_frames)
        else:
            self.n_frames = len(self.frames)
        self.pad_len = len(str(self.n_frames))

        self.window_ind = np.arange(501) - 501 // 2

        video = cv2.VideoCapture(str(self.video_filename))
        self.frame_size = (
            (int(video.get(3)), int(video.get(4)))
            if not self.crop
            else (
                self.crop[1] - self.crop[0],
                self.crop[3] - self.crop[2],
            )
        )
        self.ratio = (
            (self.crop[3] - self.crop[2]) / (self.crop[1] - self.crop[0])
            if self.crop
            else self.frame_size[1] / self.frame_size[0]
        )
        self.frame_rate = video.get(5)
        self.fps = int(np.round(self.frame_rate))
        video.release()

    def _set_plot_bases(self):
        """Create the figure and axes for the video."""
        plt.style.use("dark_background")
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(8, 6),
            gridspec_kw={"height_ratios": [8, 1]},
            constrained_layout=False,
        )

        axes[0].tick_params(colors="white", which="both")
        axes[0].spines["bottom"].set_color("white")
        axes[0].spines["left"].set_color("white")

        self.centroid_plot_objs = {
            bodypart: axes[0].scatter(
                [],
                [],
                s=2,
                zorder=102,
                color=color,
                label=f"{bodypart} position",
                # animated=True,
                alpha=0.6,
            )
            for color, bodypart in zip(COLOR_SWATCH, self.centroids.keys())
        }
        self.centroid_position_dot = axes[0].scatter(
            [],
            [],
            s=5,
            zorder=102,
            color="#b045f3",
            label="centroid position",
            alpha=0.6,
        )
        (self.orientation_line,) = axes[0].plot(
            [],
            [],
            color="cyan",
            linewidth=1,
            label="Orientation",
        )

        axes[0].set_xlabel("")
        axes[0].set_ylabel("")

        x_left, x_right = axes[0].get_xlim()
        y_low, y_high = axes[0].get_ylim()

        axes[0].set_aspect(
            abs((x_right - x_left) / (y_low - y_high)) * self.ratio
        )
        axes[0].spines["top"].set_color("black")
        axes[0].spines["right"].set_color("black")

        time_delta = pd.Timedelta(
            self.position_time[0] - self.position_time[-1]
        ).total_seconds()

        axes[0].legend(loc="lower right", fontsize=4)
        self.title = axes[0].set_title(
            f"time = {time_delta:3.4f}s\n frame = {0}",
            fontsize=8,
        )
        axes[0].axis("off")

        if self.likelihoods:
            self.likelihood_objs = {
                bodypart: axes[1].plot(
                    [],
                    [],
                    color=color,
                    linewidth=1,
                    clip_on=False,
                    label=bodypart,
                )[0]
                for color, bodypart in zip(
                    COLOR_SWATCH, self.likelihoods.keys()
                )
            }
            axes[1].set_ylim((0.0, 1))
            axes[1].set_xlim(
                (
                    self.window_ind[0] / self.frame_rate,
                    self.window_ind[-1] / self.frame_rate,
                )
            )
            axes[1].set_xlabel("Time [s]")
            axes[1].set_ylabel("Likelihood")
            axes[1].set_facecolor("black")
            axes[1].spines["top"].set_color("black")
            axes[1].spines["right"].set_color("black")
            axes[1].legend(loc="upper right", fontsize=4)

        self.fig = fig
        self.axes = axes

    def _get_frame(self, frame, crop_order=(2, 3, 0, 1)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not self.crop:
            return frame
        x1, x2, y1, y2 = self.crop_order
        return frame[
            self.crop[x1] : self.crop[x2], self.crop[y1] : self.crop[y2]
        ].copy()

    def _generate_single_frame(self, frame_ind):
        """Generate a single frame and save it as an image."""
        # Each frame open video bc video not picklable as multiprocessing arg
        video = cv2.VideoCapture(str(self.video_filename))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)

        ret, frame = video.read()
        if not ret:
            video.release()
            return None

        frame = self._get_frame(frame)

        _ = self.axes[0].imshow(frame)

        pos_ind = np.where(self.video_frame_inds == frame_ind)[0]

        if len(pos_ind) == 0:
            self.centroid_position_dot.set_offsets((np.NaN, np.NaN))
            for bodypart in self.centroid_plot_objs.keys():
                self.centroid_plot_objs[bodypart].set_offsets((np.NaN, np.NaN))
            self.orientation_line.set_data((np.NaN, np.NaN))
            self.title.set_text(f"time = {0:3.4f}s\n frame = {frame_ind}")
        else:
            pos_ind = pos_ind[0]
            likelihood_inds = pos_ind + self.window_ind
            neg_inds = np.where(likelihood_inds < 0)[0]
            likelihood_inds[neg_inds] = 0 if len(neg_inds) > 0 else -1

            dlc_centroid_data = self._get_centroid_data(pos_ind)

            for bodypart in self.centroid_plot_objs:
                self.centroid_plot_objs[bodypart].set_offsets(
                    _to_px(
                        data=self.centroids[bodypart][pos_ind],
                        cm_to_pixels=self.cm_to_pixels,
                    )
                )
            self.centroid_position_dot.set_offsets(dlc_centroid_data)
            _ = self._set_orient_line(frame, pos_ind)

            time_delta = pd.Timedelta(
                pd.to_datetime(self.position_time[pos_ind] * 1e9, unit="ns")
                - pd.to_datetime(self.position_time[0] * 1e9, unit="ns")
            ).total_seconds()

            self.title.set_text(
                f"time = {time_delta:3.4f}s\n frame = {frame_ind}"
            )
            if self.likelihoods:
                for bodypart in self.likelihood_objs.keys():
                    self.likelihood_objs[bodypart].set_data(
                        self.window_ind / self.frame_rate,
                        np.asarray(self.likelihoods[bodypart][likelihood_inds]),
                    )

        # Zero-padded filename based on the dynamic padding length
        frame_path = (
            self.output_temp_dir / f"temp-{frame_ind:0{self.pad_len}d}.png"
        )
        self.fig.savefig(frame_path, dpi=400)
        video.release()

        return frame_path

    def _get_centroid_data(self, pos_ind):
        def centroid_to_px(*idx):
            return _to_px(
                data=self.position_mean[idx], cm_to_pixels=self.cm_to_pixels
            )

        if not self.crop:
            return centroid_to_px(pos_ind)
        return np.hstack(
            (
                centroid_to_px((pos_ind, 0, np.newaxis)) - self.crop_offset_x,
                centroid_to_px((pos_ind, 1, np.newaxis)) - self.crop_offset_y,
            )
        )

    def _set_orient_line(self, frame, pos_ind):
        def orient_list(c):
            return [c, c + 30 * np.cos(self.orientation_mean[pos_ind])]

        if np.all(np.isnan(self.orientation_mean[pos_ind])):
            self.orientation_line.set_data((np.NaN, np.NaN))
        else:
            c0, c1 = self._get_centroid_data(pos_ind)
            self.orientation_line.set_data(orient_list(c0), orient_list(c1))

    def generate_frames_multithreaded(self):
        """Generate frames in parallel using ProcessPoolExecutor."""
        logger.info("Generating frames in parallel...")
        with ProcessPoolExecutor(max_workers=25) as executor:
            futures = {
                executor.submit(
                    self._generate_single_frame, frame_ind
                ): frame_ind
                for frame_ind in tqdm(self.frames, desc="Submitting frames")
            }
            for future in tqdm(
                as_completed(futures),
                total=self.n_frames,
                desc="Generating frames",
            ):
                try:
                    frame_path = future.result()
                except Exception as exc:
                    logger.error(
                        f"Error generating frame {futures[future]}: {exc}"
                    )

    def stitch_video_from_frames(self):
        """Stitch generated frames into a video using ffmpeg."""
        logger.info("Stitching frames into video...")

        frame_pattern = str(
            self.output_temp_dir / f"temp-%0{self.pad_len}d.png"
        )
        output_video = str(self.output_video_filename)

        ffmpeg_cmd = (
            "ffmpeg -hide_banner -loglevel error -y "
            + f"-r {self.fps} -i {frame_pattern} "
            + f"-c:v libx264 -pix_fmt yuv420p {output_video}"
        )
        os_system(ffmpeg_cmd)


def make_video(**kwargs):
    """Passthrough for VideoMaker class for backwards compatibility."""
    VideoMaker(**kwargs)
