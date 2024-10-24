# Convenience functions

# some DLC-utils copied from datajoint element-interface utils.py
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import system as os_system
from pathlib import Path
from typing import Tuple

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
        batch_size=500,
        max_workers=25,
        max_jobs_in_queue=250,
        debug=False,
        key_hash=None,
        *args,
        **kwargs,
    ):
        """Create a video from a set of position data.

        Uses batch size as frame count for processing steps. All in temp_dir.
            1. Extract frames from original video to 'orig_XXXX.png'
            2. Multithread pool frames to matplotlib 'plot_XXXX.png'
            3. Stitch frames into partial video 'partial_XXXX.mp4'
            4. Concatenate partial videos into final video output

        """
        if processor != "matplotlib":
            raise ValueError(
                "open-cv processors are no longer supported. \n"
                + "Use matplotlib or submit a feature request via GitHub."
            )

        # key_hash supports resume from previous run
        self.temp_dir = Path(temp_dir) / f"dlc_vid_{key_hash}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Temporary directory: {self.temp_dir}")

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
        self.window_ind = np.arange(501) - 501 // 2
        self.debug = debug

        self.dropped_frames = set()

        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_jobs_in_queue = max_jobs_in_queue

        self.ffmpeg_log_args = ["-hide_banner", "-loglevel", "error"]
        self.ffmpeg_fmt_args = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]

        _ = self._set_frame_info()
        _ = self._set_plot_bases()

        logger.info(
            f"Making video: {self.output_video_filename} "
            + f"in batches of {self.batch_size}"
        )
        self.process_frames()
        plt.close(self.fig)
        logger.info(f"Finished video: {self.output_video_filename}")
        logger.debug(f"Dropped frames: {self.dropped_frames}")

        shutil.rmtree(self.temp_dir)  # Clean up temp directory

    def _set_frame_info(self):
        """Set the frame information for the video."""
        logger.debug("Setting frame information")

        width, height, self.frame_rate = self._get_input_stats()
        self.frame_size = (
            (width, height)
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
        self.fps = int(np.round(self.frame_rate))

        if self.frames is None:
            self.n_frames = int(
                len(self.video_frame_inds) * self.percent_frames
            )
            self.frames = np.arange(0, self.n_frames)
        else:
            self.n_frames = len(self.frames)
        self.pad_len = len(str(self.n_frames))

    def _get_input_stats(self, video_filename=None) -> Tuple[int, int]:
        """Get the width and height of the video."""
        logger.debug("Getting video dimensions")

        video_filename = video_filename or self.video_filename
        ret = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "csv=p=0:s=x",
                video_filename,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if ret.returncode != 0:
            raise ValueError(f"Error getting video dimensions: {ret.stderr}")

        stats = ret.stdout.strip().split("x")
        width, height = tuple(map(int, stats[:-1]))
        frame_rate = eval(stats[-1])

        return width, height, frame_rate

    def _set_plot_bases(self):
        """Create the figure and axes for the video."""
        logger.debug("Setting plot bases")
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

    def _generate_single_frame(self, frame_ind):
        """Generate a single frame and save it as an image."""
        padded = self._pad(frame_ind)
        frame_file = self.temp_dir / f"orig_{padded}.png"
        if not frame_file.exists():
            self.dropped_frames.add(frame_ind)
            print(f"\rFrame not found: {frame_file}", end="")
            return
        frame = plt.imread(frame_file)
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
        frame_path = self.temp_dir / f"plot_{padded}.png"
        self.fig.savefig(frame_path, dpi=400)
        plt.cla()  # clear the current axes

        return frame_ind

    def process_frames(self):
        """Process video frames in batches and generate matplotlib frames."""

        progress_bar = tqdm(leave=True, position=0, disable=self.debug)
        progress_bar.reset(total=self.n_frames)

        for start_frame in range(0, self.n_frames, self.batch_size):
            if start_frame >= self.n_frames:  # Skip if no frames left
                break
            end_frame = min(start_frame + self.batch_size, self.n_frames) - 1
            logger.debug(f"Processing frames: {start_frame} - {end_frame}")

            output_partial_video = (
                self.temp_dir / f"partial_{self._pad(start_frame)}.mp4"
            )
            if output_partial_video.exists():
                logger.debug(f"Skipping existing video: {output_partial_video}")
                progress_bar.update(end_frame - start_frame)
                continue

            self.ffmpeg_extract(start_frame, end_frame)
            self.plot_frames(start_frame, end_frame, progress_bar)
            self.ffmpeg_stitch_partial(start_frame, str(output_partial_video))

            for frame_file in self.temp_dir.glob("*.png"):
                frame_file.unlink()  # Delete orig and plot frames

        progress_bar.close()

        logger.info("Concatenating partial videos")
        self.concat_partial_videos()

    def _debug_print(self, msg=None, end=""):
        """Print a self-overwiting message if debug is enabled."""
        if self.debug:
            print(f"\r{msg}", end=end)

    def plot_frames(self, start_frame, end_frame, progress_bar=None):
        logger.debug(f"Plotting   frames: {start_frame} - {end_frame}")
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            jobs = {}  # dict of jobs

            frames_left = end_frame - start_frame
            frames_iter = iter(range(start_frame, end_frame))

            while frames_left:
                while len(jobs) < self.max_jobs_in_queue:
                    try:
                        this_frame = next(frames_iter)
                        self._debug_print(f"Submit: {this_frame}")
                        job = executor.submit(
                            self._generate_single_frame, this_frame
                        )
                        jobs[job] = this_frame
                    except StopIteration:
                        break  # No more frames to submit

                for job in as_completed(jobs):
                    frames_left -= 1
                    try:
                        ret = job.result()
                    except IndexError:
                        ret = "IndexError"
                    self._debug_print(f"Finish: {ret}")
                    progress_bar.update()
                    del jobs[job]
        self._debug_print(end="\n")

    def ffmpeg_extract(self, start_frame, end_frame):
        """Use ffmpeg to extract a batch of frames."""
        logger.debug(f"Extracting frames: {start_frame} - {end_frame}")
        output_pattern = str(self.temp_dir / f"orig_%0{self.pad_len}d.png")

        # Use ffmpeg to extract frames
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # overwrite
            "-i",
            self.video_filename,
            "-vf",
            f"select=between(n\\,{start_frame}\\,{end_frame})",
            "-vsync",
            "vfr",
            "-start_number",
            str(start_frame),
            "-n",  # no overwrite
            output_pattern,
            *self.ffmpeg_log_args,
        ]
        ret = subprocess.run(ffmpeg_cmd, stderr=subprocess.PIPE)

        extracted = len(list(self.temp_dir.glob("orig_*.png")))
        logger.debug(f"Extracted  frames: {start_frame}, len: {extracted}")
        if extracted < self.batch_size - 1:
            logger.warning(
                f"Could not extract frames: {extracted} / {self.batch_size-1}"
            )
            one_err = "\n".join(str(ret.stderr).split("\\")[-3:-1])
            logger.debug(f"\nExtract Error: {one_err}")

    def _pad(self, frame_ind):
        return f"{frame_ind:0{self.pad_len}d}"

    def ffmpeg_stitch_partial(self, start_frame, output_partial_video):
        """Stitch a partial movie from processed frames."""
        logger.debug(f"Stitch part vid  : {start_frame}")
        frame_pattern = str(self.temp_dir / f"plot_%0{self.pad_len}d.png")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # overwrite
            "-r",
            str(self.fps),
            "-start_number",
            str(start_frame),
            "-i",
            frame_pattern,
            *self.ffmpeg_fmt_args,
            output_partial_video,
            *self.ffmpeg_log_args,
        ]
        try:
            ret = subprocess.run(
                ffmpeg_cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Error stitching partial video: {e.stderr}")

    def concat_partial_videos(self):
        """Concatenate all the partial videos into one final video."""
        partial_vids = sorted(self.temp_dir.glob("partial_*.mp4"))
        logger.debug(f"Concat part vids: {len(partial_vids)}")
        concat_list_path = self.temp_dir / "concat_list.txt"
        with open(concat_list_path, "w") as f:
            for partial_video in partial_vids:
                f.write(f"file '{partial_video}'\n")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # overwrite
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            *self.ffmpeg_fmt_args,
            str(self.output_video_filename),
            *self.ffmpeg_log_args,
        ]
        try:
            ret = subprocess.run(
                ffmpeg_cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Error stitching partial video: {e.stderr}")


def make_video(**kwargs):
    """Passthrough for VideoMaker class for backwards compatibility."""
    VideoMaker(**kwargs)
