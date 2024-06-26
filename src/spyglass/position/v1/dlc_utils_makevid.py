# Convenience functions
# some DLC-utils copied from datajoint element-interface utils.py
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

from spyglass.position.v1.dlc_utils import convert_to_pixels as _to_px
from spyglass.position.v1.dlc_utils import fill_nan
from spyglass.utils import logger

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
        processor="opencv",  # opencv, opencv-trodes, matplotlib
        video_time=None,
        frames=None,
        percent_frames=1,
        output_video_filename="output.mp4",
        cm_to_pixels=1.0,
        disable_progressbar=False,
        crop=None,
        arrow_radius=15,
        circle_radius=8,
    ):
        self.video_filename = video_filename
        self.video_frame_inds = video_frame_inds
        self.position_mean = position_mean
        self.orientation_mean = orientation_mean
        self.centroids = centroids
        self.likelihoods = likelihoods
        self.position_time = position_time
        self.processor = processor
        self.video_time = video_time
        self.frames = frames
        self.percent_frames = percent_frames
        self.output_video_filename = output_video_filename
        self.cm_to_pixels = cm_to_pixels
        self.disable_progressbar = disable_progressbar
        self.crop = crop
        self.arrow_radius = arrow_radius
        self.circle_radius = circle_radius

        if not Path(self.video_filename).exists():
            raise FileNotFoundError(f"Video not found: {self.video_filename}")

        if frames is None:
            self.n_frames = (
                int(self.orientation_mean.shape[0])
                if processor == "opencv-trodes"
                else int(len(video_frame_inds) * percent_frames)
            )
            self.frames = np.arange(0, self.n_frames)
        else:
            self.n_frames = len(frames)

        self.tqdm_kwargs = {
            "iterable": (
                range(self.n_frames - 1)
                if self.processor == "opencv-trodes"
                else self.frames
            ),
            "desc": "frames",
            "disable": self.disable_progressbar,
        }

        # init for cv
        self.video, self.frame_size = None, None
        self.frame_rate, self.out = None, None
        self.source_map = {
            "DLC": RGB_BLUE,
            "Trodes": RGB_ORANGE,
            "Common": RGB_PINK,
        }

        # intit for matplotlib
        self.image, self.title, self.progress_bar = None, None, None
        self.crop_offset_x = crop[0] if crop else 0
        self.crop_offset_y = crop[2] if crop else 0
        self.centroid_plot_objs, self.centroid_position_dot = None, None
        self.orientation_line = None
        self.likelihood_objs = None
        self.window_ind = np.arange(501) - 501 // 2

        self.make_video()

    def make_video(self):
        if self.processor == "opencv":
            self.make_video_opencv()
        elif self.processor == "opencv-trodes":
            self.make_trodes_video()
        elif self.processor == "matplotlib":
            self.make_video_matplotlib()

    def _init_video(self):
        logger.info(f"Making video: {self.output_video_filename}")
        self.video = cv2.VideoCapture(self.video_filename)
        self.frame_size = (
            (int(self.video.get(3)), int(self.video.get(4)))
            if not self.crop
            else (
                self.crop[1] - self.crop[0],
                self.crop[3] - self.crop[2],
            )
        )
        self.frame_rate = self.video.get(5)

    def _init_cv_video(self):
        _ = self._init_video()
        self.out = cv2.VideoWriter(
            filename=self.output_video_filename,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=self.frame_rate,
            frameSize=self.frame_size,
            isColor=True,
        )
        frames_log = (
            f"\tFrames start: {self.frames[0]}\n" if np.any(self.frames) else ""
        )
        inds_log = (
            f"\tVideo frame inds: {self.video_frame_inds[0]}\n"
            if np.any(self.video_frame_inds)
            else ""
        )
        logger.info(
            f"\n{frames_log}{inds_log}\tcv2 ind start: {int(self.video.get(1))}"
        )

    def _close_cv_video(self):
        self.video.release()
        self.out.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:  # if cv is already closed or does not have func
            pass
        logger.info(f"Finished video: {self.output_video_filename}")

    def _get_frame(self, frame, init_only=False, crop_order=(0, 1, 2, 3)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if init_only or not self.crop:
            return frame
        x1, x2, y1, y2 = self.crop_order
        return frame[
            self.crop[x1] : self.crop[x2], self.crop[y1] : self.crop[y2]
        ].copy()

    def _video_set_by_ind(self, time_ind):
        if time_ind == 0:
            self.video.set(1, time_ind + 1)
        elif int(self.video.get(1)) != time_ind - 1:
            self.video.set(1, time_ind - 1)

    def _all_num(self, *args):
        return all(np.all(~np.isnan(data)) for data in args)

    def _make_arrow(
        self,
        position,
        orientation,
        color,
        img,
        thickness=4,
        line_type=8,
        tipLength=0.25,
        shift=cv2.CV_8U,
    ):
        if not self._all_num(position, orientation):
            return
        arrow_tip = (
            int(position[0] + self.arrow_radius * np.cos(orientation)),
            int(position[1] + self.arrow_radius * np.sin(orientation)),
        )
        cv2.arrowedLine(
            img=img,
            pt1=tuple(position.astype(int)),
            pt2=arrow_tip,
            color=color,
            thickness=thickness,
            line_type=line_type,
            tipLength=tipLength,
            shift=shift,
        )

    def _make_circle(
        self,
        data,
        color,
        img,
        radius=None,
        thickness=-1,
        shift=cv2.CV_8U,
        **kwargs,
    ):
        if not self._all_num(data):
            return
        cv2.circle(
            img=img,
            center=tuple(data.astype(int)),
            radius=radius or self.circle_radius,
            color=color,
            thickness=thickness,
            shift=shift,
        )

    def make_video_opencv(self):
        _ = self._init_cv_video()

        if self.video_time:
            self.position_mean = {
                key: fill_nan(
                    self.position_mean[key]["position"],
                    self.video_time,
                    self.position_time,
                )
                for key in self.position_mean.keys()
            }
            self.orientation_mean = {
                key: fill_nan(
                    self.position_mean[key]["orientation"],
                    self.video_time,
                    self.position_time,
                )
                for key in self.position_mean.keys()
            }

        for time_ind in tqdm(**self.tqdm_kwargs):
            _ = self._video_set_by_ind(time_ind)

            is_grabbed, frame = self.video.read()

            if not is_grabbed:
                break

            frame = self._get_frame(frame)

            cv2.putText(
                img=frame,
                text=f"time_ind: {int(time_ind)} video frame: {int(self.video.get(1))}",
                org=(10, 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=RGB_YELLOW,
                thickness=1,
            )

            if time_ind < self.video_frame_inds[0] - 1:
                self.out.write(self._get_frame(frame, init_only=True))
                continue

            pos_ind = time_ind - self.video_frame_inds[0]

            for key in self.position_mean:
                position = _to_px(
                    data=self.position_mean[key][pos_ind],
                    cm_to_pixels=self.cm_to_pixels,
                )
                orientation = self.orientation_mean[key][pos_ind]
                cv_kwargs = {
                    "img": frame,
                    "color": self.source_map[key],
                }
                self._make_arrow(position, orientation, **cv_kwargs)
                self._make_circle(data=position, **cv_kwargs)

            self._get_frame(frame, init_only=True)
            self.out.write(frame)
        self._close_cv_video()
        return

    def make_trodes_video(self):
        _ = self._init_cv_video()

        if np.any(self.video_time):
            centroids = {
                color: fill_nan(
                    variable=data,
                    video_time=self.video_time,
                    variable_time=self.position_time,
                )
                for color, data in self.centroids.items()
            }
            position_mean = fill_nan(
                self.position_mean, self.video_time, self.position_time
            )
            orientation_mean = fill_nan(
                self.orientation_mean, self.video_time, self.position_time
            )

        for time_ind in tqdm(**self.tqdm_kwargs):
            is_grabbed, frame = self.video.read()
            if not is_grabbed:
                break

            frame = self._get_frame(frame)

            red_centroid = centroids["red"][time_ind]
            green_centroid = centroids["green"][time_ind]
            position = position_mean[time_ind]
            position = _to_px(data=position, cm_to_pixels=self.cm_to_pixels)
            orientation = orientation_mean[time_ind]

            self._make_circle(data=red_centroid, img=frame, color=RGB_YELLOW)
            self._make_circle(data=green_centroid, img=frame, color=RGB_PINK)
            self._make_arrow(
                position=position,
                orientation=orientation,
                color=RGB_WHITE,
                img=frame,
            )
            self._make_circle(data=position, img=frame, color=RGB_WHITE)
            self._get_frame(frame, init_only=True)
            self.out.write(frame)

        self._close_cv_video()

    def make_video_matplotlib(self):
        import matplotlib.animation as animation

        self.position_mean = self.position_mean["DLC"]
        self.orientation_mean = self.orientation_mean["DLC"]

        _ = self._init_video()

        video_slowdown = 1
        fps = int(np.round(self.frame_rate / video_slowdown))
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=fps, bitrate=-1)

        ret, frame = self.video.read()
        frame = self._get_frame(frame, crop_order=(2, 3, 0, 1))

        frame_ind = 0
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
        self.image = axes[0].imshow(frame, animated=True)

        logger.info(f"frame after init plot: {self.video.get(1)}")

        self.centroid_plot_objs = {
            bodypart: axes[0].scatter(
                [],
                [],
                s=2,
                zorder=102,
                color=color,
                label=f"{bodypart} position",
                animated=True,
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
            animated=True,
            alpha=0.6,
        )
        (self.orientation_line,) = axes[0].plot(
            [],
            [],
            color="cyan",
            linewidth=1,
            animated=True,
            label="Orientation",
        )

        axes[0].set_xlabel("")
        axes[0].set_ylabel("")

        ratio = (
            (self.crop[3] - self.crop[2]) / (self.crop[1] - self.crop[0])
            if self.crop
            else self.frame_size[1] / self.frame_size[0]
        )

        x_left, x_right = axes[0].get_xlim()
        y_low, y_high = axes[0].get_ylim()

        axes[0].set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
        axes[0].spines["top"].set_color("black")
        axes[0].spines["right"].set_color("black")

        time_delta = pd.Timedelta(
            self.position_time[0] - self.position_time[-1]
        ).total_seconds()

        axes[0].legend(loc="lower right", fontsize=4)
        self.title = axes[0].set_title(
            f"time = {time_delta:3.4f}s\n frame = {frame_ind}",
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
                    animated=True,
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

        self.progress_bar = tqdm(leave=True, position=0)
        self.progress_bar.reset(total=self.n_frames)

        movie = animation.FuncAnimation(
            fig,
            self._update_plot,
            frames=self.frames,
            interval=1000 / fps,
            blit=True,
        )
        movie.save(self.output_video_filename, writer=writer, dpi=400)
        self.video.release()
        plt.style.use("default")
        logger.info("finished making video with matplotlib")
        return

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

    def _update_plot(self, time_ind, *args):
        _ = self._video_set_by_ind(time_ind)

        ret, frame = self.video.read()
        if ret:
            frame = self._get_frame(frame, crop_order=(2, 3, 0, 1))
            self.image.set_array(frame)

        pos_ind = np.where(self.video_frame_inds == time_ind)[0]

        if len(pos_ind) == 0:
            self.centroid_position_dot.set_offsets((np.NaN, np.NaN))
            for bodypart in self.centroid_plot_objs.keys():
                self.centroid_plot_objs[bodypart].set_offsets((np.NaN, np.NaN))
            self.orientation_line.set_data((np.NaN, np.NaN))
            self.title.set_text(f"time = {0:3.4f}s\n frame = {time_ind}")
            self.progress_bar.update()
            return

        pos_ind = pos_ind[0]
        likelihood_inds = pos_ind + self.window_ind
        # initial implementation did not cover case of both neg and over < 0
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

        self.title.set_text(f"time = {time_delta:3.4f}s\n frame = {time_ind}")
        for bodypart in self.likelihood_objs.keys():
            self.likelihood_objs[bodypart].set_data(
                self.window_ind / self.frame_rate,
                np.asarray(self.likelihoods[bodypart][likelihood_inds]),
            )
        self.progress_bar.update()

        return (
            self.image,
            self.centroid_position_dot,
            self.orientation_line,
            self.title,
        )


def make_video(**kwargs):
    VideoMaker(**kwargs)
