import re
from collections import defaultdict
from datetime import datetime
from functools import cached_property
from hashlib import md5
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator

import datajoint as dj
import pandas as pd
from datajoint.logging import logger as dj_logger
from IPython import embed
from tqdm import tqdm

from spyglass.common import (
    AnalysisNwbfile,
    Nwbfile,
    Session,
    Subject,
    Task,
    TaskEpoch,
)
from spyglass.utils import SpyglassMixin
from spyglass.utils.database_settings import SHARED_MODULES
from spyglass.utils.dj_helper_fn import bytes_to_human_readable

# Please submit a GitHub issue to request these fields be made configurable.
MONITORED_EXTENSIONS = [".rec"]
MONITORED_DIRS = ["/nimbus", "/stelmo", "/cumulus"]

# If a file is older than 10/14/2025, it is considered "grandfathered"
# and will expire in 9 months instead of 6.
GRANDFATHER_CUTOFF = datetime(2025, 10, 14)
EXPIRATION_GRANDFATHER = 9  # months from found date to expire
EXPIRATION_MONTHS = 6  # months before a file is considered expired

schema = dj.Schema("cbroz_common_file_monitoring")


@schema
class TrackedFile(dj.Manual):
    definition = """
    file_id     : char(32)     # hash of path
    ---
    owner=''    : varchar(64)  # owner of the file
    created_at  : timestamp    # time of file creation
    size        : bigint       # size of the file in bytes
    path        : varchar(255) # path to the file
    """

    def _iter_exts(self, path: Path) -> Iterator[Path]:
        """Iterate over all files with monitored extensions in a directory."""
        iters = Iterable[Iterator[Path]] = (
            path.rglob(f"*{ext}") for ext in MONITORED_EXTENSIONS
        )
        chained = chain.from_iterable(iters)
        yield from chained

    def _to_id(self, path):
        """Generate a unique ID for a given file path."""
        return md5(path.encode()).hexdigest()

    def _insert_from_path(self, path: Path) -> dict:
        """Create a dictionary for insertion from a Path object."""
        path = Path(path)
        try:
            owner = path.owner()
        except (KeyError, OSError):
            owner = None
        return dict(
            file_id=self._to_id(path.as_posix()),  # Generate unique ID
            path=path.as_posix(),
            created_at=datetime.fromtimestamp(path.stat().st_ctime),
            size=path.stat().st_size,  # Get file size
            owner=owner,
        )

    def entry_search(self):
        """Crawl monitored directories and update the table."""
        all_paths = {Path(p).as_posix() for p in self.fetch("path")}

        inserts = []
        del_paths = []
        for root_dir in MONITORED_DIRS:
            root_path = Path(f"/{root_dir}/")
            for path in tqdm(
                self._iter_exts(root_path), desc=f"Crawling {root_dir}"
            ):
                if not path.is_file():
                    continue
                if not path.exists():
                    del_paths.append(path)
                    continue
                if path not in all_paths:
                    inserts.append(self._insert_from_path(path))

        self.insert(inserts, skip_duplicates=True)

        del_keys = self & f"path IN {tuple([str(p) for p in del_paths])}"
        if len(del_keys):
            print(f"Removing {len(del_keys)} missing files")
            del_keys.delete(safemode=False)

    def delete_missing(self):
        """Delete entries for files that no longer exist."""
        del_keys = []
        for key in tqdm(self, desc="Checking for missing files"):
            path = Path(key["path"])
            if not path.exists():
                del_keys.append(dict(file_id=key["file_id"]))
        if del_keys:
            print(f"Removing {len(del_keys)} missing files")
            self.delete(del_keys)

    def run(self):
        _ = self.entry_search()
        _ = self.delete_missing()

    @property
    def expired(self):
        """Find files that are older than the expiration threshold."""
        cutoff = f"{GRANDFATHER_CUTOFF:%Y-%m-%d}"
        grandf_template = "created_at {} " + f"'{cutoff}'"
        expire_template = "created_at < DATE_SUB(CURDATE(), INTERVAL {} MONTH)"

        grandf_restr = (
            grandf_template.format("<=")
            + " AND "
            + expire_template.format(EXPIRATION_GRANDFATHER)
        )
        expire_restr = (
            grandf_template.format(">")
            + " AND "
            + expire_template.format(EXPIRATION_MONTHS)
        )

        return self & f"({grandf_restr}) OR ({expire_restr})"


@schema
class FileMatch(SpyglassMixin, dj.Computed):
    definition = """
    -> TrackedFile
    ---
    matches          : int           # number of matching Epochs
    restriction=NULL : blob          # restriction used to find matches
    """

    class FileEpoch(dj.Part):
        definition = """
        -> FileMatch
        -> TaskEpoch
        """

    _n_epochs = len(TaskEpoch())
    _tracked_issues = defaultdict(list)

    @cached_property
    def subj_ids(self):
        """Extract subject IDs from the Session table."""
        return [
            id
            for id in set(Subject().fetch("subject_id"))
            if not id.isnumeric()
        ]

    def parse_date(self, path):
        """Extract date from the file path.

        1. Check for an 8-digit date (YYYYMMDD) or 6-digit date (MMDDYY).
        2. Convert MMDDYY to YYYYMMDD if the year is greater than 17.
        3. Handle special case for leap year error (20210229).
        """
        ret = None

        # Match YYYYMMDD or MMDDYY formats
        date_match_YY = re.search(r"(\d{8})", path)
        if date_match_YY:
            ret = date_match_YY.group(1)
        date_match_MM = re.search(r"(\d{6})", path)
        if date_match_MM:
            date_str = date_match_MM.group(1)
            date_yr = date_str[-2:]
            if int(date_yr) > 17:  # turn MMDDYY into YYYYMMDD
                ret = f"20{date_yr}{date_str[:-2]}"

        if ret == "20210229":  # special case for leap year error
            # '/nimbus/alison/em8/adjusting_*/20210229_preadj.rec'
            return "20210301"

        if ret:  # Validate the extracted date
            try:
                datetime.strptime(ret, "%Y%m%d")
                return ret
            except ValueError:
                pass

        self._tracked_issues["date"].append(path)
        return None

    def _to_tuple(self, matches):
        """Convert possible matches to a tuple for SQL IN clause.

        IN fails with 1 or 0 matches, so add fakes to simplify restriction.
        """
        fakes = ("fake1", "fake2")
        return tuple(matches) + fakes if len(matches) else fakes

    def parse_subj_ids(self, path):
        """Extract subject IDs from the file path.

        Returns a tuple of subject IDs that match the path."""
        matches = {id for id in self.subj_ids if id.lower() in path.lower()}
        if matches:
            return self._to_tuple(matches)
        self._tracked_issues["subject"].append(path)
        return None

    def parse_intervals(self, path, matches):
        """Extract task epochs interval list from the file path.

        For found matches, check the path for interval_list_name.
        """
        if len(matches) == 0:
            return None

        path_lower = path.lower()
        interval_patt = r"\d{2}_[rs]\d"  # basic ##_{s,r}# match

        intervals = set()
        for interval in matches.fetch("interval_list_name"):
            interv_lower = interval.lower()
            if interv_lower in path_lower:  # exact match
                intervals.add(interval)
            if re.match(interval_patt, interv_lower):
                # names with 01_s1 as 01S1 in the path
                if interv_lower.replace("_", "") in path_lower:
                    intervals.add(interval)
                # names with 01_s1 and s1 as a subdirectory
                if interv_lower.split("_")[1] in path.split("/"):
                    intervals.add(interval)

            if interval == interval.lower():
                continue

            # For mixed-case intervals, e.g. Lewis_Rev2_Sleep4, check parts. eg,
            # /shijie/recording_pilot/**/20240115_140107_Lewis_Rev2_Sleep4.rec
            for subinterval in interval.split("_"):
                if subinterval.isnumeric():
                    continue  # ignore likely epoch numbers
                if subinterval.lower() in path_lower.replace("_", ""):
                    intervals.add(interval)

        # Remove substring intervals: e.g., ##_Session1 and ##_Session1b
        # Take the most complete matching interval
        intervals = [
            il
            for il in intervals
            if not any(
                il != other and il.split("_")[-1] in other.split("_")[-1]
                for other in intervals
            )
        ]

        if intervals:
            return self._to_tuple(intervals)

        self._tracked_issues["interval"].append(path)
        return None

    def parse_tasks(self, path, matches):
        """Extract task names from the file path.

        For found matches, check the path for task_name.
        """
        if len(matches) == 0:
            return None

        tasks = set()
        for task in matches.fetch("task_name"):
            for subtask in task.split(" "):
                if subtask.lower() in path.lower():
                    tasks.add(task)
        if tasks:
            return self._to_tuple(tasks)

        self._tracked_issues["task"].append(path)
        return None

    def parse_epoch(self, path, matches):
        """Extract epoch number from the file path.

        For found matches, check the path for epoch number.
        """
        if len(matches) == 0:
            return None

        # Find all 2-digit numbers surrounded by underscores, if < 22
        pattern = r"(?<=_)(\d{2})(?=_|\b)"
        epochs = set(
            [int(ep) for ep in re.findall(pattern, path) if int(ep) < 22]
        )

        if len(epochs) == 1:  # Only if one epoch is found
            return list(epochs)[0]

        self._tracked_issues["epoch"].append(path)
        return None

    @cached_property
    def default_matches(self):
        matches = Session * TaskEpoch

        # Remove known duplicates from potential matches
        known_dupes = "|".join(
            [
                "__.nwb$",  # ignore files with double underscores
                "^SC200",  # SC200 are 'blind' subjects
                "^jfm",  # single duplicate
                "_new_.nwb$",  # ignore files with _new_ in the name
                "_copy_.nwb$",  # ignore files with _copy_ in the name
                "^SB2spike",
                "tutorial",
            ]
        )
        matches &= f'nwb_file_name NOT REGEXP "{known_dupes}"'
        return matches

    def make(self, key):
        date_restr, subj_restr, epoch_restr, file_date_restr = [None] * 4

        path = Path((TrackedFile & key).fetch1("path"))
        matches = self.default_matches

        # Parse the date from the path as 6 or 8 digit string.
        date_str = self.parse_date(path.as_posix())
        if date_str:  # restriction strings preserved for debugging
            date_restr = f"DATE(session_start_time) = '{date_str}'"
            matches &= date_restr
        else:
            self.insert1(dict(key, matches=0, restriction=None))
            return  # No date found, skip this file

        # Parse subject IDs from the path from subject table
        subj_ids = self.parse_subj_ids(path.as_posix())
        if subj_ids:
            subj_restr = f"subject_id IN {subj_ids}"
            matches &= subj_restr
        else:
            self.insert1(dict(key, matches=0, restriction=None))
            return

        # Parse for interval names
        intervals = self.parse_intervals(path.as_posix(), matches)
        if intervals:
            epoch_restr = f"interval_list_name IN {intervals}"
            matches &= epoch_restr

        # Parse for task names
        tasks = self.parse_tasks(path.as_posix(), matches)
        if tasks:
            task_restr = f"task_name IN {tasks}"
            matches &= task_restr

        # Parse for epoch number
        epoch = self.parse_epoch(path.as_posix(), matches)
        if epoch is not None:
            epoch_restr = f"epoch = {epoch}"
            matches &= epoch_restr

        # Check for date str in nwb_file_name for mislabeled start times
        options = len(matches)
        if 1 < options < self._n_epochs and date_str:
            file_date_restr = f"nwb_file_name LIKE '%{date_str}%'"
            matches &= file_date_restr

        debug_fields = [
            "nwb_file_name",
            "epoch",
            "subject_id",
            "session_id",
            "task_name",
            "session_start_time",
            "interval_list_name",
        ]

        options = len(matches)
        if options > 1:
            print(f"Failed to restrict: {path}")
            debug_view = dj.U(*debug_fields).aggr(matches)
            print(debug_view)
            print(f"NWBs: {set(debug_view.fetch('nwb_file_name'))}")
            return

        # Insert the match if a unique session is found
        epochs = matches.fetch("KEY", as_dict=True)
        self.insert1(
            {**key, "matches": options, "restriction": matches.restriction}
        )
        self.FileEpoch.insert([dict(key, **epoch) for epoch in epochs])

    # --------------------------- Generate reports ---------------------------

    def disk_usage(self, by_owner=False):
        """Calculate the disk usage of the FileMatch table by owner."""
        if len(self) == 0:
            return "No records in FileMatch table."

        joined = self * TrackedFile
        partials = dj.U("owner").aggr(joined, tot="SUM(size)")
        grand_total = 0
        for owner, total_size in partials.fetch(order_by="tot DESC"):
            grand_total += total_size
            if not by_owner:
                continue
            readable = bytes_to_human_readable(total_size)
            print(f"{owner:>12}: {readable:>10}")

        grand_readable = bytes_to_human_readable(grand_total)
        print(f"{'Grand Total':>12}: {grand_readable:>10}")

    def print_user_files(self):
        joined = (self * TrackedFile).proj(
            "owner", "path", "size", in_db="matches"
        )
        for owner in set(joined.fetch("owner")):
            subset = joined & f"owner = '{owner}'"
            fetched = subset.fetch(
                "path",
                "size",
                "in_db",
                order_by="in_db DESC, size DESC",
                as_dict=True,
            )
            df = pd.DataFrame(fetched)
            total = bytes_to_human_readable(df["size"].sum()).replace(" ", "")
            df["size"] = df["size"].apply(bytes_to_human_readable)
            df.to_csv(f"rec_csvs/rec_files_{owner}_{total}.csv", index=False)
            print(f"CSV: Owner {owner} {total}, {len(df)} files")

    def see_matches(self, key):
        """Display matches for a given key."""
        _ = self.ensure_single_entry()

        print(f"Matches for {key}:")
        restriction = self.fetch1("restriction")
        if restriction is None:
            print("No matches found.")
            return
        return self.default_matches & restriction


@schema
class AnalysisFileIssues(dj.Computed):
    definition = """
    -> AnalysisNwbfile
    ---
    on_disk=1  : bool # whether the analysis file exists
    can_read=1 : bool # whether the analysis file is readable
    issue=NULL : varchar(255) # description of the issue
    table=NULL : varchar(64) # name of the table that created the analysis file
    """

    # NOTE: exists and readable are SQL keywords. Use on_disk and can_read

    not_exist = dict(on_disk=False, can_read=False)
    checksum = dict(on_disk=True, can_read=False)

    @cached_property
    def analysis_children(self):
        banned = [
            "`common_nwbfile`.`analysis_nwbfile_log`",
            "`cbroz_check_files`.`__analysis_file_issues`",
        ]
        return [
            c
            for c in AnalysisNwbfile().children(as_objects=True)
            if c.full_table_name not in banned
        ]

    def get_tbl(self, key):
        ret = []
        f_key = dict(analysis_file_name=key["analysis_file_name"])
        for child in self.analysis_children:
            if child & f_key:
                ret.append(child.full_table_name)
        if len(ret) != 1:
            raise ValueError(
                f"{len(ret)} tables for {key['analysis_file_name']}: {ret}"
            )
        return ret[0]

    def make(self, key):
        """
        Check if the analysis file exists and is readable.
        """
        prev_level = dj_logger.level
        dj_logger.setLevel("ERROR")

        insert = key.copy()
        fname = None
        try:
            fname = AnalysisNwbfile().get_abs_path(key["analysis_file_name"])
        except FileNotFoundError as e:
            insert = dict(key, **self.not_exist, issue=e.args[0])
        except dj.DataJointError as e:
            insert = dict(
                key, **self.checksum, issue=e.args[0], table=self.get_tbl(key)
            )
        if fname is not None and not Path(fname).exists():
            insert.update(**self.not_exist, issue=f"path not found: {fname}")
        self.insert1(insert, skip_duplicates=True)

        dj_logger.setLevel(prev_level)

    def check_exists(self, key):
        to_check = self & key & "`can_read`=0"
        if not to_check:
            return
        for key in to_check.fetch("KEY", as_dict=True):
            this_path = None
            fname = key["analysis_file_name"]
            try:
                _ = AnalysisNwbfile().get_abs_path(fname)
            except dj.DataJointError as e:
                this_path = Path(e.args[0].split("'")[1])
            prefix = "Exists" if this_path and this_path.exists() else "Missing"
            print(f"{prefix}: {this_path}")

    def show_downstream(self, restriction=True):
        entries = (self & "can_read=0" & restriction).fetch("KEY", as_dict=True)
        if not entries:
            print("No issues found.")
            return
        ret = [(c & entries) for c in self.analysis_children if (c & entries)]
        if not ret:
            print("No issues found.")
            return
        return ret if len(ret) > 1 else ret[0]


if __name__ == "__main__":
    pop_kwargs = dict(
        order="random", display_progress=True, reserve_jobs=True, processes=15
    )

    TrackedFile().run()
    FileMatch().populate(**pop_kwargs)
    FileMatch().print_user_files()

    AnalysisFileIssues().populate(**pop_kwargs)
    AnalysisFileIssues().show_downstream(restriction=True)
