# Coding for Reuse

<!--
NOTE: Code blocks are intentionally non-black-formatted.
Disable pre-commit mdformat-black if editing. Or commit with --no-verify.
https://github.com/hukkin/mdformat-black/issues/6
-->

*Reusing code requires that it be faster to read and change than it would be to
start from scratch.*

We can speed up that process by ...

1. Making reading predictable.
2. Atomizing - separating pieces into the smallest meaningful chunks.
3. Leaving notes via type hints, docstrings, and comments
4. Getting ahead of errors
5. Automating as much of the above as possible.

This document pulls from resources like
[Tidy First](https://www.oreilly.com/library/view/tidy-first/9781098151232/) and
[SOLID Principles](https://arjancodes.com/blog/solid-principles-in-python-programming/).
Experienced object-oriented developers may find these principles familiar.

## Predictable Formatting

- Many programming languages offer flexibility in how they are written.
- Tools like `black` and `isort` take away stylistic preferences in favor of one
    norm.
- Strict line limits (e.g., 80) make it easier to do side by side comparisons in
    git interfaces.
- `black` is also useful for detecting an error on save - if it doesn't run on
    what you wrote, there's an error somewhere.

Let's look at a few examples of the same code block formatted different ways...

### Original

```python
def get_data_interface(nwbfile, data_interface_name, data_interface_class=None, unused_other_arg=None):
    ret = { 'centroid_method': "two_pt_centroid", 'points': {'point1': 'greenLED', "point2": 'redLED_C'}, 'interpolate': True}
    for module in nwbfile.processing.values():
        match = module.data_interfaces.get(data_interface_name, None)
        if match is not None:
            if data_interface_class is not None and not isinstance(match, data_interface_class):
                continue
            ret.append(match)
    if len(ret) > 1:
        print(f"Multiple data interfaces with name '{data_interface_name}' found with identifier {nwbfile.identifier}.")
    if len(ret) >= 1:
        return ret[0]
    return None
```

### Black formatted

With `black`, we have a limited line length and indents reflect embedding.

```python
def get_data_interface(  # Each arg gets its own line
    nwbfile,
    data_interface_name,
    data_interface_class=None,
    unused_other_arg=None,
):
    ret = {  # dictionaries show embedding
        "centroid_method": "two_pt_centroid",
        "points": {
            "point1": "greenLED",
            "point2": "redLED_C",
        },
        "interpolate": True,
    }
    for module in nwbfile.processing.values():
        match = module.data_interfaces.get(data_interface_name, None)
        if match is not None:
            if data_interface_class is not None and not isinstance(
                match, data_interface_class
            ):  # long lines broken up
                continue
            ret.append(match)
    if len(ret) > 1:
        print(  # long strings need to be broken up manually
            f"Multiple data interfaces with name '{data_interface_name}' "
            f"found in NWBFile with identifier {nwbfile.identifier}. "
        )
    if len(ret) >= 1:
        return ret[0]
    return None
```

### Control flow adjustments

Although subjective, we can do even better by adjusting the logic to follow how
we read.

```python
from typing import Type
def get_data_interface(...):
    ret = {...}
    # decide no input early
    data_interface_class = data_interface_class or Type
    for match in [ # generate via list comprehension
        module.get_data_interface(data_interface_name)
        for module in nwbfile.processing.values()
    ]: # only process good case, no `continue`
        if match and isinstance(match, data_interface_class):
            ret.append(match)
    if len(ret) > 1:
        print(...)
    return ret[0] if len(ret) >= 1 else None # fits on one line
```

## Atomizing

Working memory limits our ability to understand long code blocks.

We can extract pieces into separate places to give them a name and make 'one'
memory chunk out of a set of functions.

Depending on the scope, chunks can be separated with ...

1. Paragraph breaks - to group instructions together.
2. Conditional assignment - for data maps local to a function.
3. Methods of a class - for functions that deserve a separate name.
4. Helpers in a script - for functions used multiple times in a schema.
5. Util scripts in a package - for functions used throughout a project.

### Atomizing example

- Let's read the next function as if we're revisiting old code.
- This example was taken from an existing project and adjusted for
    demonstration.
- Please review without commentary and make mental notes ow what each part line
    is doing and how they relate to other lines.

<details><summary>No commentary</summary>

```python
class MyTable(dj.Computed):
    ...

    def make(self, key):
        rat_name = key["rat_name"]
        ron_all_dict = {"some_data": 1}
        tonks_all_dict = {"other_data": 2}
        try:
            if len((OtherTable & key).fetch("cluster_id")[0]) > 0:
                if rat_name == "ron":
                    data_dict = ron_all_dict
                elif rat_name == "tonks":
                    data_dict = tonks_all_dict
                else:
                    raise ValueError(f"Unsupported rat {rat_name}")
                for data_key, data_value in data_dict.items():
                    try:
                        if data_value == 1:
                            cluster_spike_times = (OtherTable & key).fetch_nwb()[
                                0
                            ]["units"]["spike_times"]
                        else:
                            cluster_spike_times = (OtherTable & key).fetch_nwb()[
                                data_value - 1
                            ]["units"]["spike_times"][data_key]
                        self.insert1(cluster_spike_times)
                    except KeyError:
                        print("cluster missing", key["nwb_file_name"])
                else:
                    print("no spikes")
        except IndexError:
            print("no data")
```

</details>

<details><summary>With Commentary</summary>

Note how the numbers correspond to their counterparts - 1Q, 1A, 2Q, 2A ...

```python
class MyTable(dj.Computed):
    ...
    def make(self, key):
        rat_name = key["rat_name"] # 1Q. Can this function handle others?
        ron_all_dict = {"some_data": 1} # 2Q. Are these parameters?
        tonks_all_dict = {"other_data": 2}
        try:  # 3Q. What error could be thrown? And by what?
            if len((OtherTable & key).fetch("cluster_id")[0]) > 0: # 4Q. What happens if none?
                if rat_name == "ron":
                    data_dict = ron_all_dict # 2A. ok, we decide the data here
                elif rat_name == "tonks":
                    data_dict = tonks_all_dict
                else: # 1Q. Ok, we can only do these two
                    raise ValueError(f"Unsupported rat {rat_name}")
                for data_key, data_value in data_dict.items(): # 2A. Maybe parameter?
                    try: # 5Q. What could throw an error?
                        if data_value == 1:
                            cluster_spike_times = (OtherTable & key).fetch_nwb()[
                                0
                            ]["units"]["spike_times"] # 6Q. What do we need this for?
                        else:
                            cluster_spike_times = (OtherTable & key).fetch_nwb()[
                                data_value - 1
                            ]["units"]["spike_times"][data_key]
                        self.insert1(cluster_spike_times) # 6A. Ok, insertion
                    except KeyError: # 5A. Mayble this fetch is unreliable?
                        print("cluster missing", key["nwb_file_name"])
                else:
                    print("no spikes") # 4A. Ok we bail if no clusters
        except IndexError: # 3A. What could have thrown this? Are we sure nothing else?
            print("no data")
```

</details>

### Embedding

- The process of stream of consciousness coding often generates an embedding
    trail from core out
- Our mental model of A -> B -> C -> D may actually read like `D( C( B( A )))`
    or ...

1. Prepare for D
2. Open a loop for C
3. Add caveat B
4. Do core process A
5. Check other condition B
6. Close D

Let's contrast with an approach that reduces embedding.

```python
class MyTable(dj.Computed):
    ...
    def _get_cluster_times(self, key, nth_file, index): # We will need times
        clust = (OtherTable & key).fetch_nwb()[nth_file]["units"]["spike_times"]
        try: # Looks like this indexing may not return the data
            return clust[index] if nth_file == 0 else clust # if/then handled here
        except KeyError: # Show as err, keep moving
            logger.error("Cluster missing", key["nwb_file_name"])

    def make(self, key):
        rat_paramsets = {"ron": {"some_data": 1}, "tonks": {"other_data": 2}} # informative variable name
        if (rat_name := key["rat_name"]) not in rat_paramsets: # walrus operator `:=` can assign within `if`
            raise ValueError(f"Unsupported rat {rat_name}") # we can only handle a subset a rats
        rat_params = rat_paramsets[rat_name] # conditional assignment

        if not len((OtherTable & key).fetch("cluster_id")[0]): # paragraph breaks separate chunks conceptually
            logger.info(f"No spikes for {key}") # log level can be adjusted at run

        insertion_list = [] # We're gonna insert something
        for file_index, file_n in rat_params.items():
            insertion_list.append(
                self._get_cluster_times(key, file_n - 1, file_index) # there it is, clusters
            )
        self.insert(insertion_list) # separate inserts to happen all at once
```

## Comments, Type hints and docstrings

It's tempting to leave comments in code, but they can become outdated and
confusing. Instead try Atomizing and using Type hints and docstrings.

Type hints are not enforced, but make it much easier to tell the design intent
when reread. Docstrings are similarly optional, but make it easy to get prompts
without looking at the code again via `help(myfunc)`

### Type hints

```python
def get_data_interface(
    nwbfile: pynwb.Nwbfile,
    data_interface_name: Union[str, list],  # one or the other
    other_arg: Dict[str, Dict[str, dj.FreeTable]] = None,  # show embedding
) -> NWBDataInterface:  # What it returns. `None` if no return
    pass
```

### Docstrings

- Spyglass uses the NumPy docstring style, as opposed to Google.
- These are rendered in the
    [API documentation](https://lorenfranklab.github.io/spyglass/latest/api/utils/nwb_helper_fn/#src.spyglass.utils.nwb_helper_fn.get_data_interface)

```python
def get_data_interface(*args, **kwargs):
    """One-line description.

    Additional notes or further description in case the one line above is
    not enough.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        Description of the arg. e.g., The NWB file object to search in.
    data_interface_name : Union[str, list]
        More here.
    data_interface_class : Dict[str, Dict[str, dj.FreeTable]], optional
        more here

    Warns
    -----
    LoggerWarning
        Why warn.

    Raises
    ------
    ValueError
        Why it would hit this error.

    Returns
    -------
    data_interface : NWBDataInterface

    Example
    -------
    > data_interface = get_data_interface(mynwb, "interface_name")
    """
    pass
```

## Error detection with linting

- Packages like `ruff` can show you bad code 'smells' while you write and fix
    some for you.
- PEP8, Flake8 and other standards will flag issues like ...
    - F401: Module imported but unused
    - E402: Module level import not at top of file
    - E713: Test for membership should be 'not in'
- `black` will fix a subset of Flake8 issues, but not all. `ruff` identifies or
    fixes these rules and [many others](https://docs.astral.sh/ruff/rules/).

## Automation

- `black`, `isort`, and `ruff` can be run on save in most IDEs by searching
    their extensions.
- `pre-commit` is a tool that can be used to run these checks before each
    commit, ensuring that all your code is formatted, as defined in a `yaml`
    file.

```yaml
default_stages: [commit, push]
exclude: (^.github/|^docs/site/|^images/)

repos:
  - repo: https://github.com/ambv/black
    rev: 24.1.1
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/codespell-project/codespell
    rev: v2.2.6
    hooks:
      - id: codespell
        args: [--toml, pyproject.toml]
        additional_dependencies:
          - tomli
```
