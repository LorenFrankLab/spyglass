"""Config-tree and interactive-prompt helpers for model training.

Small, table-agnostic utilities used by the ``Model`` training path: a
recursive label renamer for (possibly nested) DLC skeleton structures, and a
default-aware user prompt.
"""


def _rename_leaves(obj, mapping: dict):
    """Recursively rename string leaves of a nested list using *mapping*.

    Used to canonicalize body-part names inside a DLC skeleton structure
    (which may be nested) while preserving its shape. Strings absent from
    *mapping* are left unchanged.
    """
    if isinstance(obj, str):
        return mapping.get(obj, obj)
    if isinstance(obj, (list, tuple)):
        return [_rename_leaves(item, mapping) for item in obj]
    return obj


def prompt_default(key: str, default, abort_value: str = "n") -> str:
    """Prompt the user for a value, returning the default on empty input.

    Parameters
    ----------
    key : str
        Label shown in the prompt.
    default : any
        Default value returned when the user presses Enter without input.
    abort_value : str, optional
        Input that triggers a RuntimeError ("Aborted by user"), by default "n".

    Returns
    -------
    str
        The user's response, or *default* cast to str when input is blank.

    Raises
    ------
    RuntimeError
        If the user enters *abort_value*.
    """
    response = input(f"{key} [{default}]: ").strip()
    if response == abort_value:
        raise RuntimeError("Aborted by user")
    return response if response else str(default)
