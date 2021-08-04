import numpy as np
import pandas as pd

def find_closest(closest_to, list_to_search_in, direction="both", strictly=False, return_index=False):
    """Find the closest number in the array from a given number x.

    Parameters
    ----------
    closest_to : float
        The target number(s) to find the closest of.
    list_to_search_in : list
        The list of values to look in.
    direction : str
        "both" for smaller or greater, "greater" for only greater numbers and "smaller" for the closest smaller.
    strictly : bool
        False for stricly superior or inferior or True for including equal.
    return_index : bool
        If True, will return the index of the closest value in the list.

    Returns
    ----------
    closest : int
        The closest number in the array.

    """

    # Transform to arrays
    closest_to = as_vector(closest_to)
    list_to_search_in = pd.Series(as_vector(list_to_search_in))

    out = [_find_closest(i, list_to_search_in, direction, strictly, return_index) for i in closest_to]

    if len(out) == 1:
        return out[0]
    else:
        return np.array(out)


# =============================================================================
# Internal
# =============================================================================
def _find_closest(closest_to, list_to_search_in, direction="both", strictly=False, return_index=False):

    try:
        index, closest = _find_closest_single_pandas(closest_to, list_to_search_in, direction, strictly)
    except ValueError:
        index, closest = np.nan, np.nan

    if return_index is True:
        return index
    else:
        return closest


# =============================================================================
# Methods
# =============================================================================


def _findclosest_base(x, vals, direction="both", strictly=False):
    if direction == "both":
        closest = min(vals, key=lambda y: np.abs(y - x))
    if direction == "smaller":
        if strictly is True:
            closest = max(y for y in vals if y < x)
        else:
            closest = max(y for y in vals if y <= x)
    if direction == "greater":
        if strictly is True:
            closest = min(filter(lambda y: y > x, vals))
        else:
            closest = min(filter(lambda y: y >= x, vals))

    return closest


def _find_closest_single_pandas(x, vals, direction="both", strictly=False):

    if direction in ["both", "all"]:
        index = (np.abs(vals - x)).idxmin()

    if direction in ["smaller", "below"]:
        if strictly is True:
            index = (np.abs(vals[vals < x] - x)).idxmin()
        else:
            index = (np.abs(vals[vals <= x] - x)).idxmin()

    if direction in ["greater", "above"]:
        if strictly is True:
            index = (vals[vals > x] - x).idxmin()
        else:
            index = (vals[vals >= x] - x).idxmin()

    closest = vals[index]

    return index, closest



def listify(**kwargs):
    """Transforms arguments into lists of the same length.
    """
    args = kwargs
    maxi = 1

    # Find max length
    for key, value in args.items():
        if isinstance(value, str) is False:
            try:
                if len(value) > maxi:
                    maxi = len(value)
            except TypeError:
                pass

    # Transform to lists
    for key, value in args.items():
        if isinstance(value, list):
            args[key] = _multiply_list(value, maxi)
        else:
            args[key] = _multiply_list([value], maxi)

    return args


def _multiply_list(lst, length):
    q, r = divmod(length, len(lst))
    return q * lst + lst[:r]


def as_vector(x):
    """Convert to vector.

    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        out = x.values
    elif isinstance(x, (str, float, int, np.int, np.intc, np.int8, np.int16, np.int32, np.int64)):
        out = np.array([x])
    else:
        out = np.array(x)

    if isinstance(out, np.ndarray):
        shape = out.shape
        if len(shape) == 1:
            pass
        elif len(shape) != 1 and len(shape) == 2 and shape[1] == 1:
            out = out[:, 0]
        else:
            raise ValueError(
                "EPK error: we expect the user to provide a "
                "vector, i.e., a one-dimensional array (such as a "
                "list of values). Current input of shape: " + str(shape)
            )

    return out

