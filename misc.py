import numpy as np
import pandas as pd



def _events_find_label(events, event_labels=None, event_conditions=None, function_name="events_find"):
    # Get n events
    n = len(events["onset"])

    # Labels
    if event_labels is None:
        event_labels = (np.arange(n) + 1).astype(np.str)

    if len(list(set(event_labels))) != n:
        raise ValueError(
            "EPK error: "
            + function_name
            + "(): oops, it seems like the `event_labels` that you provided "
            + "are not unique (all different). Please provide "
            + str(n)
            + " distinct labels."
        )

    if len(event_labels) != n:
        raise ValueError(
            "EPK error: "
            + function_name
            + "(): oops, it seems like you provided "
            + str(n)
            + " `event_labels`, but "
            + str(n)
            + " events got detected :(. Check your event names or the event signal!"
        )

    events["label"] = event_labels

    # Condition
    if event_conditions is not None:
        if len(event_conditions) != n:
            raise ValueError(
                "EPK error: "
                + function_name
                + "(): oops, it seems like you provided "
                + str(n)
                + " `event_conditions`, but "
                + str(n)
                + " events got detected :(. Check your event conditions or the event signal!"
            )
        events["condition"] = event_conditions
    return events



def epochs_to_df(epochs):
    """Convert epochs to a DataFrame.

    Parameters
    ----------
    epochs : dict
        A dict containing one DataFrame per event/trial. Usually obtained via `epochs_create()`.


    Returns
    ----------
    DataFrame
        A DataFrame containing all epochs identifiable by the 'Label' column, which time axis
        is stored in the 'Time' column.

    """
    data = pd.concat(epochs)
    data["Time"] = data.index.get_level_values(1).values
    data = data.reset_index(drop=True)

    return data



def epochs_create(
    data,
    events=None,
    sampling_rate=1000,
    epochs_start=0,
    epochs_end=1,
    event_labels=None,
    event_conditions=None,
    baseline_correction=False,
):
    """Epoching a dataframe.

    Parameters
    ----------
    data : DataFrame
        A DataFrame containing the different signal(s) as different columns.
        If a vector of values is passed, it will be transformed in a DataFrame
        with a single 'Signal' column.
    events : list or ndarray or dict
        Events onset location. If a dict is passed (e.g., from ``events_find()``),
        will select only the 'onset' list. If an integer is passed,
        will use this number to create an evenly spaced list of events. If None,
        will chunk the signal into successive blocks of the set duration.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    epochs_start : int
        Epochs start relative to events_onsets (in seconds). The start can be negative to
        start epochs before a given event (to have a baseline for instance).
    epochs_end : int
        Epochs end relative to events_onsets (in seconds).
    event_labels : list
        A list containing unique event identifiers. If `None`, will use the event index number.
    event_conditions : list
        An optional list containing, for each event, for example the trial category, group or
        experimental conditions.
    baseline_correction : bool
        Defaults to False.


    Returns
    ----------
    dict
        A dict containing DataFrames for all epochs.

    """

    # Santize data input
    if isinstance(data, tuple):  # If a tuple of data and info is passed
        data = data[0]

    if isinstance(data, (list, np.ndarray, pd.Series)):
        data = pd.DataFrame({"Signal": list(data)})

    # Sanitize events input
    if events is None:
        max_duration = (np.max(epochs_end) - np.min(epochs_start)) * sampling_rate
        events = np.arange(0, len(data) - max_duration, max_duration)
    if isinstance(events, int):
        events = np.linspace(0, len(data), events + 2)[1:-1]
    if isinstance(events, dict) is False:
        events = _events_find_label({"onset": events}, event_labels=event_labels, event_conditions=event_conditions)

    event_onsets = list(events["onset"])
    event_labels = list(events["label"])
    if "condition" in events.keys():
        event_conditions = list(events["condition"])

    # Create epochs
    parameters = listify(
        onset=event_onsets, label=event_labels, condition=event_conditions, start=epochs_start, end=epochs_end
    )

    # Find the maximum numbers of samples in an epoch
    parameters["duration"] = np.array(parameters["end"]) - np.array(parameters["start"])
    epoch_max_duration = int(max((i * sampling_rate for i in parameters["duration"])))

    # Extend data by the max samples in epochs * NaN (to prevent non-complete data)
    length_buffer = epoch_max_duration
    buffer = pd.DataFrame(index=range(length_buffer), columns=data.columns)
    data = data.append(buffer, ignore_index=True, sort=False)
    data = buffer.append(data, ignore_index=True, sort=False)

    # Adjust the Onset of the events for the buffer
    parameters["onset"] = [i + length_buffer for i in parameters["onset"]]

    epochs = {}
    for i, label in enumerate(parameters["label"]):

        # Find indices
        start = parameters["onset"][i] + (parameters["start"][i] * sampling_rate)
        end = parameters["onset"][i] + (parameters["end"][i] * sampling_rate)

        # Slice dataframe
        epoch = data.iloc[int(start) : int(end)].copy()

        # Correct index
        epoch["Index"] = epoch.index.values - length_buffer
        epoch.index = np.linspace(
            start=parameters["start"][i], stop=parameters["end"][i], num=len(epoch), endpoint=True
        )

        if baseline_correction is True:
            baseline_end = 0 if epochs_start <= 0 else epochs_start
            epoch = epoch - epoch.loc[:baseline_end].mean()

        # Add additional
        epoch["Label"] = parameters["label"][i]
        if parameters["condition"][i] is not None:
            epoch["Condition"] = parameters["condition"][i]

        # Store
        epochs[label] = epoch

    return epochs




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


def _find_closest(closest_to, list_to_search_in, direction="both", strictly=False, return_index=False):

    try:
        index, closest = _find_closest_single_pandas(closest_to, list_to_search_in, direction, strictly)
    except ValueError:
        index, closest = np.nan, np.nan

    if return_index is True:
        return index
    else:
        return closest


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

