import numpy as np


def get_overlaps_periods(arr_a, arr_b, min_overlapped=None):
    if not arr_a.any() or not arr_b.any():
        return np.array([[]])

    # ensure that all np.arrays have the same data type
    if arr_a.dtype != "datetime64[ns]":
        arr_a = arr_a.astype('datetime64[ns]')
    if arr_b.dtype != "datetime64[ns]":
        arr_b = arr_b.astype('datetime64[ns]')

    if arr_b.size > arr_a.size:
        old_arr_a = arr_a
        arr_a = arr_b
        arr_b = old_arr_a

    # print(f"{datetime.now()} | [{'Preference Helper':35}] Overlaps Periods: ", arr_a, arr_b)

    arr_a = _cut_start_end(arr_a, arr_b)
    if arr_a.size == 0:
        return np.array([[]])

    arr_b = _cut_start_end(arr_b, arr_a)

    if not arr_a.any() or not arr_b.any():
        return np.array([[]])

    period_b_idx = 0
    new_b = True
    matching_periods = np.array([[]], dtype='datetime64[ns]')
    for period_a in arr_a:

        new_a = False
        while not new_a:
            if new_b:
                if period_b_idx >= arr_b.shape[0]:
                    break
                period_b = arr_b[period_b_idx]
                period_b_idx += 1

            if period_a[0] <= period_b[0] and period_a[1] == period_b[1]:
                matching_periods = _insert_period(matching_periods, period_b[0], period_a[1])
                new_a = True
                new_b = True

            elif period_a[0] >= period_b[0] and period_a[1] == period_b[1]:
                matching_periods = _insert_period(matching_periods, period_a[0], period_a[1])
                new_a = True
                new_b = True

            elif period_a[0] <= period_b[0] and period_a[1] < period_b[1]:
                matching_periods = _insert_period(matching_periods, period_b[0], period_a[1])
                new_a = True
                new_b = False

            elif period_a[0] > period_b[0] and period_a[1] < period_b[1]:
                matching_periods = _insert_period(matching_periods, period_a[0], period_a[1])
                new_a = True
                new_b = False

            elif period_a[0] > period_b[0] and period_a[1] > period_b[1]:
                matching_periods = _insert_period(matching_periods, period_a[0], period_b[1])
                new_a = False
                new_b = True

            elif period_a[0] <= period_b[0] and period_a[1] > period_b[1]:
                matching_periods = _insert_period(matching_periods, period_b[0], period_b[1])
                new_a = False
                new_b = True

    matching_periods = matching_periods.reshape(int(matching_periods.shape[0] / 2), 2)

    if min_overlapped is not None:
        min_overlapped = np.timedelta64(min_overlapped, "s")
        matching_periods = matching_periods[matching_periods[:, 1] - matching_periods[:, 0] >= min_overlapped]

    return matching_periods


def _cut_start_end(arr_a, arr_b):
    """throw away non-overlapping periods at the start and the end of the arrays"""
    arr_a = arr_a[~((arr_a[:, 0] < arr_b[0, 0]) & (arr_a[:, 1] < arr_b[0, 0]) |
                    (arr_a[:, 0] > arr_b[-1, 1]) & (arr_a[:, 1] > arr_b[-1, 1]))]

    return arr_a


def _insert_period(matching_periods, start, end):
    """Insert the new accepted time period to the matching time periods"""
    if matching_periods.size == 0:
        matching_periods = np.array([start, end], dtype='datetime64[s]')
    else:
        matching_periods = np.concatenate((matching_periods,
                                           np.array([start, end], dtype='datetime64[s]')),
                                          axis=0)

    return matching_periods
