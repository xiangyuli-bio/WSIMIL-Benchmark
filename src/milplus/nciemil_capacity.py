"""Native highK/lowK capacity guards, without changing any selected patches."""


def native_selection_minimum(k):
    if k not in {128, 512}:
        raise ValueError('NcIEMIL native selection K must be128 or512')
    return 2 * k


def validate_selection_capacity(records, k):
    minimum = native_selection_minimum(k)
    if any(row['count'] < minimum for row in records.values()):
        raise ValueError('NcIEMIL every original bag must have at least2K real distinct patches; no padding or duplicated instances')
