# File: read_data.py
# Author: Ronil Pancholia
# Date: 3/20/19
# Time: 4:43 PM

import numpy as np

from config import TIME_WINDOW, FREQUENCY, TRAIN_OVERLAP, TEST_OVERLAP


def create_buckets(data):
    step = int((FREQUENCY * TIME_WINDOW) / 2)
    size = FREQUENCY * TIME_WINDOW
    # data = data[-len(data) % (size):]
    buckets = [data[i: i + size] for i in range(0, len(data), step)]
    # buckets = np.array_split(data, len(data)/(FREQUENCY * TIME_WINDOW - 1))
    count = 0
    for i, b in enumerate(buckets):
        if (len(b) != size):
            count += 1

    buckets = buckets[:-count]
    return buckets


def compute_labels(buckets):
    labels = []
    for bucket in buckets:
        zeros = 0
        for label in bucket:
            if label == 0:
                zeros += 1

        if zeros > 0.5 * len(bucket):
            labels.append(0)
        else:
            labels.append(1)

    return np.asarray(labels).astype(np.long)


def read_session_data(files, multi_value=False, is_label=False):
    session_data = []
    for file_name in files:
        with open(file_name, "r") as f:
            for line in f:
                if multi_value:
                    values = [float(x) for x in line.strip().rsplit("  ")]
                else:
                    values = float(line.strip())

                session_data.append(values)

    buckets = create_buckets(session_data)

    buckets = [np.asarray(x) for x in buckets]

    if is_label:
        buckets = compute_labels(buckets)

    return np.asarray(buckets)
