# File: read_data.py
# Author: Ronil Pancholia
# Date: 3/20/19
# Time: 4:43 PM

import os
import numpy as np

def read_session_data(files, multi_value = False):
    session_data = []
    for file_name in files:
        with open(file_name, "r") as f:
            for line in f:
                if multi_value:
                    values = [float(x) for x in line.strip().rsplit("  ")]
                else:
                    values = float(line.strip())

                session_data.append(values)

    if multi_value:
        session_data = [np.asarray(x) for x in session_data]

    return np.asarray(session_data)