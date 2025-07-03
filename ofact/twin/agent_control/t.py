import pandas as pd
import numpy as np
import time
from datetime import datetime


class Resource:
    def __init__(self, name):
        self.name = name

class ProcessExecution:
    def __init__(self, data):
        self.__dict__.update(data)

values = ["a", 1000,  datetime(2023, 1, 1), [1, 2, 3], "a", 1000,  datetime(2023, 1, 1), [1, 2, 3]]

# Beispiel-Daten erstellen
resource = Resource("Process1")
new_process_execution = ProcessExecution({f'value {i}': value
                                          for i, value in enumerate(values)})

# Dictionary zur Speicherung der DataFrames
process_execution_dic = {}

# Geschwindigkeitstest für die DataFrame-Operationen
start_time_df = time.time()

for i in range(1000):
    if resource.name not in process_execution_dic.keys():
        process_execution_dic[resource.name] = pd.DataFrame([new_process_execution.__dict__])
    else:
        process_execution_dic[resource.name] = pd.concat(
            [process_execution_dic[resource.name],
             pd.DataFrame([new_process_execution.__dict__])], ignore_index=True)

end_time_df = time.time()

values = [datetime(2023, 1, 1), datetime(2023, 1, 1), 1000]
new_process_execution = ProcessExecution({f'value {i}': value
                                          for i, value in enumerate(values)})
# Geschwindigkeitstest für das Record Array
record_array = np.zeros(0, dtype=[('value 0', 'datetime64[ns]'), ('value 1', 'datetime64[ns]'), ('value 2', 'i4')])

start_time_rec = time.time()

for i in range(1000):
    new_row = np.array([(new_process_execution.__dict__['value 0'],
                         new_process_execution.__dict__['value 1'],
                         new_process_execution.__dict__['value 2'])], dtype=record_array.dtype)
    record_array = np.append(record_array, new_row)

end_time_rec = time.time()

print(f"Zeit für DataFrame (1000 Iterationen): {end_time_df - start_time_df:.6f} Sekunden")
print(f"Zeit für Record Array (1000 Iterationen): {end_time_rec - start_time_rec:.6f} Sekunden")
