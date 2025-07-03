from datetime import timedelta
from typing import Optional

import numpy as np

from ofact.env.simulation.event_discrete import EventDiscreteSimulation


class PartUnavailabilityEnv(EventDiscreteSimulation):

    def __init__(self, change_handler, start_time):
        raise Exception("Update needed")
        super(PartUnavailabilityEnv, self).__init__(change_handler=change_handler, start_time=start_time)

        self.storage_reservation_warehouse = None

    def set_storage_reservation(self, storage_reservation):
        self.storage_reservation_warehouse = list(storage_reservation.values())[0]

    def execute_process_execution(self, process_execution,
                                  notification_duration_before_completion: Optional[timedelta] = None):
        if process_execution.get_process_name() == "Material part loading":

            part_availabilities: list[np.datetime64] = \
                self.storage_reservation_warehouse.check_part_availability(process_execution.get_parts())
            if part_availabilities:
                part_availability = max(part_availabilities)

                if process_execution.executed_start_time < part_availability.item():
                    shift_time = part_availability.item() - process_execution.executed_start_time

                    self.process_executions_shifted[process_execution] = shift_time

        super().execute_process_execution(process_execution)
