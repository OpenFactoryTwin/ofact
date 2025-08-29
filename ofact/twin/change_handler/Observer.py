import numpy as np
import pandas as pd

from ofact.helpers import Singleton
from ofact.twin.state_model.entities import ActiveMovingResource, WorkStation, Warehouse, EntityType
from ofact.twin.state_model.model import StateModel


class Observer(metaclass=Singleton):

    def __init__(self):
        super(Observer, self).__init__()
        self.last_length = 0
        self.end_time = pd.Timestamp.min
        self.observation_period = pd.Timedelta(minutes=30)
        self.utilisation_df = pd.DataFrame(columns=['name', 'lead_time', 'utilisation'])
        self.process_execution_dic = {}
        self.process_execution_dic_type = [('Process Execution ID', 'i4'),  # ToDo: maybe higher values also possible
                                           ('Executed Start Time', 'datetime64[ns]'),
                                           ('Executed End Time', 'datetime64[ns]')]
        self.last_end_time = pd.Timestamp(1513393355.5,
                                          unit='s')  # Timestamp('2017-12-16 03:02:35.500000') only to initilize a compare time

    def set_observation(self, observation_period):
        self.observation_period = observation_period

    def set_digital_twin(self, digital_twin: StateModel):
        self.digital_twin = digital_twin

    def get_utilisation(self):
        return self.utilisation_df.set_index('name')['utilisation']

    def get_process_execution_df(self, resource_name):
        """
        Converts the internal NumPy array for a resource to a Pandas DataFrame.
        """
        if resource_name not in self.process_execution_dic:
            return pd.DataFrame(columns=[name for name, _ in self.process_execution_dic_type])

        arr = self.process_execution_dic[resource_name]
        df = pd.DataFrame(arr)

        # ensure datetime format
        df['Executed Start Time'] = pd.to_datetime(df['Executed Start Time'])
        df['Executed End Time'] = pd.to_datetime(df['Executed End Time'])

        return df


    def update_kpi(self):
        """
        Function to calculate the Utilisation.
        """
        if len(self.digital_twin.process_executions) < self.last_length:
            self.last_length = 0
            self.last_end_time = pd.Timestamp(1513393355.5, unit='s')
            self.end_time = pd.Timestamp(1513393355.5, unit='s')
            self.utilisation_df = pd.DataFrame(columns=['name', 'lead_time', 'utilisation'])  # ToDo: change to numpy
            self.process_execution_dic = {}

        if len(self.digital_twin.process_executions) > self.last_length:
            self.update_process_execution_df()
            if self.last_end_time < self.end_time:
                self.update_utilisation()
        else:
            return

    def update_process_execution_df(self):
        new_pe_range = len(self.digital_twin.process_executions) - self.last_length
        new_executions = self.digital_twin.process_executions[-new_pe_range:]
        mask = (new_executions['Event Type'] == 'PLAN') | (new_executions['Event Type'] == b'PLAN')

        # Gefiltertes Array mit nur PLAN Einträgen
        plan_executions = new_executions[mask]

        # Dann die Process Execution Spalte extrahieren (ist Objekt-Datentyp)
        plan_process_executions = plan_executions['Process Execution']
        for new_process_execution in plan_process_executions:
            for resource in new_process_execution.get_resources():
                if isinstance(resource, (ActiveMovingResource, WorkStation, Warehouse)):
                    new_entry = np.array([
                        (
                            new_process_execution.identification,
                            np.datetime64(new_process_execution._executed_start_time),
                            np.datetime64(new_process_execution._executed_end_time)
                        )
                    ], dtype=self.process_execution_dic_type)

                    if resource.name not in self.process_execution_dic:
                        self.process_execution_dic[resource.name] = new_entry
                    else:
                        self.process_execution_dic[resource.name] = np.concatenate(
                            [self.process_execution_dic[resource.name], new_entry]
                        )

                    if new_process_execution.executed_end_time > self.end_time:
                        self.end_time = new_process_execution.executed_end_time

        self.last_length = len(self.digital_twin.process_executions)

    def update_utilisation(self):
        mask = (self.digital_twin.process_executions['Event Type'] == 'ACTUAL') | \
               (self.digital_twin.process_executions['Event Type'] == b'ACTUAL')

        # Gefilterte Endzeiten
        actual_end_times = self.digital_twin.process_executions['Executed End Time'][mask]

        # Maximalen Zeitstempel holen
        start_time = pd.to_datetime(actual_end_times.max())
        for resource, arr in self.process_execution_dic.items():
            # Filter: nur Werte mit Endzeit > start_time
            end_times = arr['Executed End Time']
            valid_mask = end_times > np.datetime64(start_time)
            filtered_arr = arr[valid_mask]

            # Truncate Startzeiten, wenn sie < start_time
            start_times = filtered_arr['Executed Start Time'].copy()
            start_times[start_times < np.datetime64(start_time)] = np.datetime64(start_time)

            # Duplikate entfernen basierend auf 'Process Execution ID'
            _, unique_indices = np.unique(filtered_arr['Process Execution ID'], return_index=True)
            unique_filtered = filtered_arr[unique_indices]

            # Lead Time berechnen
            lead_times = unique_filtered['Executed End Time'] - start_times[unique_indices]
            sum_lead_time = lead_times.sum()

            # In Sekunden oder Minuten je nach Beobachtungsperiode
            #lead_time_seconds = sum_lead_time / np.timedelta64(1, 's')

            utilisation = (sum_lead_time / self.observation_period)

            # Speichern oder aktualisieren
            self.utilisation_df = self.utilisation_df[self.utilisation_df['name'] != resource]
            self.utilisation_df = pd.concat([
                self.utilisation_df,
                pd.DataFrame([{
                    'name': resource,
                    'lead_time': pd.to_timedelta(sum_lead_time, unit='s'),
                    'utilisation': utilisation
                }])
            ], ignore_index=True)

        self.last_end_time = self.end_time

    def get_num_process(self,resources_name):
        """
        #for resource in resources:
        resource=resources[0]
        if type(resource) is EntityType:
            if 'Warehouse' in resource.name:
                workstation = self.digital_twin.stationary_resources[resource]
            elif resource.super_entity_type.name == 'work station':
                workstation = self.digital_twin.stationary_resources[resource]
        """
        if resources_name in self.process_execution_dic.keys():
            station_df=pd.DataFrame(self.process_execution_dic[resources_name])
            station_df = station_df.drop_duplicates(subset=['Process Execution ID'])
            num_order= len(station_df)
        else:
            num_order=0

        return num_order