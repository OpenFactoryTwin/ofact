import numpy as np
import pandas as pd
import asyncio

from click import DateTime
from pandas import Timedelta

from ofact.helpers import Singleton
from ofact.planning_services.scenario_analytics.scenario_handling.multi import update_kpi_dfs
from ofact.twin.state_model.entities import ActiveMovingResource, StationaryResource, WorkStation, Warehouse, EntityType
from ofact.twin.state_model.model import StateModel
from ofact.twin.state_model.processes import ProcessExecution


class Observer(metaclass=Singleton):

    def __init__(self):
        super(Observer, self).__init__()
        self.last_length = 0
        self.end_time = pd.Timestamp.min
        self.observation_period = Timedelta(minutes=30)
        self.utilisation_df = pd.DataFrame(columns=['name', 'lead_time','utilisation'])
        self.process_execution_dic = {}
        self.last_end_time=pd.Timestamp(1513393355.5, unit='s') #Timestamp('2017-12-16 03:02:35.500000') only to initilize a compare time

    def set_observation(self,observation_period):
        self.observation_period = observation_period

    def set_digital_twin(self, digital_twin: StateModel):
        self.digital_twin = digital_twin


    def get_utilisation(self):
        return self.utilisation_df.set_index('name')['utilisation']


    def update_kpi(self):
        """
        Function to calculate the Utilisation.
        """
        if len(self.digital_twin.process_executions) < self.last_length:
            self.last_length=0
            self.last_end_time=pd.Timestamp(1513393355.5, unit='s')
            self.end_time = pd.Timestamp(1513393355.5, unit='s')
            self.utilisation_df = pd.DataFrame(columns=['name', 'lead_time','utilisation'])
            self.process_execution_dic = {}

        if len(self.digital_twin.process_executions)> self.last_length:
            self.update_process_execution_df()
            if self.last_end_time<self.end_time:
                self.update_utilisation()
        else:
            return

    def update_process_execution_df(self):
        new_PE_range = len(self.digital_twin.process_executions) - self.last_length
        df = pd.DataFrame(self.digital_twin.process_executions[-new_PE_range:])
        df=df.loc[df['Event Type']=='PLAN']
        for new_process_execution in df['Process Execution']:
            for resource in new_process_execution.get_resources():
                if isinstance(resource, (ActiveMovingResource, WorkStation, Warehouse)):
                    if resource.name not in self.process_execution_dic.keys():
                        self.process_execution_dic[resource.name] = pd.DataFrame([new_process_execution.__dict__])
                    else:
                        self.process_execution_dic[resource.name] = pd.concat(
                            [self.process_execution_dic[resource.name],
                             pd.DataFrame([new_process_execution.__dict__])], ignore_index=True)
                    if new_process_execution.executed_end_time > self.end_time:
                        self.end_time = new_process_execution.executed_end_time
        self.last_length = len(self.digital_twin.process_executions)

    def update_utilisation(self):
        df = pd.DataFrame(self.digital_twin.process_executions)
        df=df.loc[df['Event Type']=='ACTUAL'].reset_index(drop=True)
        start_time=df['Executed End Time'].max()
        #observation_end = pd.Timestamp(start_time + self.observation_period)
        for resource in self.process_execution_dic.keys():
            # Stelle sicher, dass 'executed_end_time' das richtige Format hat
            self.process_execution_dic[resource]['_executed_end_time'] = pd.to_datetime(
                self.process_execution_dic[resource]['_executed_end_time']
            )
            # Filtere die Daten
            self.process_execution_dic[resource] = self.process_execution_dic[resource].loc[
                self.process_execution_dic[resource]['_executed_end_time'] > start_time
                ]
            # set start time to observation start time
            self.process_execution_dic[resource].loc[self.process_execution_dic[resource][
                    '_executed_start_time'] < start_time, '_executed_start_time'] = start_time

            resource_df = self.process_execution_dic[resource].copy()
            if 'identification' not in resource_df.columns:
                resource_df=resource_df.T
            resource_df = resource_df.drop_duplicates(subset=['identification'])
            resource_df['_executed_lead_time'] = resource_df['_executed_end_time'] - resource_df['_executed_start_time']
            sum_time = resource_df['_executed_lead_time'].sum()
            if self.utilisation_df.empty or not (self.utilisation_df["name"] == resource).any():
                # Wenn keine Zeile existiert, füge sie hinzu
                new_row = pd.DataFrame({"name": [resource], "lead_time": [sum_time]})
                self.utilisation_df = pd.concat([self.utilisation_df, new_row], ignore_index=True)
            else:
                # Wenn die Zeile existiert, aktualisiere den Wert
                self.utilisation_df.loc[self.utilisation_df["name"] == resource, "lead_time"] = sum_time

        self.utilisation_df['utilisation']=(self.utilisation_df['lead_time']/self.observation_period)*100
        self.last_end_time=self.end_time

    def get_workstation(self,resources : EntityType):
        for resource in resources:
            if type(resource) is EntityType:
                if 'Warehouse' in resource.name:
                    workstation = self.digital_twin.stationary_resources[resource]
                elif resource.super_entity_type.name == 'work station':
                    workstation = self.digital_twin.stationary_resources[resource]
        if workstation[0].name in self.process_execution_dic.keys():
            station_df=self.process_execution_dic[workstation[0].name]
            station_df = station_df.drop_duplicates(subset=['identification'])
            num_order= len(station_df)
        else:
            num_order=0
        return workstation[0], num_order