from __future__ import annotations

from typing import TYPE_CHECKING

from ofact.twin.agent_control.behaviours.planning.part import PartRequest

if TYPE_CHECKING:
    from ofact.twin.state_model.entities import StationaryResource


class LimitedPartRequest(PartRequest):

    def _derive_process_requirements(self, requested_entity_types, locations_of_demand, order):
        """Derive transport of the parts needed"""

        # Loading - needed for all entities if not in the same resource
        # Transport only for different position of storage and location of demand

        if locations_of_demand is None:
            loading_needed = True
        else:
            loading_needed = False

        if isinstance(locations_of_demand, list):
            if len(locations_of_demand) == 1:
                location_of_demand = locations_of_demand[0]
            else:
                raise NotImplementedError
        else:
            location_of_demand = locations_of_demand

        lead_times = {}

        loading_demands: dict[StationaryResource, list[dict]] = {}
        transport_demands: dict[StationaryResource, list[dict]] = {}
        for resource in self.agent._resources:
            if resource.identification == location_of_demand.identification:
                continue

            for part_entity_type, amount in requested_entity_types:

                if loading_needed:
                    raise NotImplementedError

                # calculate the time needed between the unloading to the locations of demand and the loading process to
                # a transport resource (here the part should be available)
                # dependent on resource and entity_type
                # ToDo: calculate the lead_time (use heuristics) - max/ mean
                if loading_demands:
                    raise NotImplementedError
                if transport_demands:
                    for resource_, transport_demand_d_lst in transport_demands.items():
                        for transport_demand_d in transport_demand_d_lst:
                            transport_demand_d: dict
                            lead_time_part = sum(transport_demand_d["lead_times"])
                            lead_times[(resource, part_entity_type)] = lead_time_part

        process_executions_material_provision = loading_demands | transport_demands

        return process_executions_material_provision, lead_times
