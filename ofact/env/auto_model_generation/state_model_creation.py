from __future__ import annotations

from ofact.twin.state_model.model import StateModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ofact.twin.state_model.basic_elements import DigitalTwinObject
    from ofact.twin.state_model.entities import EntityType, StationaryResource, PassiveMovingResource
    from ofact.twin.state_model.processes import Process, EntityTransformationNode
    from ofact.twin.state_model.sales import Order, Feature, FeatureCluster, Customer


class StateModelCreation:

    def get_state_model(self, entity_types, plant, parts, obstacles, stationary_resources, passive_moving_resources,
                           active_moving_resources, entity_transformation_nodes, processes, process_executions,
                           order_pool, customer_base, features, feature_clusters):
        """Create a new state model from the event log."""

        state_model = StateModel(entity_types=entity_types,
                                 plant=plant,
                                 parts=parts,
                                 obstacles=obstacles,
                                 stationary_resources=stationary_resources,
                                 passive_moving_resources=passive_moving_resources,
                                 active_moving_resources=active_moving_resources,
                                 entity_transformation_nodes=entity_transformation_nodes,
                                 processes=processes,
                                 process_executions=process_executions,
                                 order_pool=order_pool,
                                 customer_base=customer_base,
                                 features=features,
                                 feature_clusters=feature_clusters)

        return state_model

    def extend_state_model(self, state_model: StateModel,
                           entity_types, plant, parts, obstacles, stationary_resources, passive_moving_resources,
                           active_moving_resources, entity_transformation_nodes, processes, process_executions,
                           order_pool, customer_base, features, feature_clusters):
        """Extend an already existing state model from the event log.
        Differentiation between unique and non-unique objects depending on the object "type"/class.
        ToDo: match of parts, resources, ....!!!
        """

        state_model_matching_dict_sm = self._get_state_model_matching_dict(state_model.get_entity_types())
        state_model_matching_dict_new = self._get_state_model_matching_dict(entity_types)
        for sm_object_sm_id, sm_object_new in state_model_matching_dict_new.items():
            sm_object_new: EntityType
            if sm_object_sm_id not in state_model_matching_dict_sm:
                state_model.add_entity_type(sm_object_new)

        # Note: no new plant added

        # state_model_matching_dict_sm = self._get_state_model_matching_dict(state_model.get_parts())
        parts_list = []
        for single_parts_list in list(parts.values()):
            parts_list.extend(single_parts_list)
        state_model.add_parts(parts_list)  # ToDo: maybe matching required between sources
        # state_model_matching_dict_new = self._get_state_model_matching_dict(parts_list)
        # for sm_object_sm_id, sm_object_new in state_model_matching_dict_new.items():
        #     sm_object_new: Part
        #     if sm_object_sm_id not in state_model_matching_dict_sm:
        #         state_model.add_part(sm_object_new)

        # Note: no new obstacles added

        state_model_matching_dict_sm = self._get_state_model_matching_dict(state_model.get_stationary_resources())
        stationary_resources_list = []
        for single_stationary_resources_list in list(stationary_resources.values()):
            stationary_resources_list.extend(single_stationary_resources_list)
        state_model_matching_dict_new = self._get_state_model_matching_dict(stationary_resources_list)
        for sm_object_sm_id, sm_object_new in state_model_matching_dict_new.items():
            sm_object_new: StationaryResource
            if sm_object_sm_id not in state_model_matching_dict_sm:
                state_model.add_stationary_resource(sm_object_new)

        state_model_matching_dict_sm = self._get_state_model_matching_dict(state_model.get_passive_moving_resources())
        passive_moving_resources_list = []
        for single_passive_moving_resources_list in list(passive_moving_resources.values()):
            passive_moving_resources_list.extend(single_passive_moving_resources_list)
        state_model_matching_dict_new = self._get_state_model_matching_dict(passive_moving_resources_list)
        for sm_object_sm_id, sm_object_new in state_model_matching_dict_new.items():
            sm_object_new: PassiveMovingResource
            if sm_object_sm_id not in state_model_matching_dict_sm:
                state_model.add_passive_moving_resource(sm_object_new)

        state_model_matching_dict_sm = (
            self._get_state_model_matching_dict(state_model.get_entity_transformation_nodes()))
        state_model_matching_dict_new = self._get_state_model_matching_dict(entity_transformation_nodes)
        for sm_object_sm_id, sm_object_new in state_model_matching_dict_new.items():
            sm_object_new: EntityTransformationNode
            if sm_object_sm_id not in state_model_matching_dict_sm:
                state_model.add_entity_transformation_node(sm_object_new)

        state_model_matching_dict_sm = self._get_state_model_matching_dict(state_model.get_processes())
        processes_list = []
        for single_processes_list in list(processes.values()):
            processes_list.extend(single_processes_list)
        state_model_matching_dict_new = self._get_state_model_matching_dict(processes_list)
        for sm_object_sm_id, sm_object_new in state_model_matching_dict_new.items():
            sm_object_new: Process
            if sm_object_sm_id not in state_model_matching_dict_sm:
                state_model.add_process(sm_object_new)

        # state_model_matching_dict_sm = self._get_state_model_matching_dict(state_model.get_process_executions_list())
        state_model_matching_dict_new = self._get_state_model_matching_dict(process_executions)
        for process_execution in process_executions:
            state_model.add_process_execution(process_execution)
        # for sm_object_sm_id, sm_object_new in state_model_matching_dict_new.items():
        #     sm_object_new: ProcessExecution
        #     if sm_object_sm_id not in state_model_matching_dict_sm:
        #         state_model.add_process_execution(sm_object_new)

        state_model_matching_dict_sm = self._get_state_model_matching_dict(state_model.get_orders())
        state_model_matching_dict_new = self._get_state_model_matching_dict(order_pool)
        for sm_object_sm_id, sm_object_new in state_model_matching_dict_new.items():
            sm_object_new: Order
            if sm_object_sm_id not in state_model_matching_dict_sm:
                state_model.add_order(sm_object_new)

        state_model_matching_dict_sm = self._get_state_model_matching_dict(state_model.get_customers())
        state_model_matching_dict_new = self._get_state_model_matching_dict(customer_base)
        for sm_object_sm_id, sm_object_new in state_model_matching_dict_new.items():
            sm_object_new: Customer
            if sm_object_sm_id not in state_model_matching_dict_sm:
                state_model.add_customer(sm_object_new)

        state_model_matching_dict_sm = self._get_state_model_matching_dict(state_model.get_features())
        state_model_matching_dict_new = self._get_state_model_matching_dict(features)
        for sm_object_sm_id, sm_object_new in state_model_matching_dict_new.items():
            sm_object_new: Feature
            if sm_object_sm_id not in state_model_matching_dict_sm:
                state_model.add_feature(sm_object_new)
        # ToDo: Has the second model the same standard objects?

        state_model_matching_dict_sm = self._get_state_model_matching_dict(state_model.get_feature_clusters())
        feature_clusters_list = []
        for single_feature_clusters_list in list(feature_clusters.values()):
            feature_clusters_list.extend(single_feature_clusters_list)
        state_model_matching_dict_new = self._get_state_model_matching_dict(feature_clusters_list)
        for sm_object_sm_id, sm_object_new in state_model_matching_dict_new.items():
            sm_object_new: FeatureCluster
            if sm_object_sm_id not in state_model_matching_dict_sm:
                state_model.add_feature_cluster(sm_object_new)

        return state_model

    def _get_state_model_matching_dict(self, state_model_objects: list[DigitalTwinObject]):
        state_model_matching_dict = {state_model_object.get_static_model_id()[1:]: state_model_object
                                     for state_model_object in state_model_objects}
        return state_model_matching_dict
