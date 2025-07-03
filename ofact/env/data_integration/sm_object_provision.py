from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Optional, Union, Dict

from ofact.env.model_administration.helper import get_attr_value
from ofact.env.model_administration.sm_object_handling import abbreviations
from ofact.helpers import convert_to_datetime
from ofact.twin.state_model.entities import Entity
from ofact.twin.state_model.processes import ProcessExecution
from ofact.twin.state_model.basic_elements import DomainSpecificAttributes, ProcessExecutionTypes
import numpy as np

if TYPE_CHECKING:
    from ofact.twin.state_model.basic_elements import DigitalTwinObject


def set_value(object_: dict | object, attr: str, value):
    """Set a value to the attr of the object_. The object_ can be a dict or an DT-object"""
    attr_value_object = get_attr_value(object_, attr)

    attr_value_object = copy(attr_value_object)
    value_type = type(value) if value == value else np.nan
    operation_ = operation_mapper(attr_value_object, value, value_type)
    if attr == "delivery_date_actual" and value == value:
        print("delivery_date_actual", value)
    attr_value_object = operation_(attr_value_object, value)

    if isinstance(object_, dict):
        object_[attr] = attr_value_object

    else:
        if isinstance(object_, ProcessExecution):
            completely_filled, _ = object_.completely_filled()
            if completely_filled:  # if already filled, maybe ToDo: an update needed
                return object_
            try:
                setattr(object_, attr, attr_value_object)
            except:
                pass
        else:
            setattr(object_, attr, attr_value_object)

    return object_


memory = {}


def memoize_operation_mapper_string(operation_mapper_string):
    def inner(attr_value_object, value_type):
        if (attr_value_object, value_type) not in memory:
            memory[(attr_value_object, value_type)] = operation_mapper_string(attr_value_object, value_type)
        return memory[(attr_value_object, value_type)]

    return inner


def operation_mapper(attr_value_object, value, value_type):
    if isinstance(attr_value_object, str):
        operation_ = operation_mapper_string(attr_value_object, value_type)

    elif isinstance(attr_value_object, list):
        operation_ = operation_mapper_list(attr_value_object, value, value_type)

    elif isinstance(attr_value_object, dict):
        operation_ = operation_mapper_dict(attr_value_object, value_type)

    elif isinstance(attr_value_object, DomainSpecificAttributes):
        operation_ = do_nothing_value

    else:
        operation_ = do_nothing_value

    return operation_


@memoize_operation_mapper_string
def operation_mapper_string(attr_value_object, value_type):
    list_and_not_dict = [elem
                         for elem in attr_value_object.split(" | ")
                         if "list" in elem and not "dict" in elem]  # [0: (4 if len(elem) >= 4 else len(elem))]
    if list_and_not_dict:
        if value_type is None or value_type != value_type:
            operation_ = to_none  # would be overwritten at instantiation

        elif value_type == list:
            operation_ = do_nothing_value

        else:
            operation_ = to_list

    elif "dict" in attr_value_object:  # [0: (4 if len(attr_value_object) >= 4 else len(attr_value_object))]
        operation_ = do_nothing_value

    elif "datetime" in attr_value_object and value_type == str:
        operation_ = str_to_datetime

    else:
        operation_ = do_nothing_value

    return operation_


def operation_mapper_list(attr_value_object, value, value_type):
    if value and value_type == value_type:
        if len(attr_value_object) > 0:
            # check the list if the list contains tuples
            if isinstance(attr_value_object[0], tuple):
                operation_ = extend_tuple_by_list

            else:
                operation_ = append_to_list

        else:
            operation_ = append_to_list

    else:
        operation_ = do_nothing

    return operation_


def operation_mapper_dict(attr_value_object, value_type):
    operation_ = update_dict

    return operation_


def to_none(attr_value_object, value_):
    return None


def str_to_datetime(attr_value_object, value_):
    try:
        datetime_object = convert_to_datetime(value_)
    except:
        print
    attr_value_object = datetime_object

    return attr_value_object


def to_list(attr_value_object, value_):
    if value_ is not None:
        return [value_]
    return []


def to_empty_list(attr_value_object, value_):
    return []


def append_to_list(attr_value_object, value_):
    if not isinstance(value_, list):
        if value_ is not None:
            attr_value_object.append(value_)
    else:
        attr_value_object.extend([elem for elem in value_ if elem is not None])
    return attr_value_object


def extend_tuple_by_list(attr_value_object, value_):
    value = get_tuple_list(value_)

    attr_value_object.extend(value)
    return attr_value_object


def update_dict(attr_value_object, value_):
    attr_value_object.update(value_)
    return attr_value_object


def do_nothing_value(attr_value_object, value_):
    return value_


def do_nothing(attr_value_object, value):
    return attr_value_object


def get_tuple_list(elem_):
    if not isinstance(elem_, list):
        tuple_list = [(elem_,)]
        return tuple_list

    if not isinstance(elem_[0], tuple):
        tuple_list = [(single_elem,) for single_elem in elem_]
    else:
        tuple_list = elem_

    return tuple_list


class StateModelObjectProvider:

    def __init__(self, state_model, object_cache, change_handler):
        self._state_model = state_model
        self._object_cache = object_cache
        self._change_handler = change_handler

    def get_plant(self, name="", name_space="static_model"):
        return self._get_state_model_object(value=name, class_name="Plant", name_space=name_space)

    def get_entity_type(self, name, name_space="static_model"):
        return self._get_state_model_object(value=name, class_name="EntityType", name_space=name_space)

    def get_stationary_resource(self, name="", entity_type=None, name_space="static_model"):
        state_model_object = self._get_state_model_object(value=name, class_name="StationaryResource",
                                                          name_space=name_space)
        if isinstance(state_model_object, dict):
            state_model_object_new = None
            for class_name in ["WorkStation", "Warehouse", "Storage", "ConveyorBelt", "NonStationaryResource"]:
                match class_name:
                    case "WorkStation":
                        state_model_object_new = self._get_state_model_object(value=name, class_name="WorkStation",
                                                          name_space=name_space)
                    case "Warehouse":
                        state_model_object_new = self._get_state_model_object(value=name, class_name="Warehouse",
                                                          name_space=name_space)
                    case "Storage":
                        state_model_object_new = self._get_state_model_object(value=name, class_name="Storage",
                                                          name_space=name_space)
                    case "ConveyorBelt":
                        state_model_object_new = self._get_state_model_object(value=name, class_name="ConveyorBelt",
                                                          name_space=name_space)
                    case "NonStationaryResource":
                        state_model_object_new = self._get_state_model_object(value=name, class_name="NonStationaryResource",
                                                          name_space=name_space)  # ToDo: quick_run: should be changed

                if not isinstance(state_model_object_new, dict):
                    state_model_object = state_model_object_new
                    break

        if not isinstance(state_model_object, dict):
            return state_model_object

        name = "_".join(name.split("_")[:-1])

        if entity_type is None:
            entity_type = self.get_entity_type(name=name)
        set_value(object_=state_model_object, attr="entity_type", value=entity_type)
        set_value(object_=state_model_object, attr="process_execution_plan", value=self.get_pep(name))

        return state_model_object

    def get_pep(self, name, name_space="static_model"):
        return self._get_state_model_object(value=name, class_name="ProcessExecutionPlan",
                                                         name_space=name_space)

    def get_part(self, name="", entity_type=None, name_space="static_model", individual_attributes=None):

        state_model_object = self._get_state_model_object(value=name, class_name="Part",
                                                          name_space=name_space)

        if not isinstance(state_model_object, dict):
            return state_model_object
        state_model_object = self._get_state_model_object(value=name, class_name="Part",
                                                          name_space=name_space)

        name = "_".join(name.split("_")[:-1])

        if entity_type is None:
            entity_type = self.get_entity_type(name=name)
        set_value(object_=state_model_object, attr="entity_type", value=entity_type)

        if individual_attributes is not None:
            self._update_domain_specific_attributes(state_model_object, individual_attributes)

        return state_model_object

    def get_order(self, name="", name_space="static_model"):
        return self._get_state_model_object(value=name, class_name="Order", name_space=name_space)

    def get_customer(self, name="", name_space="static_model"):
        return self._get_state_model_object(value=name, class_name="Customer", name_space=name_space)

    def get_feature(self, name="", name_space="static_model"):
        return self._get_state_model_object(value=name, class_name="Feature", name_space=name_space)

    def get_process_execution_plan(self, execution_id, process, executed_start_time, executed_end_time,
                                   parts_involved, resources_used, main_resource, origin, destination,
                                   resulting_quality, order, source_application, name_space="static_model",
                                   individual_attributes=None):
        state_model_object = self._get_state_model_object(value=execution_id, class_name="ProcessExecution",
                                                          name_space=name_space)

        if not isinstance(state_model_object, dict):
            return state_model_object

        set_value(object_=state_model_object, attr="event_type", value=ProcessExecutionTypes.PLAN)
        set_value(object_=state_model_object, attr="process", value=process)
        set_value(object_=state_model_object, attr="executed_start_time", value=executed_start_time)
        set_value(object_=state_model_object, attr="executed_end_time", value=executed_end_time)
        set_value(object_=state_model_object, attr="parts_involved", value=parts_involved)
        set_value(object_=state_model_object, attr="resources_used", value=resources_used)
        set_value(object_=state_model_object, attr="main_resource", value=main_resource)
        set_value(object_=state_model_object, attr="origin", value=origin)
        set_value(object_=state_model_object, attr="destination", value=destination)
        set_value(object_=state_model_object, attr="resulting_quality", value=resulting_quality)
        set_value(object_=state_model_object, attr="order", value=order)
        set_value(object_=state_model_object, attr="source_application", value=source_application)
        if individual_attributes is not None:
            self._update_domain_specific_attributes(state_model_object, individual_attributes)

        return state_model_object

    def get_value_added_processes(self, name, name_space="static_model"):
        state_model_object = self._get_state_model_object(value=name,
                                                          class_name="ValueAddedProcess",
                                                          name_space=name_space)
        return state_model_object

    def get_processes(self, name, name_space="static_model"):
        state_model_object = self._get_state_model_object(value=name,
                                                          class_name="Process",
                                                          name_space=name_space)
        return state_model_object

    def _update_domain_specific_attributes(self, state_model_object, individual_attributes):
        if not isinstance(individual_attributes, dict):
            state_model_object.domain_specific_attributes.add_attributes(new_attributes=individual_attributes,
                                                                         all_attributes_allowed=True)
        else:
            if isinstance(state_model_object["domain_specific_attributes"], dict):
                state_model_object["domain_specific_attributes"].update(individual_attributes)
            else:
                state_model_object["domain_specific_attributes"] = individual_attributes

    def _get_state_model_object(self, value, class_name,
                                situated_in=None, name_space=None, new_possible=True) -> (
            [Optional[Union[Dict, DigitalTwinObject]], bool]):

        object_changed = False
        # 2. level: physical world cache
        current_object_from_cache = self._object_cache.get_object(name_space=name_space, id_=value,
                                                                  class_name=class_name)

        if current_object_from_cache is None:

            value = "_" + str(value)
            if not value.endswith(abbreviations[class_name]):
                value += abbreviations[class_name]
            value = value.replace(" ", "_")
            current_object_from_cache = self._object_cache.get_object(name_space=name_space, id_=value,
                                                                      class_name=class_name)

        if current_object_from_cache is not None:
            return current_object_from_cache  # , object_changed

        if class_name in ["Part", "PassiveMovingResource", "ActiveMovingResource"]:
            objects_already_planned = (
                self._object_cache.get_objects_already_planned(type_=class_name))

        else:
            objects_already_planned = []

        # 3. level: digital twin
        unique_value = value
        old_value = value

        current_object_from_dt = (
            self.get_object_by_external_identification_dt(name_space=name_space, external_id=value,
                                                          class_name=class_name,
                                                          situated_in=situated_in,
                                                          objects_already_planned=objects_already_planned))

        self._update_objects_domain_specific(current_object_from_dt)

        # ToDo
        # if domain_specific_static_refinements is not None:
        #     current_object_from_dt, object_changed = (
        #         self.refine_with_static_attributes(current_object_from_dt, domain_specific_static_refinements,
        #                                            class_name))
        # else:
        #     object_changed = False

        if current_object_from_dt is not None:
            if class_name == "Part":  # ToDo: Why only the part
                current_object_from_dt.external_identifications = copy({name_space: [unique_value]})
                self._object_cache.cache_object_already_planned(type_=class_name,
                                                                object_=current_object_from_dt)

            return current_object_from_dt  # , object_changed

        if class_name in ["Part", "PassiveMovingResource", "ActiveMovingResource"]:
            properties = self._get_object_properties(current_object_from_dt, objects_already_planned,
                                                     class_name, value,
                                                     name_space, situated_in, old_value)

            if properties:
                if "storage_places" in properties:
                    storages_to_store = properties["storage_places"]
                    # maybe integrate objects that are not a dict in the subsequent instantiation process
                    for _, storages in storages_to_store.items():
                        for storage in storages:
                            self._change_handler.add_object(storage)
            else:
                print("Class name:", class_name)

        else:
            properties = None

        # 4. level: state model class kwargs
        if new_possible is True:
            current_object_dict = self._get_state_model_class_dict(class_name,
                                                                   name_space, value)

            # ToDo
            # if domain_specific_static_refinements is not None:
            #     current_object_dict, object_changed = (
            #         self.refine_with_static_attributes(current_object_dict, domain_specific_static_refinements,
            #                                            data_entry_mapping.state_model_class))
            # else:
            object_changed = False
        else:
            return None  # , object_changed

        if properties is not None:
            for key, value in current_object_dict.items():
                if key in properties:
                    current_object_dict[key] = properties[key]

        if class_name == "PassiveMovingResource":
            del current_object_dict["physical_body"]

        if (class_name == "Part" and
                get_attr_value(current_object_dict, "name") == "str"):
            raise Exception("The object of type part is not completely filled:", current_object_dict, value)

        # logger.debug(f"Current object: {class_name}  {value}")
        return current_object_dict  #, object_changed

    def _get_state_model_class_dict(self, class_name, name_space, value):
        """Returns an unwritten class params dict from class 'class_name'"""
        current_object_dict = self._state_model.get_object_attributes(object_class_name=class_name)
        if not current_object_dict:
            raise NotImplementedError

        current_object_dict = set_value(object_=current_object_dict,
                                        attr="external_identifications",
                                        value={name_space: [value]})

        if "name" in current_object_dict:
            current_object_dict = set_value(object_=current_object_dict,
                                            attr="name",
                                            value=value)

        return current_object_dict

    def get_object_by_external_identification_dt(self, name_space, external_id, class_name,
                                                 situated_in: None | Entity = None,
                                                 objects_already_planned: None | list = None):
        """
        Get digital_twin object by external identification

        Parameters
        ----------
        situated_in: is used to determine only objects that are situated in the situated_in
        objects_already_planned: objects that are excluded in the determination
        because they are already selected in another context
        """

        digital_twin_objects = (
            self._state_model.get_object_by_external_identification(name_space=name_space, external_id=external_id,
                                                                    class_name=class_name,
                                                                    from_cache=True))

        if len(digital_twin_objects) == 0:
            return None

        if situated_in is not None:
            if hasattr(digital_twin_objects[0], "situated_in"):
                try:
                    digital_twin_objects = [digital_twin_object
                                            for digital_twin_object in digital_twin_objects
                                            if digital_twin_object.check_situated_in(entity=situated_in) is True]
                except:
                    raise Exception("Situated could not be checked: ..", situated_in)

        if objects_already_planned is not None:
            digital_twin_objects = list(set(digital_twin_objects).difference(set(objects_already_planned)))

        if len(digital_twin_objects) == 1:
            return digital_twin_objects[0]

        elif len(digital_twin_objects) > 1:
            return digital_twin_objects[0]

    def _update_objects_domain_specific(self, object_):
        """
        Used to update already existing objects
        Use Case: e.g. the abilities of the worker changes over time (this can be determined from the new data)
        """
        pass

    def _get_object_properties(self, current_object_from_dt, objects_already_planned, class_name, value, name_space,
                               situated_in, old_value):

        properties = None
        if current_object_from_dt is None:
            objects_already_planned = [object_
                                       for object_ in objects_already_planned
                                       if old_value in object_.external_identifications[name_space][0]]

        # maybe to special
        if objects_already_planned:
            if class_name == "Part":
                properties = self._get_part_properties(situated_in, objects_already_planned, name_space, old_value)

        else:
            if class_name == "PassiveMovingResource":
                properties = self._get_pmr_properties_without_reference(name_space=name_space,
                                                                        class_name=class_name,
                                                                        situated_in=situated_in,
                                                                        external_name=value)
            elif class_name == "ActiveMovingResource":
                properties = self._get_amr_properties_without_reference(name_space=name_space,
                                                                        class_name=class_name,
                                                                        situated_in=situated_in,
                                                                        external_name=value)

        return properties

    def _get_part_properties(self, situated_in, objects_already_planned, name_space, old_value):

        if situated_in is not None:
            objects_already_planned_situated_in = \
                [object_already_planned
                 for object_already_planned in objects_already_planned
                 if object_already_planned.check_situated_in(entity=situated_in) is True]

        else:
            objects_already_planned_situated_in = objects_already_planned

        part_properties = None
        if objects_already_planned_situated_in:
            for object_ in objects_already_planned_situated_in:
                if not [name
                        for name in object_.external_identifications[name_space]
                        if old_value == " ".join([elem for elem in name.split(" ")[1:2]])]:
                    continue
                object_already_planned_situated_in = object_
                part_properties = \
                    {"entity_type": object_already_planned_situated_in.entity_type,
                     "name": object_already_planned_situated_in.name,
                     "situated_in": object_already_planned_situated_in.situated_in}
                break

        return part_properties

    def _get_pmr_properties_without_reference(self, name_space, class_name, situated_in, external_name):
        reference_object, nsr_properties = (
            self._get_non_stationary_properties_without_reference(name_space, class_name, situated_in, external_name))

        # maybe differentiate the objects
        pmr_properties = (nsr_properties |
                          {"service_life": reference_object.service_life})

        return pmr_properties

    def _get_amr_properties_without_reference(self, name_space, class_name, situated_in, external_name):
        reference_object, nsr_properties = (
            self._get_non_stationary_properties_without_reference(name_space, class_name, situated_in, external_name))

        # maybe differentiate the objects
        amr_properties = (nsr_properties |
                          {"energy_level": reference_object.speed,
                           "speed": reference_object.speed,
                           "energy_consumption": reference_object.speed,
                           "energy_capacity": reference_object.speed})

        return amr_properties

    def _get_non_stationary_properties_without_reference(self, name_space, class_name, situated_in, external_name):
        available_objects = self._state_model.get_objects_by_class_name(class_name)

        reference_object = available_objects[0]

        # maybe differentiate the objects
        nsr_properties = {"entity_type": reference_object.entity_type,
                          "name": external_name,
                          "situated_in": reference_object.situated_in,
                          "process_execution_plan": reference_object.process_execution_plan.duplicate(),
                          "plant": reference_object.plant,
                          "costs_per_second": reference_object.costs_per_second,
                          "orientation": reference_object.orientation,
                          "storage_places":
                              reference_object._storage_places.duplicate_for_instantiation(without_situated_in=True),
                          "position": reference_object.get_position(),
                          "width": reference_object.get_width(),
                          "length": reference_object.get_length()}

        return reference_object, nsr_properties
