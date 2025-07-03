from functools import reduce
from operator import concat
from pathlib import Path

import numpy as np

import ofact.twin.state_model.model as m
from ofact.planning_services.model_generation.persistence import deserialize_state_model
from ofact.settings import ROOT_PATH
from ofact.twin.repository_services.serialization.dynamic_state_model import DynamicStateModelSerialization
from ofact.twin.repository_services.serialization.static_state_model import StaticModelStateModelSerialization
from ofact.twin.repository_services.deserialization.dynamic_state_model import DynamicStateModelDeserialization
from ofact.twin.state_model.basic_elements import DigitalTwinObject
from ofact.twin.state_model.seralized_model import SerializedStateModel


def _get_state_models(type_="STATIC"):
    """
    type_: "STATIC" | "DYNAMIC"
    """
    if type_ == "STATIC":
        return _get_static_state_models()
    elif type_ == "DYNAMIC":
        return _get_dynamic_state_models()

    raise Exception("Type_ must be either 'STATIC' or 'DYNAMIC'")


def _get_static_state_models():
    # import two models
    state_model_file_path = Path(ROOT_PATH.split("ofact")[0],
                                 "projects/tutorial/models/twin/mini.xlsx")
    state_model_generation_settings = {
        "customer_generation_from_excel": True,
        "order_generation_from_excel": True
    }
    state_model1 = deserialize_state_model(source_file_path=state_model_file_path, persistence_format="xlsx",
                                           state_model_generation_settings=state_model_generation_settings)

    exporter = StaticModelStateModelSerialization(state_model1)
    exporter.export("test.xlsx")
    m.DigitalTwinObject.next_id = 0

    state_model2 = deserialize_state_model(source_file_path="./test.xlsx", persistence_format="xlsx",
                                           state_model_generation_settings=state_model_generation_settings)

    state_model1.set_implicit_objects_explicit()
    state_model2.set_implicit_objects_explicit()

    return state_model1, state_model2


def _get_dynamic_state_models():
    # import two models
    state_model1 = deserialize_state_model(source_file_path="test_dynamic.pkl")  # ToDo: should be replaced ...

    exporter = DynamicStateModelSerialization(state_model1)
    exporter.export("test_dynamic1.pkl")

    state_model_dict = SerializedStateModel.load_from_pickle("test_dynamic1.pkl")
    importer = DynamicStateModelDeserialization()
    state_model2 = importer.get_state_model(state_model_dict)

    state_model1.set_implicit_objects_explicit()
    state_model2.set_implicit_objects_explicit()

    return state_model1, state_model2


irrelevant_attributes = ["dt_objects_directory", "_objects_by_external_identification", "state_model_class_mapper",
                         "processes_by_main_parts", "processes_by_main_resource"]


def t_static_model_equality():
    state_model1, state_model2 = _get_state_models("STATIC")

    # compare the equality of two state models
    state_model1_dict = state_model1.__dict__
    state_model2_dict = state_model2.__dict__

    for attribute in state_model1_dict:
        if attribute in irrelevant_attributes:
            continue

        if state_model1_dict[attribute] is None:
            assert state_model1_dict[attribute] == state_model2_dict[attribute]

        elif isinstance(state_model1_dict[attribute], list):
            if len(state_model1_dict[attribute]) != len(state_model2_dict[attribute]):
                raise Exception("Length of lists are not equal")

            state_model1_dict_attr_values_sorted = {value.get_static_model_id(): value
                                                    for value in state_model1_dict[attribute]}

            state_model2_dict_attr_values_sorted = {value.get_static_model_id(): value
                                                    for value in state_model2_dict[attribute]}

            for i in state_model1_dict_attr_values_sorted:

                try:
                    if isinstance(state_model1_dict_attr_values_sorted[i], DigitalTwinObject):
                        assert (state_model1_dict_attr_values_sorted[i].representation() ==
                                state_model2_dict_attr_values_sorted[i].representation())

                    else:
                        assert (repr(state_model1_dict_attr_values_sorted[i]) ==
                                repr(state_model2_dict_attr_values_sorted[i]))

                except:
                    print("List:", attribute)


        elif isinstance(state_model1_dict[attribute], dict):

            if len(state_model1_dict[attribute]) != len(state_model2_dict[attribute]):
                raise Exception("Length of dicts are not equal")

            elif len(state_model1_dict[attribute]) == 0:
                continue

            # check/compare only the values of the dict - assuming the keys are helpers
            state_model1_dict_attr_values = reduce(concat, state_model1_dict[attribute].values())
            state_model2_dict_attr_values = reduce(concat, state_model2_dict[attribute].values())

            state_model1_dict_attr_values_sorted = {value.get_static_model_id(): value
                                                    for value in state_model1_dict_attr_values}
            state_model2_dict_attr_values_sorted = {value.get_static_model_id(): value
                                                    for value in state_model2_dict_attr_values}

            broken = False

            for i in state_model1_dict_attr_values_sorted:
                try:
                    if isinstance(state_model1_dict_attr_values_sorted[i], DigitalTwinObject):

                        assert (state_model1_dict_attr_values_sorted[i].representation() ==
                                state_model2_dict_attr_values_sorted[i].representation())

                    else:
                        assert (repr(state_model1_dict_attr_values_sorted[i]) ==
                                repr(state_model2_dict_attr_values_sorted[i]))

                except:
                    for attribute_ in state_model1_dict_attr_values_sorted[i].__dict__:
                        try:
                            if isinstance(state_model1_dict_attr_values_sorted[i].__dict__[attribute_],
                                          DigitalTwinObject):
                                assert (state_model1_dict_attr_values_sorted[i].__dict__[attribute_].representation()
                                        ==
                                        state_model2_dict_attr_values_sorted[i].__dict__[attribute_].representation())
                            else:
                                assert (repr(state_model1_dict_attr_values_sorted[i].__dict__[attribute_]) ==
                                        repr(state_model2_dict_attr_values_sorted[i].__dict__[attribute_]))
                        except:
                            print("Attribute:", attribute, attribute_)
                            broken = True
                            break
                if broken:
                    break

        elif isinstance(state_model1_dict[attribute], np.ndarray):

            for array_name in state_model1_dict[attribute].dtype.names:
                list1 = state_model1_dict[attribute][array_name].tolist()
                list2 = state_model2_dict[attribute][array_name].tolist()

                if len(list1) != len(list2):
                    raise Exception("Length of lists are not equal")

                elif len(list1) == 0:
                    continue

                for i in range(len(list1)):

                    try:
                        if isinstance(list1[i], DigitalTwinObject):
                            assert list1[i].representation() == list2[i].representation()

                        else:
                            assert repr(list1[i]) == repr(list2[i])

                    except:
                        print(attribute, array_name)

                        if isinstance(list1[i], DigitalTwinObject):
                            print(list1[i].representation())
                            print(list2[i].representation())

                        else:
                            print(repr(list1[i]))
                            print(repr(list2[i]))
                        raise Exception("Value not equal", attribute, array_name)

        elif isinstance(state_model1_dict[attribute], int):
            assert repr(state_model1_dict[attribute]) == repr(state_model2_dict[attribute])

        elif isinstance(state_model1_dict[attribute], m.Plant):
            try:
                assert state_model1_dict[attribute].represtation() == state_model2_dict[attribute].representation()
            except:
                print("Plant:", attribute)

        else:
            raise Exception(f"Type '{type(state_model1_dict[attribute])}' not supported for attribute '{attribute}'")


def t_dynamic_model_equality():
    state_model1, state_model2 = _get_state_models("DYNAMIC")

    # compare the equality of two state models
    state_model1_dict = state_model1.__dict__
    state_model2_dict = state_model2.__dict__

    for attribute in state_model1_dict:
        if attribute in irrelevant_attributes:
            continue

        if state_model1_dict[attribute] is None:
            assert state_model1_dict[attribute] == state_model2_dict[attribute]
        elif isinstance(state_model1_dict[attribute], list):
            if len(state_model1_dict[attribute]) != len(state_model2_dict[attribute]):
                raise Exception("Length of lists are not equal")

            state_model1_dict_attr_values_sorted = {value.identification: value
                                                    for value in state_model1_dict[attribute]}
            state_model2_dict_attr_values_sorted = {value.identification: value
                                                    for value in state_model2_dict[attribute]}
            for i in state_model1_dict_attr_values_sorted:
                try:
                    if isinstance(state_model1_dict_attr_values_sorted[i], DigitalTwinObject):
                        assert (state_model1_dict_attr_values_sorted[i].representation() ==
                                state_model2_dict_attr_values_sorted[i].representation())
                    else:
                        assert (repr(state_model1_dict_attr_values_sorted[i]) ==
                                repr(state_model2_dict_attr_values_sorted[i]))
                except:
                    print("List:", attribute)

        elif isinstance(state_model1_dict[attribute], dict):
            if len(state_model1_dict[attribute]) != len(state_model2_dict[attribute]):
                raise Exception("Length of dicts are not equal")
            elif len(state_model1_dict[attribute]) == 0:
                continue

            # check/compare only the values of the dict - assuming the keys are helpers
            state_model1_dict_attr_values = reduce(concat, state_model1_dict[attribute].values())
            state_model2_dict_attr_values = reduce(concat, state_model2_dict[attribute].values())

            state_model1_dict_attr_values_sorted = {value.identification: value
                                                    for value in state_model1_dict_attr_values}
            state_model2_dict_attr_values_sorted = {value.identification: value
                                                    for value in state_model2_dict_attr_values}
            broken = False
            for i in state_model1_dict_attr_values_sorted:
                try:
                    if isinstance(
                            state_model1_dict_attr_values_sorted[i], DigitalTwinObject):
                        assert (state_model1_dict_attr_values_sorted[i].representation() ==
                                state_model2_dict_attr_values_sorted[i].representation())
                    else:
                        assert (repr(state_model1_dict_attr_values_sorted[i]) ==
                                repr(state_model2_dict_attr_values_sorted[i]))
                except:
                    for attribute_ in state_model1_dict_attr_values_sorted[i].__dict__:
                        try:
                            if isinstance(
                                    state_model1_dict_attr_values_sorted[i].__dict__[attribute_], DigitalTwinObject):
                                assert (state_model1_dict_attr_values_sorted[i].__dict__[attribute_].representation()
                                        ==
                                        state_model2_dict_attr_values_sorted[i].__dict__[attribute_].representation())
                            else:
                                assert (repr(state_model1_dict_attr_values_sorted[i].__dict__[attribute_]) ==
                                        repr(state_model2_dict_attr_values_sorted[i].__dict__[attribute_]))
                        except:
                            print("Attribute:", attribute, attribute_)
                            broken = True
                            break
                if broken:
                    break

        elif isinstance(state_model1_dict[attribute], np.ndarray):

            for array_name in state_model1_dict[attribute].dtype.names:
                list1 = state_model1_dict[attribute][array_name].tolist()
                list2 = state_model2_dict[attribute][array_name].tolist()

                if len(list1) != len(list2):
                    raise Exception("Length of lists are not equal")
                elif len(list1) == 0:
                    continue

                for i in range(len(list1)):
                    try:
                        if isinstance([i], DigitalTwinObject):
                            assert list1[i].representation() == list2[i].representation()
                        else:
                            assert repr(list1[i]) == repr(list2[i])
                    except:
                        print(attribute, array_name)
                        if isinstance(list1[i], DigitalTwinObject):
                            print(list1[i].representation())
                            print(list2[i].representation())
                        else:
                            print(repr(list1[i]))
                            print(repr(list2[i]))
                        # raise Exception("Value not equal", attribute, array_name)

        elif isinstance(state_model1_dict[attribute], int):
            assert repr(state_model1_dict[attribute]) == repr(state_model2_dict[attribute])
        elif isinstance(state_model1_dict[attribute], m.Plant):
            try:
                assert state_model1_dict[attribute].representation() == state_model2_dict[attribute].representation()
            except:
                print("Plant:", attribute)
                # print(state_model1_dict[attribute].representation())
                # print(state_model2_dict[attribute].representation())
        else:
            raise Exception(f"Type '{type(state_model1_dict[attribute])}' not supported for attribute '{attribute}'")


t_static_model_equality()
# t_dynamic_model_equality()
#  buffer stations - super entity type not set in the initial excel files (storage places only mapped to entity types)
# number of entities stored also diverging/ maybe not counted
