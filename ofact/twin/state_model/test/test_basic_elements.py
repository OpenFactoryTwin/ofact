from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np

import ofact.twin.state_model.basic_elements as be


# Test the dynamic attributes
def _instantiate_dynamic_attribute_change_tracking(attribute_type) -> be.DynamicAttributeChangeTracking:
    if attribute_type == "single_object":
        attribute_change_tracker_class = be.SingleObjectAttributeChangeTracker
        attribute_value = None

    elif attribute_type == "list":
        attribute_change_tracker_class = be.ListAttributeChangeTracker
        attribute_value = []

    else:
        raise Exception("Invalid attribute type")

    dynamic_attribute_change_tracking = be.DynamicAttributeChangeTracking(attribute_change_tracker_class,
                                                                          current_time=None,
                                                                          attribute_value=attribute_value,
                                                                          process_execution=None)
    return dynamic_attribute_change_tracking


def _get_dynamic_attribute_change_tracking_single(changes_to_add, current_time):
    dynamic_attribute_change_tracking = _instantiate_dynamic_attribute_change_tracking("single_object")

    class ProcessExecution:
        pass

    process_execution = ProcessExecution()
    attribute_values = np.arange(changes_to_add)

    for i in range(changes_to_add):
        current_time += timedelta(seconds=60)
        attribute_value = attribute_values[i]
        dynamic_attribute_change_tracking.add_change(current_time=current_time, attribute_value=attribute_value,
                                                     process_execution_plan=None,
                                                     process_execution=process_execution,
                                                     sequence_already_ensured=True)

    return dynamic_attribute_change_tracking


def _get_dynamic_attribute_change_tracking_list(changes_to_add, current_time):
    dynamic_attribute_change_tracking = _instantiate_dynamic_attribute_change_tracking("list")

    class ProcessExecution:
        pass

    process_execution = ProcessExecution()
    attribute_values = np.arange(changes_to_add)
    attributes_added = []
    for i in range(changes_to_add):
        current_time += timedelta(seconds=60)
        attribute_value = attribute_values[i]
        attributes_added.append(attribute_value)
        dynamic_attribute_change_tracking.add_change(current_time=current_time,
                                                     attribute_value=attribute_value,
                                                     process_execution_plan=None,
                                                     change_type="ADD",
                                                     process_execution=process_execution,
                                                     sequence_already_ensured=True)

    return dynamic_attribute_change_tracking, attributes_added

changes_to_add = 20000
current_time = datetime.now()
dynamic_attribute_change_tracking_single = _get_dynamic_attribute_change_tracking_single(changes_to_add, current_time)
dynamic_attribute_change_tracking_list, attributes_added = (
    _get_dynamic_attribute_change_tracking_list(changes_to_add, current_time))


def test_add_change_attribute_change_tracking_single_object(mocker):
    assert dynamic_attribute_change_tracking_single.get_change_history_length() == changes_to_add


def test_add_change_attribute_change_tracking_list(mocker):


    assert dynamic_attribute_change_tracking_list.get_change_history_length() == changes_to_add


def test_get_change_array_change_attribute_change_tracking_single_object(mocker):

    dynamic_attribute_change_tracking_single.get_change_array()


def test_get_change_array_change_attribute_change_tracking_list(mocker):

    dynamic_attribute_change_tracking_list.get_change_array()


def test_get_changes_change_attribute_change_tracking_single_object(mocker):

    start_time = deepcopy(current_time)
    dynamic_attribute_change_tracking = dynamic_attribute_change_tracking_single

    changes_to_transfer = max(changes_to_add % (dynamic_attribute_change_tracking.recent_changes_max_memory -
                                                dynamic_attribute_change_tracking.recent_changes_min_memory),
                              dynamic_attribute_change_tracking.recent_changes_min_memory)

    # time from beginning
    dynamic_attribute_change_tracking.get_changes(start_time, start_time + timedelta(seconds=3600))
    # time from end
    dynamic_attribute_change_tracking.get_changes(current_time - timedelta(seconds=3600), current_time)
    # both recent and distant past
    dynamic_attribute_change_tracking.get_changes(current_time - timedelta(seconds=(changes_to_transfer + 30) * 60),
                                                  current_time - timedelta(seconds=(changes_to_transfer - 30) * 60))


def test_get_changes_array_change_attribute_change_tracking_list(mocker):

    start_time = deepcopy(current_time)
    dynamic_attribute_change_tracking = dynamic_attribute_change_tracking_list

    changes_to_transfer = max(changes_to_add % (dynamic_attribute_change_tracking.recent_changes_max_memory -
                                                dynamic_attribute_change_tracking.recent_changes_min_memory),
                              dynamic_attribute_change_tracking.recent_changes_min_memory)

    # time from beginning
    dynamic_attribute_change_tracking.get_changes(start_time, start_time + timedelta(seconds=3600))
    # time from end
    dynamic_attribute_change_tracking.get_changes(current_time - timedelta(seconds=3600), current_time)
    # both recent and distant past
    dynamic_attribute_change_tracking.get_changes(current_time - timedelta(seconds=(changes_to_transfer + 30) * 60),
                                                  current_time - timedelta(seconds=(changes_to_transfer - 30) * 60))


def test_get_version_change_attribute_change_tracking_single_object(mocker):

    start_time = deepcopy(current_time)
    dynamic_attribute_change_tracking = dynamic_attribute_change_tracking_single

    dynamic_attribute_change_tracking.get_version(start_time)


def test_get_version_change_attribute_change_tracking_list(mocker):
    start_time = deepcopy(current_time)
    dynamic_attribute_change_tracking = dynamic_attribute_change_tracking_list

    dynamic_attribute_change_tracking.get_version(start_time)
