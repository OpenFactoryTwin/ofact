"""
#############################################################
This program and the accompanying materials are made available under the
terms of the Apache License, Version 2.0 which is available at
https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

SPDX-License-Identifier: Apache-2.0
#############################################################

Used to read the resource schedules as input (modeled in Excel files)
and update the schedule of the resources in the state model.
The schedule is an input parameter for the simulation and influences the availability of the resources.
If no schedule is provided, a resource is always available.
"""

# Imports 1: Standard Imports
from datetime import datetime, date, time, timedelta
from typing import TYPE_CHECKING, Dict, Union, Optional

# Imports 1: PIP Imports
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.state_model.probabilities import SingleValueDistribution

if TYPE_CHECKING:
    from ofact.twin.state_model.time import ProcessExecutionPlan


def minute_interval(start, end):  # not used until now
    reverse = False
    if start > end:
        start, end = end, start
        reverse = True

    hour_delta = end.hour - start.hour
    minute_delta = end.minute - start.minute
    second_delta = end.second - start.second

    delta = time(hour=hour_delta, minute=minute_delta, second=second_delta)

    return delta


def get_schedules_resource(resource_scheduling_path: str, start_time: Optional[datetime] = None):
    """
    Retrieves the schedules resource from the specified worker planning file.
    :param resource_scheduling_path: The path to the worker planning file.
    :returns: A dictionary mapping worker names to their available times.
    """

    settings_df = pd.read_excel(resource_scheduling_path, sheet_name="Settings", header=[0, 1])
    mode_df = settings_df["Modes"].dropna(axis=1)
    mode = mode_df.columns[0]

    match mode:
        case "Week":
            schedule_sheets = pd.read_excel(resource_scheduling_path, sheet_name=None)
            weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            resource_available_times_week = []
            for weekday in weekdays:
                weekday_schedule = schedule_sheets[weekday]
                resource_available_times = _get_single_resource_schedule(weekday_schedule, start_time)
                resource_available_times_week.append(resource_available_times)
            resource_available_times_list = resource_available_times_week

        case "Single":
            general_schedule_sheet = pd.read_excel(resource_scheduling_path,
                                                   sheet_name="General")
            resource_available_times = _get_single_resource_schedule(general_schedule_sheet, start_time)
            resource_available_times_list = [resource_available_times]

        case _:
            raise Exception(f"Mode {mode} not supported")

    return resource_available_times_list

def get_full_time_equivalents_resources(resource_scheduling_path: str) -> pd.DataFrame:
    settings_df = pd.read_excel(resource_scheduling_path, sheet_name="Settings", header=[0, 1])
    mode_df = settings_df["Modes"].dropna(axis=1)
    mode = mode_df.columns[0]

    match mode:
        case "Week":
            schedule_sheets = pd.read_excel(resource_scheduling_path, sheet_name=None)
            weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            resource_available_times_week = []
            for weekday in weekdays:
                weekday_schedule = schedule_sheets[weekday]
                resource_available_times = _get_full_time_equivalents_day_resources(weekday_schedule)
                resource_available_times["Time"] = [weekday + " " + str(i)
                                                  for i in resource_available_times["Time"]]
                resource_available_times["Index Time"] = resource_available_times["Time"]
                resource_available_times.set_index("Index Time", inplace=True)  # individual rows ..
                resource_available_times_week.append(resource_available_times)
            resource_available_times_df = pd.concat(resource_available_times_week,
                                                    axis=0)  # something else required

        case "Single":
            general_schedule_sheet = pd.read_excel(resource_scheduling_path,
                                                   sheet_name="General")
            resource_available_times_df = _get_full_time_equivalents_day_resources(general_schedule_sheet)

        case _:
            raise Exception(f"Mode {mode} not supported")

    return resource_available_times_df


def _get_full_time_equivalents_day_resources(schedule_df: pd.DataFrame):
    times_df = schedule_df[schedule_df.columns[3:]]  # 3 (fourth column) is schedule/xlsx file dependent
    times_df = pd.DataFrame(times_df.apply(lambda col: col.value_counts().get("x", 0)),
                            columns=["FTE (Schedule Resolution)"])
    times_df.index = pd.to_datetime([str(t)
                                     for t in times_df.index.to_list()],
                                    format="%H:%M:%S")

    fte_per_hour = times_df.groupby(pd.Grouper(freq="h")).sum()
    share_of_a_hour = (times_df.index[1] - times_df.index[0]).seconds / 3600
    fte_per_hour = fte_per_hour * share_of_a_hour
    additional_indexes = [index
                            for index in fte_per_hour.index
                            if index not in times_df.index]
    diff_df = pd.DataFrame({"FTE (Schedule Resolution)": [0 for i in range(len(additional_indexes))]},
                           index=additional_indexes)
    times_df = pd.concat([diff_df, times_df])
    fte_per_hour.columns = ["FTE (Hour Resolution)"]
    times_df.loc[fte_per_hour.index, "FTE (Hour Resolution)"] = fte_per_hour
    times_df.fillna(0, inplace=True)
    times_df["Time"] = times_df.index
    times_df["Time"] = times_df["Time"].dt.time
    # sort the columns
    times_df = times_df[["Time"] + [col for col in times_df.columns if col != "Time"]]

    return times_df


def _get_single_resource_schedule(schedule_df: pd.DataFrame, start_time):
    columns = schedule_df.columns.to_list()

    columns_datetime = [_get_column_name(column_name, start_time)
                        for column_name in columns]
    schedule_df.columns = columns_datetime

    schedule_columns = [column_name
                        for column_name in columns_datetime
                        if isinstance(column_name, datetime)]

    time_deltas = list(set([schedule_columns[column_idx + 1] - schedule_columns[column_idx]
                            for column_idx in range(0, len(schedule_columns) - 1)]))

    if len(time_deltas) > 1:
        raise Exception("The time deltas are not equal")

    time_delta = time_deltas[0]

    if "MA" in schedule_df.columns:
        resource_name = "MA"
    elif "Resource" in schedule_df.columns:
        resource_name = "Resource"
    else:
        resource_name = None

    resource_available_times = {}
    for worker_name, resource_schedule in schedule_df.groupby(by=resource_name):
        schedule_extract = resource_schedule[schedule_columns]
        schedule_mask = ((schedule_extract == "x") | (schedule_extract == "X") | (schedule_extract == 1)).any().to_numpy()
        start_time_df = schedule_extract.iloc[:, schedule_mask]
        start_times = list(start_time_df.columns)

        available_times = _get_available_times(start_times, time_delta)
        resource_available_times[worker_name] = available_times

    return resource_available_times


def _get_column_name(column_name, start_time):
    if isinstance(column_name, time):
        if start_time:
            date_ = start_time.date()
        else:
            date_ = date.today()
        return datetime.combine(date_, column_name)
    try:
        column_name_converted = datetime.strptime(column_name.lstrip("0"), "%H:%M").time()
    except:
        return column_name

    if isinstance(column_name_converted, time):
        if start_time:
            date_ = start_time.date()
        else:
            date_ = date.today()
        return datetime.combine(date_, column_name_converted)
    else:
        return column_name


def _get_available_times(start_times, time_delta):
    available_times = []
    start_time = None

    for i in range(0, len(start_times)):
        start_time_interim = start_times[i]

        if start_time is None:
            start_time = start_time_interim

        if i + 1 >= len(start_times):
            end_time = start_time_interim + time_delta
            available_times.append((start_time.time(), end_time.time()))
            start_time = None

        elif not (start_times[i + 1] == start_time_interim + time_delta):
            end_time = start_time_interim + time_delta
            available_times.append((start_time.time(), end_time.time()))
            start_time = None

    return available_times


def update_digital_twin_schedules(resources, resource_available_times, start_time_stamp: datetime):
    # needed because the schedule is set up day wise
    start_time_stamp = start_time_stamp.replace(hour=0, minute=0, second=0)

    resources_with_names = {resource.name: resource
                            for resource in resources}
    for day_schedule in resource_available_times:
        for resource_name, available_times in day_schedule.items():

            if resource_name not in resources_with_names:
                names = [name
                         for name in resources_with_names
                         if resource_name in name]
                if names:
                    resource_name = names[0]
                else:
                    print(f"The resource schedule of resource with name {resource_name} cannot be set, "
                      f"because the name is not set in the digital twin model (xlsx-file).")
                    continue
            resource = resources_with_names[resource_name]
            resource_plan: ProcessExecutionPlan = resource.process_execution_plan
            resource_plan.set_available_times(available_times, start_time_stamp, horizont=0)

        start_time_stamp += timedelta(days=1)


def get_resource_performance(resource_scheduling_path, sheet_name):
    df = pd.read_excel(resource_scheduling_path, sheet_name=sheet_name)

    performance_df = df[["Resource", "Efficiency/Speed"]]

    performance_df = performance_df.dropna()
    performance_distribution_df = performance_df["Efficiency/Speed"]
    # for speed as probability distribution
    performance_distribution_s = (
        performance_distribution_df.apply(lambda x:
                                          eval("SingleValueDistribution(" + str(x) + ")")
                                          if x == x else 1))
    # Explicitly cast to object dtype
    performance_df["Efficiency/Speed"] = performance_df["Efficiency/Speed"].astype(object)
    performance_df.loc[:, ["Efficiency/Speed"]] = performance_distribution_s

    resource_performance: Dict[str, Union[SingleValueDistribution]] = dict(zip(performance_df["Resource"],
                                                                               performance_df["Efficiency/Speed"]))

    return resource_performance


def update_resource_performances(resources, resource_performances):
    resources_with_names = {resource.name: resource
                            for resource in resources}

    for resource_name, performance in resource_performances.items():
        if resource_name not in resources_with_names:
            print(f"The resource efficiency/speed of resource with name {resource_name} cannot be set, "
                  f"because the name is not set in the digital twin model (xlsx-file).")
            continue
        resource = resources_with_names[resource_name]

        if hasattr(resource, "efficiency"):
            resource.efficiency = performance
        elif hasattr(resource, "speed"):
            resource.speed = performance
        else:
            raise Exception


if __name__ == "__main__":
    from ofact.settings import ROOT_PATH

    #resource_planning_path = str(ROOT_PATH) + f"/projects/bicycle_world/scenarios/current/models/resource/settings.xlsx"
    #schedules_resources = get_schedules_resource(resource_planning_path)
    #resource_performance = get_resource_performance(resource_planning_path, "Settings")
    #print(schedules_resources)
    #print(resource_performance)
    resource_planning_path = str(ROOT_PATH).rsplit("ofact", 1)[0] + f"/projects/bicycle_world/scenarios/current/models/resource/schedule_s1.xlsx"
    schedule = get_full_time_equivalents_resources(resource_planning_path)

    schedule
