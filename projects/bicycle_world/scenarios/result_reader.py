import os
import statistics
from datetime import datetime

import dill as pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from ofact.twin.agent_control.behaviours.planning.tree.helpers import get_overlaps_periods
from ofact.settings import ROOT_PATH


def lead_time(df):
    new_df = \
        pd.DataFrame({"End": df["End"].iloc[np.r_[-1]][df.index[-1]],
                      "Start": df["Start"].iloc[np.r_[0]][df.index[0]],
                      "Duration": df["End"].iloc[np.r_[-1]][df.index[-1]] - df["Start"].iloc[np.r_[0]][df.index[0]]},
                     index=[df.index[0]])

    return new_df


def calc_utilization(df, start_end):
    start, end = start_end
    time_periods_to_consider = df[(df["End"] > start) & (df["Start"] < end)][["Start", "End"]].to_numpy()

    overlapped_time_periods_cutted = get_overlaps_periods(time_periods_to_consider, np.array([[start, end]]))
    if overlapped_time_periods_cutted.size:
        utilized_time = (overlapped_time_periods_cutted[:, 1] - overlapped_time_periods_cutted[:, 0]).sum()
    else:
        utilized_time = np.timedelta64(0, "s")

    return utilized_time


def get_input_data(general_scenario, file_name, WIP):
    scenario = f"{WIP}"
    results_folder_path = f"projects/bicycle_world/scenarios/{general_scenario}/results"
    resource_pickle_path = os.path.join(ROOT_PATH, fr'{results_folder_path}/resources{file_name}{scenario}.pkl')
    waiting_times_pickle_path = os.path.join(ROOT_PATH,
                                             fr'{results_folder_path}/waiting_times{file_name}{scenario}.pkl')
    order_pickle_path = os.path.join(ROOT_PATH, fr'{results_folder_path}/order{file_name}{scenario}.pkl')

    with open(resource_pickle_path, 'rb') as inp:
        resources_table = pickle.load(inp)
    with open(waiting_times_pickle_path, 'rb') as inp:
        waiting_times = pickle.load(inp)
    with open(order_pickle_path, 'rb') as inp:
        orders_d = pickle.load(inp)

    return resources_table, waiting_times, orders_d


def calc_waiting_times(general_scenario, file_name, wips, work_station_names):
    for WIP in wips:
        resources_table, waiting_times, orders_d = get_input_data(general_scenario, file_name, WIP)

        workstation_dfs = {}
        for resource_name, resource_df in resources_table.items():
            if resource_name not in work_station_names:
                continue

            waiting_time_df = pd.DataFrame({"Start": [], "End": [], "Reason": []})
            waiting_time_df["Start"] = resource_df["End"][:-1].reset_index(drop=True)
            waiting_time_df["End"] = resource_df["Start"][1:].reset_index(drop=True)
            workstation_dfs[resource_name] = waiting_time_df

        order_waiting_times = {}
        for resource_name, resource_df in resources_table.items():

            if "Main Part AGV" not in resource_name:
                continue

            for name, order_df in resource_df.groupby("Work Order ID"):
                time_stamp_last_entry = None
                order_df.loc[
                    order_df["Blocker Name"].str.contains('body_kit_as'), "Blocker Name"] = 'frame and handlebar'
                order_df.loc[
                    order_df["Blocker Name"].str.contains('gear_shift_brakes_as'), "Blocker Name"] = \
                    'gear shift and brakes'
                order_df.loc[
                    order_df["Blocker Name"].str.contains('lightning_pedal_saddle_as'), "Blocker Name"] = \
                    'lightning and  pedal and saddle'
                order_df.loc[
                    order_df["Blocker Name"].str.contains('painting_as'), "Blocker Name"] = 'painting'
                order_df.loc[
                    order_df["Blocker Name"].str.contains('wheel_as'), "Blocker Name"] = 'frame and wheel'

                order_id = order_df["Work Order ID"].iloc[0]
                order_entries = []

                for idx, order_row in order_df.iterrows():
                    if order_row["Blocker Name"] in workstation_dfs:
                        resource_waiting_df = workstation_dfs[order_row["Blocker Name"]]
                        time_slots_mean_time = \
                            resource_waiting_df[(resource_waiting_df["End"] > time_stamp_last_entry) &
                                                (resource_waiting_df["End"] <= order_row["Start"])]
                        if time_slots_mean_time.shape[0]:
                            waiting_time_before = time_slots_mean_time.iloc[-1]
                            if waiting_time_before["Start"] > time_stamp_last_entry:
                                new_entry = {"Start": time_stamp_last_entry,
                                             "End": waiting_time_before["Start"],
                                             "Reason": "Station occupied",
                                             "Participating station": order_row["Blocker Name"]}
                                order_entries.append(new_entry)

                            elif waiting_time_before["Start"] < time_stamp_last_entry:
                                new_entry = {"Start": waiting_time_before["Start"],
                                             "End": time_stamp_last_entry,
                                             "Reason": "Main product transport",
                                             "Participating station": order_row["Blocker Name"]}
                                order_entries.append(new_entry)
                                new_entry = {"Start": time_stamp_last_entry,
                                             "End": waiting_time_before["End"],
                                             "Reason": "Material supply",
                                             "Participating station": order_row["Blocker Name"]}
                                order_entries.append(new_entry)
                                continue

                            if waiting_time_before["Start"] != waiting_time_before["End"]:
                                new_entry = {"Start": waiting_time_before["Start"],
                                             "End": waiting_time_before["End"],
                                             "Reason": "Material supply",
                                             "Participating station": order_row["Blocker Name"]}
                                order_entries.append(new_entry)
                        else:
                            new_entry = {"Start": time_stamp_last_entry,
                                         "End": order_row["Start"],
                                         "Reason": "Station occupied",
                                         "Participating station": order_row["Blocker Name"]}
                            order_entries.append(new_entry)

                    time_stamp_last_entry = order_row["End"]

                order_waiting_times[order_id] = pd.DataFrame(order_entries)
                order_waiting_times[order_id]["Duration"] = order_waiting_times[order_id]["End"] - \
                                                            order_waiting_times[order_id]["Start"]

        return order_waiting_times


def get_output_data(general_scenario, file_name, wips, work_station_names, order_waiting_times, focus):
    wip_results_over_time = {}
    for WIP in wips:
        resources_table, waiting_times, orders_d = get_input_data(general_scenario, file_name, WIP)

        earliest_start_time = datetime.max
        latest_end_time = datetime.min
        resource_lst = []
        order_df = pd.DataFrame()
        for resource_name, resource_df in resources_table.items():
            resource_d = {"name": resource_name, "time_table": resource_df}
            resource_lst.append(resource_d)
            if resource_df.shape[0] == 0:
                continue
            start_time = resource_df["Start"].iloc[0]
            end_time = resource_df["End"].iloc[-1]
            if start_time < earliest_start_time:
                earliest_start_time = start_time
            if end_time > latest_end_time:
                latest_end_time = end_time

            if "Individual" not in resource_name and "Main Part AGV" not in resource_name and focus == "MATERIAL_SUPPLY":
                continue

            if resource_name not in work_station_names and "Main" not in resource_name and focus == "MAIN":
                continue

            resource_d["used_time"] = (resource_df["End"] - resource_df["Start"]).sum()

            if "Main Part AGV" in resource_name:
                order_df_batch = resource_df.groupby("Work Order ID").apply(lead_time)
                order_df_batch = order_df_batch.reset_index(level=1, drop=True)
                order_df = pd.concat([order_df, order_df_batch])

        consideration_period = latest_end_time - earliest_start_time
        bin_size = 10
        date_range_ = pd.date_range(earliest_start_time, latest_end_time, freq=f"{bin_size}min")
        time_periods = np.concatenate([np.expand_dims(date_range_[0:-1], axis=1),
                                       np.expand_dims(date_range_[1:], axis=1)], axis=1)

        utilization_df = pd.DataFrame()

        transport_times = []
        for resource_d in resource_lst:
            if not ("Individual" in resource_d["name"] or "Main Part AGV" in resource_d["name"]) and \
                    focus == "MATERIAL_SUPPLY":
                continue

            if resource_d["name"] not in work_station_names and focus == "MAIN":
                continue
            if "used_time" in resource_d:
                if "Main " not in resource_d["name"]:
                    transport_times.append(resource_d["time_table"]["Duration"].sum().total_seconds() * 4 /
                                           resource_d["time_table"]["Duration"].shape[0])

                    utilizations = \
                        [calc_utilization(resource_d["time_table"], time_period) / np.timedelta64(bin_size, "m")
                         for time_period in time_periods]
                    utilization_s = pd.DataFrame(utilizations, columns=[resource_d["name"]])
                    utilization_df = pd.concat([utilization_df, utilization_s], axis=1)

                resource_d["utilization"] = resource_d["used_time"] / consideration_period
                # print(resource_d["name"], " ", resource_d["utilization"])

        # print("Mean transport time: ", statistics.mean(transport_times))

        index_len = len(utilization_df.index)
        target_value = utilization_df[int(index_len / 2) - 25: int(index_len / 2) + 25].mean(axis=1).mean(axis=0)
        settled_df = utilization_df[utilization_df.mean(axis=1).rolling(3, min_periods=3).sum() >
                                    target_value * 3]
        first_index = settled_df.index[0]
        last_index = settled_df.index[-1]

        utilization_clean_df = utilization_df[first_index: last_index] * 100

        first_time_stamp = earliest_start_time + pd.Timedelta(bin_size * first_index, "min")
        latest_time_stamp = earliest_start_time + pd.Timedelta((bin_size + 1) * last_index, "min")

        order_df = order_df[(order_df["Start"] > first_time_stamp) & (order_df["End"] < latest_time_stamp)]
        # order_waiting_df_settled = {order_id: order_waiting_df
        #                             for order_id, order_waiting_df in order_waiting_times.items()
        #                             if order_id in order_df.index.to_list()}
        #
        # all_waiting_times = pd.concat(list(order_waiting_df_settled.values()))
        # waiting_times_material_supply_work_stations = {}
        # waiting_times_occupied_work_stations = {}
        # waiting_times_main_product_work_stations = {}
        # for work_station_name in work_station_names:
        #     waiting_times = all_waiting_times.loc[all_waiting_times["Participating station"] == work_station_name]
        #     waiting_times_material_supply_work_stationwork_stations[work_station_name] = \
        #         waiting_times.loc[waiting_times["Reason"] == "Material supply"]["Duration"].to_list()
        #     waiting_times_occupied_work_stations[work_station_name] = \
        #         waiting_times.loc[waiting_times["Reason"] == "Station occupied"]["Duration"].to_list()
        #     waiting_times_main_product_work_stations[work_station_name] = \
        #         waiting_times.loc[waiting_times["Reason"] == "Main product transport"]["Duration"].to_list()

        # order_waiting_df_settled
        orders_mean = int(order_df["Duration"].mean().seconds / 60)

        orders_min = int(order_df["Duration"].min().seconds / 60)
        orders_max = int(order_df["Duration"].max().seconds / 60)

        resources_waiting_time_mean = {}
        resources_waiting_time_sum = {}
        resources_waiting_times_a = {}

        for resource_name, waiting_times_lst in waiting_times.items():
            if not ("Individual" in resource_name or "Main Part AGV" in resource_name) and \
                    focus == "MATERIAL_SUPPLY":
                continue

            if resource_name not in work_station_names and focus == "MAIN":
                continue

            resources_waiting_time_mean[resource_name] = statistics.mean(waiting_times_lst)  # box_plot
            resources_waiting_time_sum[resource_name] = sum(waiting_times_lst)
            resources_waiting_times_a[resource_name] = np.array(waiting_times_lst)

        wip_results_over_time[WIP] = {"Utilization Raw": utilization_clean_df,
                                      "Utilization Mean": utilization_clean_df.mean(axis=1).mean(axis=0),
                                      "Utilization": utilization_clean_df.mean(axis=0),
                                      "Lead Time Mean": orders_mean,
                                      "Lead Time Min": orders_min,
                                      "Lead Time Max": orders_max,
                                      # "Waiting time Material supply": waiting_times_material_supply_work_stations,
                                      # "Waiting time Station occupied": waiting_times_occupied_work_stations,
                                      # "Waiting time Main product": waiting_times_main_product_work_stations,
                                      "Waiting Time Mean": resources_waiting_time_mean,
                                      "Waiting Time Sum": resources_waiting_time_sum,
                                      "Waiting Time array": resources_waiting_times_a,
                                      "Bin Size": bin_size}

    return wip_results_over_time


def visualize(wip_results_over_time, wips):
    for WIP in wips:
        results_over_time = wip_results_over_time[WIP]

        results_over_time["Utilization Raw"].plot()
        results_over_time["Utilization Raw"].mean(axis=1).plot(figsize=(20, 5), linewidth=5.0)

        bin_size = results_over_time["Bin Size"]
        plt.xlabel(f'time ({bin_size} s bins)')
        plt.ylabel('Utilization (%)')
        plt.title(f'Utilization (ends cut) \n '
                  f'for a WIP of {WIP} Mean {results_over_time["Utilization Raw"].mean(axis=1).mean(axis=0)} \n '
                  f'Lead Time (min): Mean {results_over_time["Lead Time Mean"]}, '
                  f'Min {results_over_time["Lead Time Min"]}, '
                  f'Max {results_over_time["Lead Time Max"]}',
                  style='italic', bbox={'facecolor': 'grey', 'alpha': 0.5})

        plt.grid(True)
        plt.show()
        # fig, axs = plt.subplots(1, 5, figsize=(16, 6))
        # for idx, (work_station_name, waiting_times) in \
        #         enumerate(results_over_time["Waiting time Material supply"].items()):
        #     axs[idx].boxplot([waiting_time.seconds for waiting_time in waiting_times], labels=[work_station_name])
        # plt.show()

        # "Waiting Time Mean"
        # "Waiting Time Sum"
        # for resource, waiting_time_array in results_over_time["Waiting Time array"].items():
        #     plt.boxplot(waiting_time_array, label=resource)

        # if results_over_time["Waiting Time array"]:
        #     fig, ax = plt.subplots()
        #     ax.boxplot(results_over_time["Waiting Time array"].values())
        #     ax.set_xticklabels([name[:10] for name in list(results_over_time["Waiting Time array"].keys())])
        #     # plt.show()

    utilization_mean = {wip: info_d["Utilization Mean"] for wip, info_d in wip_results_over_time.items()}
    utilization = {wip: info_d["Utilization"] for wip, info_d in wip_results_over_time.items()}
    lead_mean = {wip: info_d["Lead Time Mean"] for wip, info_d in wip_results_over_time.items()}
    lead_max = {wip: info_d["Lead Time Max"] for wip, info_d in wip_results_over_time.items()}
    lead_min = {wip: info_d["Lead Time Min"] for wip, info_d in wip_results_over_time.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(list(utilization_mean.keys()), list(utilization_mean.values()), label="mean", linewidth=5.0)
    for index in range(len(list(utilization_mean.keys()))):
        if list(utilization_mean.keys())[index] == 19:
            ax1.text(list(utilization_mean.keys())[index] - 1, list(utilization_mean.values())[index] + 1,
                     round(list(utilization_mean.values())[index], 1), size=12)
    if focus == "MAIN":
        len_utilizations = len(list(utilization.values())[0])
        for i in range(len_utilizations):
            if list(list(utilization.values())[0].keys())[i] != "lightning and pedal  and saddle":
                label = list(list(utilization.values())[0].keys())[i]
            else:
                label = "lighting and pedal and saddle"
            ax1.plot(list(utilization.keys()), np.array(list(utilization.values()))[:, i],
                     label=label)
    ax1.legend(loc="best")
    ax1.plot(list(utilization_mean.keys()), list(utilization_mean.values()), "ro")
    for index in range(len(list(utilization_mean.keys()))):
        if list(utilization_mean.keys())[index] == 19:
            ax1.plot(list(utilization_mean.keys())[index], list(utilization_mean.values())[index], "wo")
    ax1.set_xlabel(f'WIP (Amount)')
    ax1.set_ylabel('Capacity Utilization (%)')
    ax1.set_title(f'Capacity Utilization Workstations \n ', style='italic', fontdict={'fontsize': rcParams['axes.titlesize']})
    ax1.grid(True)

    ax2.plot(list(lead_mean.keys()), list(lead_mean.values()), label="mean", linewidth=5.0)
    for index in range(len(list(lead_mean.keys()))):
        if list(lead_mean.keys())[index] == 19:
            ax2.text(list(lead_mean.keys())[index] - 1, list(lead_mean.values())[index] + 10,
                     round(list(lead_mean.values())[index], 1), size=12)

    ax2.plot(list(lead_mean.keys()), list(lead_mean.values()), "ro")
    for index in range(len(list(lead_mean.keys()))):
        if list(lead_mean.keys())[index] == 19:
            ax2.plot(list(lead_mean.keys())[index], list(lead_mean.values())[index], "wo")
    ax2.plot(list(lead_max.keys()), list(lead_max.values()), label="max")
    ax2.plot(list(lead_min.keys()), list(lead_min.values()), label="min")
    ax2.legend(loc="best")
    ax2.set_xlabel(f'WIP (Amount)')
    ax2.set_ylabel('Lead Time (min)')
    ax2.set_title(f'Lead Time Orders \n ', style='italic', fontdict={'fontsize': rcParams['axes.titlesize'],
                                                                     'fontweight': rcParams['axes.titleweight']})
    ax2.grid(True)
    plt.show()


def visualize_scenario_comparison(wip_results_over_time1, wip_results_over_time2, wips, scenario_name1, scenario_name2):
    for WIP in wips:
        results_over_time1 = wip_results_over_time1[WIP]
        results_over_time2 = wip_results_over_time2[WIP]

        results_over_time1["Utilization Raw"].mean(axis=1).plot(figsize=(20, 5), color="blue", label=scenario_name1)
        results_over_time2["Utilization Raw"].mean(axis=1).plot(figsize=(20, 5), color="cyan", label=scenario_name2)

        bin_size = results_over_time1["Bin Size"]
        plt.xlabel(f'time ({bin_size} s bins)')
        plt.ylabel('Utilization (%)')
        plt.title(f'Utilization (ends cut) \n '
                  f'for a WIP of {WIP} \n {scenario_name1} mean \n'
                  f'{round(results_over_time1["Utilization Raw"].mean(axis=1).mean(axis=0), 2)} \n '
                  f'{scenario_name2} mean {round(results_over_time2["Utilization Raw"].mean(axis=1).mean(axis=0), 2)}',
                  style='italic')

        plt.grid(True)
        plt.show()

    utilization_mean1 = {wip: info_d["Utilization Mean"] for wip, info_d in wip_results_over_time1.items()}
    utilization_mean2 = {wip: info_d["Utilization Mean"] for wip, info_d in wip_results_over_time2.items()}
    utilization1 = {wip: info_d["Utilization"] for wip, info_d in wip_results_over_time1.items()}
    utilization2 = {wip: info_d["Utilization"] for wip, info_d in wip_results_over_time2.items()}
    lead_mean1 = {wip: info_d["Lead Time Mean"] for wip, info_d in wip_results_over_time1.items()}
    lead_mean2 = {wip: info_d["Lead Time Mean"] for wip, info_d in wip_results_over_time2.items()}
    lead_max1 = {wip: info_d["Lead Time Max"] for wip, info_d in wip_results_over_time1.items()}
    lead_max2 = {wip: info_d["Lead Time Max"] for wip, info_d in wip_results_over_time2.items()}
    lead_min1 = {wip: info_d["Lead Time Min"] for wip, info_d in wip_results_over_time1.items()}
    lead_min2 = {wip: info_d["Lead Time Min"] for wip, info_d in wip_results_over_time2.items()}

    # waiting_time_material_supply1 = \
    #     {wip: sum([waiting_time.seconds
    #                for waiting_times in list(info_d["Waiting time Material supply"].values())
    #                for waiting_time in waiting_times]) / (3600 * len(list(info_d["Waiting time Material supply"].values())))
    #      for wip, info_d in wip_results_over_time1.items()}
    # waiting_time_material_supply2 = \
    #     {wip: sum([waiting_time.seconds
    #                for waiting_times in list(info_d["Waiting time Material supply"].values())
    #                for waiting_time in waiting_times]) / (3600 * len(list(info_d["Waiting time Material supply"].values())))
    #      for wip, info_d in wip_results_over_time2.items()}
    # waiting_time_station_occupied1 = \
    #     {wip: sum([waiting_time.seconds
    #                for waiting_times in list(info_d["Waiting time Station occupied"].values())
    #                for waiting_time in waiting_times]) / (3600 * len(list(info_d["Waiting time Station occupied"].values())))
    #      for wip, info_d in wip_results_over_time1.items()}
    # waiting_time_station_occupied2 = \
    #     {wip: sum([waiting_time.seconds
    #                for waiting_times in list(info_d["Waiting time Station occupied"].values())
    #                for waiting_time in waiting_times]) / (3600 * len(list(info_d["Waiting time Station occupied"].values())))
    #      for wip, info_d in wip_results_over_time2.items()}
    # waiting_time_main_product1 = \
    #     {wip: sum([waiting_time.seconds
    #                for waiting_times in list(info_d["Waiting time Main product"].values())
    #                for waiting_time in waiting_times]) / (3600 * len(list(info_d["Waiting time Main product"].values())))
    #      for wip, info_d in wip_results_over_time1.items()}
    # waiting_time_main_product2 = \
    #     {wip: sum([waiting_time.seconds
    #                for waiting_times in list(info_d["Waiting time Main product"].values())
    #                for waiting_time in waiting_times]) / (3600 * len(list(info_d["Waiting time Main product"].values())))
    #      for wip, info_d in wip_results_over_time2.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # if focus == "MAIN":
    #     len_utilizations1 = len(list(utilization1.values())[0])
    #     for i in range(len_utilizations1):
    #         ax1.plot(list(utilization1.keys()), np.array(list(utilization1.values()))[:, i],
    #                  # label=scenario_name1 + " " + list(list(utilization1.values())[0].keys())[i],
    #                  color="blue", alpha=0.25 + i * 0.1)
    #     len_utilizations2 = len(list(utilization2.values())[0])
    #     for i in range(len_utilizations2):
    #         ax1.plot(list(utilization2.keys()), np.array(list(utilization2.values()))[:, i],
    #                  # label=scenario_name2 + " " + list(list(utilization2.values())[0].keys())[i],
    #                  color="cyan", alpha=0.25 + i * 0.15)
    ax1.plot(list(utilization_mean1.keys()), list(utilization_mean1.values()), label=f"mean {scenario_name1}",
             color="blue", linewidth=5.0)
    ax1.plot(list(utilization_mean2.keys()), list(utilization_mean2.values()), label=f"mean {scenario_name2}",
             color="cyan", linewidth=5.0)
    ax1.plot(list(utilization_mean1.keys()), list(utilization_mean1.values()), "bo")
    ax1.plot(list(utilization_mean2.keys()), list(utilization_mean2.values()), "co")
    ax1.set_xticks(np.array(list(utilization_mean1.keys())))
    # ax1.bar(np.array(list(utilization_mean1.keys())) - 0.15, list(utilization_mean1.values()), label=f"mean {scenario_name1}",
    #          color="blue", width=0.3)
    # ax1.bar(np.array(list(utilization_mean2.keys())) + 0.15, list(utilization_mean2.values()), label=f"mean {scenario_name2}",
    #          color="orange", width=0.3)
    # ax1.set(ylim=[50, 100])
    #
    # XX = pd.Series(list(utilization_mean1.values()), index=np.array(list(utilization_mean1.keys())) - 0.15)
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
    #                                figsize=(5, 6))
    # ax1.spines['bottom'].set_visible(False)
    # ax1.tick_params(axis='x', which='both', bottom=False)
    # ax2.spines['top'].set_visible(False)
    #
    # bs = 10
    # ts = 50
    #
    # ax2.set_ylim(0, bs)
    # ax1.set_ylim(ts, 100)
    # ax1.set_yticks(np.arange(50, 101, 50))
    #
    # bars1 = ax1.bar(XX.index, XX.values)
    # bars2 = ax2.bar(XX.index, XX.values)
    #
    # for tick in ax2.get_xticklabels():
    #     tick.set_rotation(0)
    # d = .015
    # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    # ax1.plot((-d, +d), (-d, +d), **kwargs)
    # ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    # kwargs.update(transform=ax2.transAxes)
    # ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    # ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    #
    # for b1, b2 in zip(bars1, bars2):
    #     posx = b2.get_x() + b2.get_width() / 2.
    #     if b2.get_height() > bs:
    #         ax2.plot((posx - 3 * d, posx + 3 * d), (1 - d, 1 + d), color='k', clip_on=False,
    #                  transform=ax2.get_xaxis_transform())
    #     if b1.get_height() > ts:
    #         ax1.plot((posx - 3 * d, posx + 3 * d), (- d, + d), color='k', clip_on=False,
    #                  transform=ax1.get_xaxis_transform())

    ax1.legend(loc="best")
    ax1.set_xlabel(f'Part Supply AGV Fleet size (Amount)')
    ax1.set_ylabel('Capacity Utilization (%)')
    # ax1.set_title(f'Capacity Utilization Workstations \n ', style='italic')
    ax1.set_title(f'Capacity Utilization Part AGVs \n ', style='italic')
    ax1.grid(True)

    ax2.plot(list(lead_mean1.keys()), list(lead_mean1.values()),
             label=f"mean {scenario_name1}", color="blue", linewidth=5.0)
    ax2.plot(list(lead_mean1.keys()), list(lead_mean1.values()), "bo")
    ax2.plot(list(lead_max1.keys()), list(lead_max1.values()),
             label=f"max {scenario_name1}", color="blue", alpha=0.75)
    ax2.plot(list(lead_min1.keys()), list(lead_min1.values()),
             label=f"min {scenario_name1}", color="blue", alpha=0.25)
    ax2.plot(list(lead_mean2.keys()), list(lead_mean2.values()),
             label=f"mean {scenario_name2}", color="cyan", linewidth=5.0)
    ax2.plot(list(lead_mean2.keys()), list(lead_mean2.values()), "co")
    ax2.plot(list(lead_max2.keys()), list(lead_max2.values()),
             label=f"max {scenario_name2}", color="cyan", alpha=0.75)
    ax2.plot(list(lead_min2.keys()), list(lead_min2.values()),
             label=f"min {scenario_name2}", color="cyan", alpha=0.25)
    ax2.legend(loc="best")
    ax2.set_xlabel(f'Part Supply AGV Fleet size (Amount)')
    ax2.set_ylabel('Lead Time (min)')
    ax2.set_title(f'Lead Time Orders \n ', style='italic')
    ax2.set_xticks(np.array(list(lead_min1.keys())))
    ax2.grid(True)
    plt.show()

    # plt.plot(list(waiting_time_material_supply1.keys()), list(waiting_time_material_supply1.values()),
    #          label=f"{scenario_name1} material supply sum", color="blue", linewidth=5.0, alpha=0.33)
    # plt.plot(list(waiting_time_material_supply1.keys()), list(waiting_time_material_supply1.values()), "bo")
    #
    # plt.plot(list(waiting_time_material_supply2.keys()), list(waiting_time_material_supply2.values()),
    #          label=f"{scenario_name2} material supply sum", color="orange", linewidth=5.0, alpha=0.33)
    # plt.plot(list(waiting_time_material_supply2.keys()), list(waiting_time_material_supply2.values()), "ro")
    #
    # plt.plot(list(waiting_time_station_occupied1.keys()), list(waiting_time_station_occupied1.values()),
    #          label=f"{scenario_name1} Station occupied sum", color="blue", linewidth=5.0, alpha=0.66)
    # plt.plot(list(waiting_time_station_occupied1.keys()), list(waiting_time_station_occupied1.values()), "bo")
    #
    # plt.plot(list(waiting_time_station_occupied2.keys()), list(waiting_time_station_occupied2.values()),
    #          label=f"{scenario_name2} Station occupied sum", color="orange", linewidth=5.0, alpha=0.66)
    # plt.plot(list(waiting_time_station_occupied2.keys()), list(waiting_time_station_occupied2.values()), "ro")
    #
    # plt.plot(list(waiting_time_main_product1.keys()), list(waiting_time_main_product1.values()),
    #          label=f"{scenario_name1} Main product sum", color="blue", linewidth=5.0)
    # plt.plot(list(waiting_time_main_product1.keys()), list(waiting_time_main_product1.values()), "bo")
    #
    # plt.plot(list(waiting_time_main_product2.keys()), list(waiting_time_main_product2.values()),
    #          label=f"{scenario_name2} Main product sum", color="orange", linewidth=5.0)
    # plt.plot(list(waiting_time_main_product2.keys()), list(waiting_time_main_product2.values()), "ro")
    #
    # plt.legend(loc="best")
    # plt.xlabel(f'WIP (Amount)')
    # plt.ylabel('Waiting Time (h)')
    # plt.title(f'Waiting Time\n ', style='italic')
    # plt.grid(True)
    # plt.show()

font = {'size'   : 15}

matplotlib.rc('font', **font)
file_name = "base_general1_19_"  # "bicycle_world_base_general_" #
# [2, 3, 4, 5, 6]  #
wips = [3, 4, 5, 6]#, 6, 7, 8]  # [3, 4, 5, 6, 7]  # [6, 8, 10, 12, 14, 16, 18, 19, 20, 21, 22, 24, 26, 28, 30]  #   #

focus =  "MATERIAL_SUPPLY"  # "MAIN" #
work_station_names = ["frame and handlebar", "gear shift and brakes", "lightning and pedal  and saddle", "wheel",
                          "painting"]

general_scenario1 = "vehicle_availability"  # #"base", "vehicle_availability"
general_scenario2 = "base"

scenario_name1 = "Decoupled"
scenario_name2 = "Integrated"

# order_waiting_times1 = calc_waiting_times(general_scenario1, file_name, wips, work_station_names)
order_waiting_times1 = {}
wip_results_over_time1 = get_output_data(general_scenario1, file_name, wips, work_station_names,
                                         order_waiting_times1, focus)
visualize(wip_results_over_time1, wips)

#order_waiting_times2 = calc_waiting_times(general_scenario2, file_name, wips, work_station_names)
order_waiting_times2 = {}
wip_results_over_time2 = get_output_data(general_scenario2, file_name, wips, work_station_names,
                                         order_waiting_times2, focus)
visualize(wip_results_over_time2, wips)
#
visualize_scenario_comparison(wip_results_over_time1, wip_results_over_time2, wips,
                              scenario_name1, scenario_name2)
