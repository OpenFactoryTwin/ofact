import os
from datetime import datetime, timedelta
import dill as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ofact.twin.state_model.processes import ProcessExecution
from ofact.settings import ROOT_PATH


# visualize
def visualize(results, order_delivery_dates_planned, order_delivery_dates_actual, id_name_match):
    schedule = pd.DataFrame(results)
    start_min = schedule['Start'].min()
    schedule['Start'] = schedule['Start'] - start_min
    schedule['Finish'] = schedule['Finish'] - start_min
    ORDERS = sorted(list(schedule['Order'].unique()))
    WORK_CENTERS = sorted(list(schedule['Work Center'].unique()))
    makespan = schedule['Finish'].max()
    order_delivery_dates_actual = [(order_id, order_delivery_date - start_min)
                                   for order_id, order_delivery_date in order_delivery_dates_actual]
    order_delivery_dates_planned = [(order_id, order_delivery_date - start_min)
                                    for order_id, order_delivery_date in order_delivery_dates_planned]

    bar_style_o = {'alpha': 1, 'lw': 9, 'solid_capstyle': 'butt'}
    bar_style_o2 = {'alpha': .5, 'lw': 10, 'solid_capstyle': 'butt'}
    bar_style_w = {'alpha': 1, 'lw': 9, 'solid_capstyle': 'butt'}
    bar_style_w2 = {'alpha': .5, 'lw': 10, 'solid_capstyle': 'butt'}
    text_style = {'color': 'black', 'weight': 'bold', 'ha': 'center', 'va': 'center'}
    colors = mpl.cm.Dark2.colors
    colors2 = mpl.cm.get_cmap("tab20c").colors
    schedule.sort_values(by=['Order', 'Start'])
    schedule.set_index(['Order', 'Work Center'], inplace=True)
    schedule.sort_index(inplace=True)

    fig, ax = plt.subplots(2, 1, figsize=(schedule.shape[0] / 1.5, int(5 + (len(ORDERS) + len(WORK_CENTERS)) / 4)))

    for jdx, j in enumerate(ORDERS, 1):
        for mdx, m in enumerate(WORK_CENTERS, 1):
            if (j, m) in schedule.index:
                xs = schedule.loc[(j, m), 'Start']
                xf = schedule.loc[(j, m), 'Finish']
                for idx, xs_single in enumerate(xs):
                    xf_single = xf[idx]
                    ax[0].plot([xs_single, xf_single], [jdx] * 2, c=colors2[17], **bar_style_o2)
                    ax[0].plot([xs_single, xf_single], [jdx] * 2, c=colors[mdx % 7], **bar_style_o)
                    ax[0].text((xs_single + xf_single) / 2, jdx, "", **text_style)  # m,

                    ax[1].plot([xs_single, xf_single], [mdx] * 2, c=colors2[17], **bar_style_w2)
                    ax[1].plot([xs_single, xf_single], [mdx] * 2, c=colors[jdx % 7], **bar_style_w)
                    ax[1].text((xs_single + xf_single) / 2, mdx, "", **text_style)  # j,

    ax[0].set_title('Order Schedule')
    ax[0].set_ylabel('Order')
    ax[1].set_title('Resource Schedule')
    ax[1].set_ylabel('Resource')

    considered_order_delivery_dates_actual = [(order_id, order_date)
                                              for order_id, order_date in order_delivery_dates_actual
                                              for order_str in ORDERS
                                              if str(order_id) == order_str]
    considered_order_delivery_dates_planned = [(order_id, order_date)
                                               for order_id, order_date in order_delivery_dates_planned
                                               for order_str in ORDERS
                                               if str(order_id) == order_str]
    for idx, s in enumerate([ORDERS, WORK_CENTERS]):
        ax[idx].set_ylim(0.5, len(s) + 0.5)
        ax[idx].set_yticks(range(1, 1 + len(s)))
        # ax[idx].set_yticklabels(s)
        # , fontdict={'fontsize': mpl.rcParams['axes.grid'],
        #                                     'fontweight': mpl.rcParams['axes.titleweight']}
        # ax[idx].text(makespan, ax[idx].get_ylim()[0] - 0.2, "{0:0.1f}".format(makespan), ha='center', va='top')
        # ax[idx].plot([makespan] * 2, ax[idx].get_ylim(), 'r--')
        if idx == 0:  # order
            for idx2, (order_id, order_delivery_date_actual) in enumerate(considered_order_delivery_dates_actual):
                ax[idx].text(-200, np.mean(np.linspace(*ax[0].get_ylim(), len(s) + 1)[idx2:idx2 + 2]),
                             str(order_id), verticalalignment="baseline")
                if 0 < order_delivery_date_actual < makespan:
                    ax[idx].plot([order_delivery_date_actual] * 2,
                                 np.linspace(*ax[0].get_ylim(), len(s) + 1)[idx2:idx2 + 2],
                                 'b-.')
                if 0 < considered_order_delivery_dates_planned[idx2][1] < makespan:
                    ax[idx].plot([considered_order_delivery_dates_planned[idx2][1]] * 2,
                                 np.linspace(*ax[0].get_ylim(), len(s) + 1)[idx2:idx2 + 2],
                                 'v-.')
        elif idx == 1:
            for idx3, workcenter_id in enumerate(WORK_CENTERS):
                ax[idx].text(-250, np.mean(np.linspace(*ax[1].get_ylim(), len(s) + 1)[idx3:idx3 + 2]),
                             str(id_name_match[workcenter_id])[:7],  # + str(workcenter_id)
                             verticalalignment="baseline")
        # ax[idx].set_xlabel('Time')
        ax[idx].grid(True)
        ax[idx].set_yticklabels(s)
        ax[idx].set_xticklabels([(datetime.fromtimestamp(x_tick_label + start_min).replace(second=0, microsecond=0)
                                  + timedelta(hours=2)).time()
                                 for x_tick_label in ax[idx].get_xticks()])
    fig.tight_layout()
    plt.show()


def get_visualization_input(digital_twin):
    process_executions_with_time_stamps = {}
    process_executions_list = digital_twin.get_process_executions_list()
    for process_execution in process_executions_list:
        process_executions_with_time_stamps.setdefault(process_execution.executed_start_time,
                                                       []).append(process_execution)

    time_stamps = sorted(list(process_executions_with_time_stamps.keys()))

    process_execution_order_match = \
        {process_execution: order
         for process_execution in process_executions_list
         for part_tuple in process_execution.parts_involved
         for order in digital_twin.get_orders()
         for product in order.products
         if product.identification == part_tuple[0].identification}

    results_plan = []
    results_actual = []
    problem = 0
    id_name_match = {}

    for time_stamp in time_stamps:
        for process_execution in process_executions_with_time_stamps[time_stamp]:
            for resource in process_execution.resources_used:
                if resource.identification not in id_name_match:
                    if "Main Part AGV" in resource.name:
                        id_name_match[resource.identification] = "AGV " + resource.name.split("AGV ")[-1]
                    else:
                        id_name_match[resource.identification] = resource.name

                entry = {}

                entry['Duration'] = int((process_execution.executed_end_time -
                                         process_execution.executed_start_time).seconds)
                entry['Start'] = process_execution.executed_start_time.timestamp()
                entry['Finish'] = process_execution.executed_end_time.timestamp()

                if process_execution not in process_execution_order_match:
                    entry['Order'] = str(0)
                else:
                    entry['Order'] = str(int(process_execution_order_match[process_execution].identification))

                entry['Work Center'] = int(resource.identification)

                if process_execution.event_type == ProcessExecution.EventTypes.PLAN:
                    results_plan.append(entry)
                elif process_execution.event_type == ProcessExecution.EventTypes.ACTUAL:
                    results_actual.append(entry)
                else:
                    raise ValueError("Problem")

    order_delivery_dates_planned = []
    order_delivery_dates_actual = []
    for order in digital_twin.get_orders():
        order_delivery_dates_planned.append((order.identification,
                                             order.delivery_date_planned.timestamp()))
        order_delivery_dates_actual.append((order.identification,
                                            order.delivery_date_actual.timestamp()))
        # print(order.delivery_date_planned)
        # print(order.delivery_date_actual)
        # # print([feature.name for feature in order.features_completed])
        # print("\n")

    return results_plan, results_actual, order_delivery_dates_planned, order_delivery_dates_actual, id_name_match


# scenarios
scenario = "bicycle_world_debug"


# file_path = f'projects/bicycle_world/models/bicycle_world_{scenario}.pkl'
file_path = f"projects/bicycle_world/results/{scenario}.pkl"
digital_twin_pickle_path = os.path.join(ROOT_PATH, os.path.normpath(file_path))
with open(digital_twin_pickle_path, 'rb') as inp:
    digital_twin = pickle.load(inp)

results_plan, results_actual, order_delivery_dates_planned, order_delivery_dates_actual, id_name_match = \
    get_visualization_input(digital_twin)

phase = 1
visualize(results_plan[int(len(results_plan) / 4) * phase:int(len(results_plan) / 4) * (phase + 1)],
          order_delivery_dates_planned, order_delivery_dates_actual, id_name_match)

visualize(results_actual[int(len(results_actual) / 4) * phase: int(len(results_actual) / 4) * (phase + 1)],
          order_delivery_dates_planned, order_delivery_dates_actual, id_name_match)
