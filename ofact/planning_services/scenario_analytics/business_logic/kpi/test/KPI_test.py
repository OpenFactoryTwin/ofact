import pandas as pd

import unittest
import dill as pickle
import os
from ofact.planning_services.scenario_analytics.scenario_handling.single import SingleScenarioHandler
from ofact.settings import ROOT_PATH

# ToDo: update required!
class KPI_test(unittest.TestCase):
    # scenarios
    scenario = 'Test_file_kpi'
    # scenario = 'digital_twinbase_19_3KPI'
    # scenario = "base_mini"
    # scenario = "base"
    # scenario = "one_further_agv"
    # scenario = "two_further_agv"
    # scenario = "three_further_agv"
    # scenario = "four_further_agv"
    # scenario = "five_further_agv"
    # scenario = "six_further_agv"
    # scenario = "four_painting"
    # scenario = "four_gear"

    file_path = f'projects/bicycle_world/scenarios/base/results/{scenario}.pkl'
    # file_path = "DigitalTwin/projects/bicycle_world/models/debug.pkl"
    digital_twin_pickle_path = os.path.join(ROOT_PATH, os.path.normpath(file_path))

    """"
    digital_twin_objects = StaticModelGenerator.from_pickle(digital_twin_pickle_path=digital_twin_pickle_path)

    all_digital_twin_objects = digital_twin_objects.get_all_digital_twin_objects()
    digital_twin = digital_twin_objects.get_digital_twin()



    """
    with open(digital_twin_pickle_path, 'rb') as inp:
        try:
            analytics_data_base = pickle.load(inp)
        except EOFError:
            analytics_data_base = list(inp)
    SingleScenarioHandler(analytics_data_base=analytics_data_base)

    # KPInitializer(digital_twin=digital_twin)
    order_ids = SingleScenarioHandler.analytics_data_base.digital_twin_df.loc[
        SingleScenarioHandler.analytics_data_base.digital_twin_df['Customer ID'].notnull(), 'Customer ID']
    order_ids = order_ids.drop_duplicates()
    order_ids = pd.to_numeric(order_ids, errors='coerce')
    order_ids = order_ids.dropna().astype(int)
    order_ids = order_ids.to_list()
    part_ids = SingleScenarioHandler.analytics_data_base.digital_twin_df.loc[
        SingleScenarioHandler.analytics_data_base.digital_twin_df['Entity Type ID'].notnull(), 'Entity Type ID']
    part_ids = part_ids.drop_duplicates()
    part_ids = part_ids.to_list()
    process_ids = SingleScenarioHandler.analytics_data_base.digital_twin_df.loc[
        SingleScenarioHandler.analytics_data_base.digital_twin_df['Process ID'].notnull(), 'Process ID']
    process_ids = process_ids.drop_duplicates()
    process_ids = process_ids.to_list()
    resource_ids = SingleScenarioHandler.analytics_data_base.digital_twin_df.loc[
        SingleScenarioHandler.analytics_data_base.digital_twin_df['Resource Used ID'].notnull(), 'Resource Used ID']
    resource_ids = resource_ids.drop_duplicates()
    resource_ids = resource_ids.to_list()
    start_time = SingleScenarioHandler.analytics_data_base.digital_twin_df.loc[
        SingleScenarioHandler.analytics_data_base.digital_twin_df['Start Time'].notnull(), 'Start Time']
    start_time = min(start_time)
    # date_format = "%Y-%m-%d %H:%M:%S.%f"
    # start_time=1690761600
    end_time = SingleScenarioHandler.analytics_data_base.digital_twin_df.loc[
        SingleScenarioHandler.analytics_data_base.digital_twin_df['End Time'].notnull(), 'End Time']
    end_time = max(end_time)

    # end_time=1690847999

    # --------------------------------------ORDER---------------------------------------------------------------------------
    def test_get_reference_value_by_id_order(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_reference_value_by_id([34427], 'ORDER')
        referenz = referenz[34427]
        self.assertEqual(referenz, 'Woo34427', 'Reference value by id is wrong')

        referenz = kpi_admin.get_reference_value_by_id([34428], 'ORDER')
        referenz = referenz[34428]
        self.assertEqual(referenz, 'Lav34428', 'Reference value by id is wrong')

    def test_get_customer_name_by_id_order(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_customer_name_by_id([34427], 'ORDER')
        referenz = referenz['Customer Name'][34427]
        self.assertEqual(referenz, 'Woodrow Justin', 'Customer Name by id is wrong')

        referenz = kpi_admin.get_customer_name_by_id([34426], 'ORDER')
        referenz = referenz['Customer Name'][34426]
        self.assertEqual(referenz, 'Tana Burgleih', 'Customer Name by id is wrong')

    def test_get_amount_objects_order(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_amount_objects(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                self.process_ids,
                                                self.resource_ids, "ACTUAL", 'ORDER', True, 'base_general', False)
        referenz_one = referenz[34427]
        referenz_two = referenz[34430]

        self.assertEqual(referenz_one, 2, 'get amount objects is wrong')
        self.assertEqual(referenz_two, 1, 'get amount objects is wrong')

    def test_get_relativ_objects_order(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        part_id = self.part_ids
        liste = list(range(212, 250, 1))
        part_ids = list(set(liste) & set(part_id))
        referenz = kpi_admin.get_relative_objects(self.start_time, self.end_time, self.order_ids, part_ids,
                                                  self.process_ids,
                                                  self.resource_ids, "ACTUAL", 'ORDER', 'base_general', False)

        referenz_one = referenz[34428]  # 100
        referenz_two = referenz[34426]  # 50

        self.assertEqual(referenz_one, 100, 'get relativ objects is wrong')
        self.assertEqual(referenz_two, 50, 'get relativ objects is wrong')

    def test_start_end_time_order_order(self):
        kpi_admin = SingleScenarioHandler.lead_time
        referenz = kpi_admin.get_start_end_time_order(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                      self.process_ids,
                                                      self.resource_ids, "ACTUAL", 'ORDER', 'base_general', False)
        referenz_one = referenz['Start Time [s]'][34426]
        referenz_two = referenz['End Time [s]'][34427]

        self.assertEqual(referenz_one, 1691689930, 'start_end_time is wrong')
        self.assertEqual(referenz_two, 1691696777, 'start_end_time is wrong')

    def test_get_delivery_reliability_order(self):
        kpi_admin = SingleScenarioHandler.delivery_reliability
        referenz = kpi_admin.get_delivery_reliability(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                      self.process_ids,
                                                      self.resource_ids, 'ORDER', 'base_general', False)
        referenz_one = referenz[34426]
        referenz_two = referenz[34427]

        self.assertEqual(referenz_one, 0, 'delivery_reliability is wrong')
        self.assertEqual(referenz_two, 0, 'delivery_reliability is wrong')

    def test_get_delivery_delay_order(self):
        kpi_admin = SingleScenarioHandler.delivery_reliability
        referenz = kpi_admin.get_delivery_delay(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                self.process_ids,
                                                self.resource_ids, 'ORDER', 'base_general', False)

        referenz_one = referenz[34426]
        referenz_two = referenz[34429]

        self.assertEqual(referenz_one, 29837812, 'delivery_delay is wrong')
        self.assertEqual(referenz_two, 0, 'delivery_delay is wrong')

    def test_get_lead_time_order(self):
        kpi_admin = SingleScenarioHandler.lead_time
        referenz = kpi_admin.get_lead_time(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                           self.process_ids,
                                           self.resource_ids, "ACTUAL", 'ORDER', 'base_general', False)
        referenz_one = referenz['total_lead_time_wt'][34426]
        referenz_two = referenz['var_lead_time_wt'][34427]

        self.assertEqual(referenz_one, 13211, 'lead_time is wrong')
        self.assertEqual(referenz_two, 11407192.333333332, 'lead_time is wrong')

    def test_get_quality_order(self):
        kpi_admin = SingleScenarioHandler.quality
        referenz = kpi_admin.get_quality(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                         self.process_ids,
                                         self.resource_ids, "ACTUAL", 'ORDER', 'base_general', False)

        referenz_one = referenz['Resulting Quality'][34426]
        referenz_two = referenz['Resulting Quality'][34427]

        self.assertEqual(referenz_one, 100, 'quality is wrong')
        self.assertEqual(referenz_two, 100, 'quality is wrong')

    def test_get_performance_order(self):
        kpi_admin = SingleScenarioHandler.performance
        referenz = kpi_admin.get_performance(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, 'ORDER', 'base_general', False)

        referenz_one = referenz[34426]
        referenz_two = referenz[34427]

        self.assertEqual(referenz_one, 92.52163708376119, 'performance is wrong')
        self.assertEqual(referenz_two, 92.03633870414811, 'performance is wrong')

    # ---------------------------Order Summory------------------------------------------------------------------------------
    def test_get_amount_objects_order_summary(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_amount_objects(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                self.process_ids,
                                                self.resource_ids, "ACTUAL", 'ORDER', True, 'base_general', True)
        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 5, 'get amount objects is wrong')

    def test_get_relativ_objects_order_summary(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        part_id = self.part_ids
        liste = list(range(212, 250, 1))
        part_ids = list(set(liste) & set(part_id))
        referenz = kpi_admin.get_relative_objects(self.start_time, self.end_time, self.order_ids, part_ids,
                                                  self.process_ids,
                                                  self.resource_ids, "ACTUAL", 'ORDER', 'base_general', True)

        referenz_one = referenz[0]  # 100

        self.assertEqual(referenz_one, 40, 'get relativ objects is wrong')

    def test_start_end_time_order_order_summary(self):
        kpi_admin = SingleScenarioHandler.lead_time
        referenz = kpi_admin.get_start_end_time_order(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                      self.process_ids,
                                                      self.resource_ids, "ACTUAL", 'ORDER', 'base_general', True)
        referenz_one = referenz['Start Time [s]'][0]
        referenz_two = referenz['End Time [s]'][0]

        self.assertEqual(referenz_one, 1691689929, 'start_end_time is wrong')
        self.assertEqual(referenz_two, 1691696777, 'start_end_time is wrong')

    def test_get_delivery_reliability_order_summary(self):
        kpi_admin = SingleScenarioHandler.delivery_reliability
        referenz = kpi_admin.get_delivery_reliability(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                      self.process_ids,
                                                      self.resource_ids, 'ORDER', 'base_general', True)
        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 0, 'delivery_reliability is wrong')

    def test_get_delivery_delay_order_summary(self):
        kpi_admin = SingleScenarioHandler.delivery_reliability
        referenz = kpi_admin.get_delivery_delay(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                self.process_ids,
                                                self.resource_ids, 'ORDER', 'base_general', True)

        self.assertEqual(referenz, 14918602.75, 'delivery_delay is wrong')

    def test_get_lead_time_order_summary(self):
        kpi_admin = SingleScenarioHandler.lead_time
        referenz = kpi_admin.get_lead_time(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                           self.process_ids,
                                           self.resource_ids, "ACTUAL", 'ORDER', 'base_general', True)
        referenz_one = referenz['total_lead_time_wt'][0]
        referenz_two = referenz['var_lead_time_wt'][0]

        self.assertEqual(referenz_one, 33504, 'lead_time is wrong')
        self.assertEqual(referenz_two, 26327866, 'lead_time is wrong')

    def test_get_quality_order_summary(self):
        kpi_admin = SingleScenarioHandler.quality
        referenz = kpi_admin.get_quality(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                         self.process_ids,
                                         self.resource_ids, "ACTUAL", 'ORDER', 'base_general', True)

        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 100, 'quality is wrong')

    def test_get_performance_order_summary(self):
        kpi_admin = SingleScenarioHandler.performance
        referenz = kpi_admin.get_performance(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, 'ORDER', 'base_general', True)

        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 94.3709327548807, 'performance is wrong')

    # ---------------------------Product-------------------------------------------------------------------------------------

    def test_get_reference_value_by_id_product(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_reference_value_by_id(self.part_ids, 'PRODUCT')

        referenz_one = referenz[243]
        referenz_two = referenz[210]
        self.assertEqual(referenz_one, 'Silikonentferner', 'reference_value_by_id is wrong')
        self.assertEqual(referenz_two, 'Salsa Cutthroat Rahmenkit', 'refernece_value_by_id is wrong')

    def test_get_amount_objects_product(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_amount_objects(self.start_time, self.end_time,
                                                self.order_ids, self.part_ids, self.process_ids,
                                                self.resource_ids,
                                                "ACTUAL", 'PRODUCT', True, 'base_gerneral', False)
        referenz_one = referenz[210]
        referenz_two = referenz[214]

        self.assertEqual(referenz_one, 2, 'amount_objects is wrong')
        self.assertEqual(referenz_two, 1, 'amount_objects is wrong')

    def test_get_relative_objects_product(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_relative_objects(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                  self.process_ids,
                                                  self.resource_ids, "ACTUAL", 'PRODUCT', 'base_general', False)
        referenz_one = referenz[210]
        referenz_two = referenz[214]

        self.assertEqual(referenz_one, 100, 'relative_objects is wrong')
        self.assertEqual(referenz_two, 100, 'relative_objects is wrong')

    def test_get_target_quantity_product(self):
        kpi_admin = SingleScenarioHandler.quality
        referenz = kpi_admin.get_quality(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                         self.process_ids,
                                         self.resource_ids, "ACTUAL", 'PRODUCT', 'base_general', False)

        referenz_one = referenz['Resulting Quality'][210]
        referenz_two = referenz['Resulting Quality'][214]

        self.assertEqual(referenz_one, 100, 'quality is wrong')
        self.assertEqual(referenz_two, 100, 'quality is wrong')

    def test_get_difference_percentage_product(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_difference_percentage(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                       self.process_ids,
                                                       self.resource_ids, 'ACTUAL', 'PRODUCT', 'base_general', False)
        referenz_one = referenz[210]
        referenz_two = referenz[218]

        self.assertEqual(referenz_one, 100, 'differnce_percentage is wrong')
        self.assertEqual(referenz_two, 0, 'difference_percentage is wrong')

    def test_get_delivery_reliability_product(self):
        kpi_admin = SingleScenarioHandler.delivery_reliability
        referenz = kpi_admin.get_delivery_reliability(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                      self.process_ids,
                                                      self.resource_ids, 'PRODUCT', 'base_general', False)
        referenz_one = referenz[210]
        referenz_two = referenz[218]

        self.assertEqual(referenz_one, 0, 'delivery_reliability is wrong')
        self.assertEqual(referenz_two, 0, 'delivery_reliability is wrong')

    def test_get_lead_time_product(self):
        kpi_admin = SingleScenarioHandler.lead_time
        referenz = kpi_admin.get_lead_time(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                           self.process_ids,
                                           self.resource_ids, "ACTUAL", 'PRODUCT', 'base_general', False)
        referenz_one = referenz['total_lead_time_wt'][210]
        referenz_two = referenz['var_lead_time_wt'][241]

        self.assertEqual(referenz_one, 14018, 'lead_time is wrong')
        self.assertEqual(referenz_two, 176854.3, 'lead_time is wrong')

    def test_get_performance_product(self):
        kpi_admin = SingleScenarioHandler.performance
        referenz = kpi_admin.get_performance(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, 'PRODUCT', 'base_general', False)

        referenz_one = referenz[210]
        referenz_two = referenz[214]

        self.assertEqual(referenz_one, 96.35311871227366, 'performance is wrong')
        self.assertEqual(referenz_two, 90.00000000000001, 'performance is wrong')

    # ---------------------------Product summary----------------------------------------------------------------------------

    def test_get_amount_objects_product_summary(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_amount_objects(self.start_time, self.end_time,
                                                self.order_ids, self.part_ids, self.process_ids,
                                                self.resource_ids,
                                                "ACTUAL", 'PRODUCT', True, 'base_gerneral', True)
        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 5, 'amount_objects is wrong')

    def test_get_relative_objects_product_summary(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_relative_objects(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                  self.process_ids,
                                                  self.resource_ids, "ACTUAL", 'PRODUCT', 'base_general', True)
        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 100, 'relative_objects is wrong')

    def test_get_target_quantity_product_summary(self):
        kpi_admin = SingleScenarioHandler.quality
        referenz = kpi_admin.get_quality(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                         self.process_ids,
                                         self.resource_ids, "ACTUAL", 'PRODUCT', 'base_general', True)

        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 100, 'quality is wrong')

    def test_get_difference_percentage_product_summary(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_difference_percentage(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                       self.process_ids,
                                                       self.resource_ids, 'ACTUAL', 'PRODUCT', 'base_general', True)
        referenz_one = referenz['All']

        self.assertEqual(referenz_one, 100, 'differnce_percentage is wrong')

    def test_get_delivery_reliability_product_summary(self):
        kpi_admin = SingleScenarioHandler.delivery_reliability
        referenz = kpi_admin.get_delivery_reliability(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                      self.process_ids,
                                                      self.resource_ids, 'PRODUCT', 'base_general', True)
        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 0, 'delivery_reliability is wrong')

    def test_get_lead_time_product_summary(self):
        kpi_admin = SingleScenarioHandler.lead_time
        referenz = kpi_admin.get_lead_time(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                           self.process_ids,
                                           self.resource_ids, "ACTUAL", 'PRODUCT', 'base_general', True)

        referenz_one = referenz['total_lead_time_wt'][0]
        referenz_two = referenz['var_lead_time_wt'][0]

        self.assertEqual(referenz_one, 8376, 'lead_time is wrong')
        self.assertEqual(referenz_two, 8301298.722222221, 'lead_time is wrong')

    def test_get_performance_product_summary(self):
        kpi_admin = SingleScenarioHandler.performance
        referenz = kpi_admin.get_performance(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, 'PRODUCT', 'base_general', True)

        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 93.35788929125714, 'performance is wrong')

    # ------------------------------PROCESS---------------------------------------------------------------------------------

    def test_get_reference_value_by_id_process(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_reference_value_by_id(self.process_ids, 'PROCESS')

        referenz_one = referenz[36030]
        referenz_two = referenz[36024]
        self.assertEqual(referenz_one, 'transport access', 'reference_value_by_id is wrong')
        self.assertEqual(referenz_two, 'Material part loading', 'refernece_value_by_id is wrong')

    def test_get_amount_pe_process(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_amount_pe(self.start_time, self.end_time,
                                           self.order_ids, self.part_ids, self.process_ids,
                                           self.resource_ids,
                                           "ACTUAL", 'PROCESS', True, 'base_gerneral', False)
        referenz_one = referenz[36030]
        referenz_two = referenz[36024]

        self.assertEqual(referenz_one, 0, 'amount_pe is wrong')
        self.assertEqual(referenz_two, 168, 'amount_pe is wrong')

    def test_get_relative_pe_process(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_relative_pe(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, "ACTUAL", 'PROCESS', 'base_general', False)
        referenz_one = referenz[36024]
        referenz_two = referenz[36032]

        self.assertEqual(referenz_one, 100, 'relative_objects is wrong')
        self.assertEqual(referenz_two, 100, 'relative_objects is wrong')

    def test_get_delivery_reliability_process(self):
        kpi_admin = SingleScenarioHandler.delivery_reliability
        referenz = kpi_admin.get_delivery_reliability(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                      self.process_ids,
                                                      self.resource_ids, 'PROCESS', 'base_general', False)
        referenz_one = referenz[36007]
        referenz_two = referenz[36030]

        self.assertEqual(referenz_one, 100, 'delivery_reliability is wrong')
        self.assertEqual(referenz_two, 0, 'delivery_reliability is wrong')

    def test_get_lead_time_process(self):
        kpi_admin = SingleScenarioHandler.lead_time
        referenz = kpi_admin.get_lead_time(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                           self.process_ids,
                                           self.resource_ids, "ACTUAL", 'PROCESS', 'base_general', False)
        referenz_one = referenz['total_lead_time_wt'][36031]
        referenz_two = referenz['var_lead_time_wt'][36032]

        self.assertEqual(referenz_one, 3297, 'lead_time is wrong')
        self.assertEqual(referenz_two, 111.24721984602223, 'lead_time is wrong')

    def test_get_quality_process(self):
        kpi_admin = SingleScenarioHandler.quality
        referenz = kpi_admin.get_quality(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                         self.process_ids,
                                         self.resource_ids, "ACTUAL", 'PROCESS', 'base_general', False)

        referenz_one = referenz['Resulting Quality'][36007]
        referenz_two = referenz['Resulting Quality'][36008]

        self.assertEqual(referenz_one, 100, 'quality is wrong')
        self.assertEqual(referenz_two, 100, 'quality is wrong')

    def test_get_performance_process(self):
        kpi_admin = SingleScenarioHandler.performance
        referenz = kpi_admin.get_performance(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, 'PROCESS', 'base_general', False)

        referenz_one = referenz[36007]
        referenz_two = referenz[36699]

        self.assertEqual(referenz_one, 90.00000000000999, 'performance is wrong')
        self.assertEqual(referenz_two, 0, 'performance is wrong')

    # ---------------------------Process Summary----------------------------------------------------------------------------
    def test_get_amount_pe_process_summary(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_amount_pe(self.start_time, self.end_time,
                                           self.order_ids, self.part_ids, self.process_ids,
                                           self.resource_ids,
                                           "ACTUAL", 'PROCESS', True, 'base_gerneral', True)
        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 755, 'amount_pe is wrong')

    def test_get_relative_pe_process_summary(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_relative_pe(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, "ACTUAL", 'PROCESS', 'base_general', True)
        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 100, 'relative_objects is wrong')

    def test_get_delivery_reliability_process_summary(self):
        kpi_admin = SingleScenarioHandler.delivery_reliability
        referenz = kpi_admin.get_delivery_reliability(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                      self.process_ids,
                                                      self.resource_ids, 'PROCESS', 'base_general', True)
        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 100, 'delivery_reliability is wrong')

    def test_get_lead_time_process_summary(self):
        kpi_admin = SingleScenarioHandler.lead_time
        referenz = kpi_admin.get_lead_time(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                           self.process_ids,
                                           self.resource_ids, "ACTUAL", 'PROCESS', 'base_general', True)

        referenz_one = referenz['total_lead_time_wt'][0]
        referenz_two = referenz['var_lead_time_wt'][0]

        self.assertEqual(referenz_one, 403.0416666666667, 'lead_time is wrong')
        self.assertEqual(referenz_two, 8551.43766613152, 'lead_time is wrong')

    def test_get_quality_process_summary(self):
        kpi_admin = SingleScenarioHandler.quality
        referenz = kpi_admin.get_quality(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                         self.process_ids,
                                         self.resource_ids, "ACTUAL", 'PROCESS', 'base_general', True)

        referenz_one = referenz[0]
        self.assertEqual(referenz_one, 100, 'quality is wrong')

    def test_get_performance_process_summary(self):
        kpi_admin = SingleScenarioHandler.performance
        referenz = kpi_admin.get_performance(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, 'PROCESS', 'base_general', True)

        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 93.81716879879271, 'performance is wrong')

    # ----------------------------Resource----------------------------------------------------------------------------------

    def test_get_reference_value_by_id_resource(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_reference_value_by_id(self.resource_ids, 'RESOURCE')

        referenz_one = referenz[110]
        referenz_two = referenz[5324]
        self.assertEqual(referenz_one, 'Main Part AGV 8', 'reference_value_by_id is wrong')
        self.assertEqual(referenz_two, 'Main Warehouse', 'refernece_value_by_id is wrong')

    def test_get_amount_pe_resource(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_amount_pe(self.start_time, self.end_time,
                                           self.order_ids, self.part_ids, self.process_ids,
                                           self.resource_ids,
                                           "ACTUAL", 'RESOURCE', True, 'base_gerneral', False)
        referenz_one = referenz[5324]
        referenz_two = referenz[5094]

        self.assertEqual(referenz_one, 187, 'amount_pe is wrong')
        self.assertEqual(referenz_two, 2, 'amount_pe is wrong')

    def test_get_relative_pe_resource(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_relative_pe(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, "ACTUAL", 'RESOURCE', 'base_general', False)
        referenz_one = referenz[5324]
        referenz_two = referenz[5094]

        self.assertEqual(referenz_one, 100, 'relative_pe is wrong')
        self.assertEqual(referenz_two, 100, 'relative_pe is wrong')

    def test_get_delivery_reliability_resource(self):
        kpi_admin = SingleScenarioHandler.delivery_reliability
        referenz = kpi_admin.get_delivery_reliability(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                      self.process_ids,
                                                      self.resource_ids, 'RESOURCE', 'base_general', False)
        referenz_one = referenz[98]
        referenz_two = referenz[162]

        self.assertEqual(referenz_one, 100, 'delivery_reliability is wrong')
        self.assertEqual(referenz_two, 100, 'delivery_reliability is wrong')

    def test_get_lead_time_resource(self):
        kpi_admin = SingleScenarioHandler.lead_time
        referenz = kpi_admin.get_lead_time(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                           self.process_ids,
                                           self.resource_ids, "ACTUAL", 'RESOURCE', 'base_general', False)
        referenz_one = referenz['total_lead_time_wt'][126]
        referenz_two = referenz['var_lead_time_wt'][158]

        self.assertEqual(referenz_one, 6730, 'lead_time is wrong')
        self.assertEqual(referenz_two, 213.70624654505255, 'lead_time is wrong')

    def test_get_quality_resource(self):
        kpi_admin = SingleScenarioHandler.quality
        referenz = kpi_admin.get_quality(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                         self.process_ids,
                                         self.resource_ids, "ACTUAL", 'RESOURCE', 'base_general', False)

        referenz_one = referenz['Resulting Quality'][98]
        referenz_two = referenz['Resulting Quality'][188]

        self.assertEqual(referenz_one, 100, 'quality is wrong')
        self.assertEqual(referenz_two, 100, 'quality is wrong')

    def test_get_performance_resource(self):
        kpi_admin = SingleScenarioHandler.performance
        referenz = kpi_admin.get_performance(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, 'RESOURCE', 'base_general', False)

        referenz_one = referenz[120]
        referenz_two = referenz[104]

        self.assertEqual(referenz_one, 90.00000000000017, 'performance is wrong')
        self.assertEqual(referenz_two, 100.21777216427641, 'performance is wrong')

    def test_get_utilisation_resource(self):
        kpi_admin = SingleScenarioHandler.availability
        referenz = kpi_admin.get_utilisation(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, "ACTUAL", 'RESOURCE', 'base_general', False)

        referenz_one = referenz[98]
        referenz_two = referenz[156]

        self.assertEqual(referenz_one, 49.404101995565405, 'utilisation is wrong')
        self.assertEqual(referenz_two, 5.7649667405764955, 'utilisation is wrong')

    def test_get_availability_resource(self):
        kpi_admin = SingleScenarioHandler.availability
        SingleScenarioHandler.analytics_data_base.start_time_observation_period = self.start_time
        SingleScenarioHandler.analytics_data_base.end_time_observation_period = self.end_time
        referenz = kpi_admin.get_availability(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                              self.process_ids,
                                              self.resource_ids, "ACTUAL", 'RESOURCE', 'base_general', False)

        referenz_one = referenz[110]
        referenz_two = referenz[5324]

        self.assertEqual(referenz_one, 96.21674057649666, 'availability is wrong')
        self.assertEqual(referenz_two, 3.215077605321507, 'availability is wrong')

    def test_get_ore_resource(self):
        kpi_admin = SingleScenarioHandler.ore
        kpi_quality = SingleScenarioHandler.quality.get_quality(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                                self.process_ids,
                                                                self.resource_ids, "ACTUAL", 'RESOURCE', 'base_general', False)
        kpi_performance = SingleScenarioHandler.performance.get_performance(self.start_time, self.end_time, self.order_ids,
                                                                            self.part_ids,
                                                                            self.process_ids,
                                                                            self.resource_ids, 'RESOURCE', 'base_general',
                                                                            False)
        kpi_availability = SingleScenarioHandler.availability.get_availability(self.start_time, self.end_time, self.order_ids,
                                                                               self.part_ids,
                                                                               self.process_ids,
                                                                               self.resource_ids, "ACTUAL", 'RESOURCE',
                                                                       'base_general', False)
        referenz = kpi_admin.get_ore(kpi_quality,
                                     kpi_performance,
                                     kpi_availability)

        referenz_one = referenz[98]
        referenz_two = referenz[5206]

        self.assertEqual(referenz_one, 87.76333992094877, 'ore is wrong')
        self.assertEqual(referenz_two, 0.07483370288249167, 'ore is wrong')

    # --------------------Resource Summary-----------------------------------------------------------------------------------
    def test_get_amount_pe_resource_summary(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_amount_pe(self.start_time, self.end_time,
                                           self.order_ids, self.part_ids, self.process_ids,
                                           self.resource_ids,
                                           "ACTUAL", 'RESOURCE', True, 'base_gerneral', True)
        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 1238, 'amount_pe is wrong')

    def test_get_relative_pe_resource_summary(self):
        kpi_admin = SingleScenarioHandler.analytics_data_base_administration
        referenz = kpi_admin.get_relative_pe(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, "ACTUAL", 'RESOURCE', 'base_general', True)
        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 100, 'relative_pe is wrong')

    def test_get_delivery_reliability_resource_summary(self):
        kpi_admin = SingleScenarioHandler.delivery_reliability
        referenz = kpi_admin.get_delivery_reliability(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                      self.process_ids,
                                                      self.resource_ids, 'RESOURCE', 'base_general', True)
        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 100, 'delivery_reliability is wrong')

    def test_get_lead_time_resource_summary(self):
        kpi_admin = SingleScenarioHandler.lead_time
        referenz = kpi_admin.get_lead_time(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                           self.process_ids,
                                           self.resource_ids, "ACTUAL", 'RESOURCE', 'base_general', True)
        referenz_one = referenz['total_lead_time_wt'][0]
        referenz_two = referenz['var_lead_time_wt'][0]

        self.assertEqual(referenz_one, 777.6741573033707, 'lead_time is wrong')
        self.assertEqual(referenz_two, 5785.337353652228, 'lead_time is wrong')

    def test_get_quality_resource_summary(self):
        kpi_admin = SingleScenarioHandler.quality
        referenz = kpi_admin.get_quality(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                         self.process_ids,
                                         self.resource_ids, "ACTUAL", 'RESOURCE', 'base_general', True)

        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 100, 'quality is wrong')

    def test_get_performance_resource_summary(self):
        kpi_admin = SingleScenarioHandler.performance
        referenz = kpi_admin.get_performance(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, 'RESOURCE', 'base_general', True)

        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 91.11024075711077, 'performance is wrong')

    def test_get_utilisation_resource_summary(self):
        kpi_admin = SingleScenarioHandler.availability
        referenz = kpi_admin.get_utilisation(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                             self.process_ids,
                                             self.resource_ids, "ACTUAL", 'RESOURCE', 'base_general', True)

        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 5.5947769002715555, 'utilisation is wrong')

    def test_get_availability_resource_summary(self):
        kpi_admin = SingleScenarioHandler.availability
        SingleScenarioHandler.kpi_administration.start_time_observation_period = self.start_time
        SingleScenarioHandler.kpi_administration.end_time_observation_period = self.end_time
        referenz = kpi_admin.get_availability(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                              self.process_ids,
                                              self.resource_ids, "ACTUAL", 'RESOURCE', 'base_general', True)

        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 23.434346894541463, 'availability is wrong')

    def test_get_ore_resource_summary(self):
        kpi_admin = SingleScenarioHandler.ore
        kpi_quality = SingleScenarioHandler.quality.get_quality(self.start_time, self.end_time, self.order_ids, self.part_ids,
                                                                self.process_ids,
                                                                self.resource_ids, "ACTUAL", 'RESOURCE', 'base_general', True)
        kpi_performance = SingleScenarioHandler.performance.get_performance(self.start_time, self.end_time, self.order_ids,
                                                                            self.part_ids,
                                                                            self.process_ids,
                                                                            self.resource_ids, 'RESOURCE', 'base_general', True)
        kpi_availability = SingleScenarioHandler.availability.get_availability(self.start_time, self.end_time, self.order_ids,
                                                                               self.part_ids,
                                                                               self.process_ids,
                                                                               self.resource_ids, "ACTUAL", 'RESOURCE',
                                                                       'base_general', True)
        referenz = kpi_admin.get_ore(kpi_quality,
                                     kpi_performance,
                                     kpi_availability)

        referenz_one = referenz[0]

        self.assertEqual(referenz_one, 21.351089875473235, 'ore is wrong')


if __name__ == '__main__':
    unittest.main()
