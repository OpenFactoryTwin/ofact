"""
# TODO Add Module Description
@last update: ?.?.2022
"""

# Imports Part 1: Standard Imports
import unittest
# Imports Part 2: PIP Imports
import pandas as pd
# Imports Part 3: Project Imports
import ofact.twin.state_model.time as t


class TestWorkCalender(unittest.TestCase):

    def setUp(self):
        start_time_period = t.pd.Timestamp(2021, 3, 1)
        end_time_period = t.pd.Timestamp(2021, 4, 30)

        # company vacation
        company_vacation = [t.pd.Timestamp(2021, 3, 5)]

        work_calendar_parameters = {"amount_shift": 3, "shift_begin": (6, 0),
                                    "shift_format": [(4, 0), (0, 30), (3, 30)],
                                    "week_start": (0, 6), "week_end": (5, 6)}
        # calendar
        self.calendar = t.WorkCalender(name="TestWorkCalendar",
                                       base_unit_freq='T',
                                       start_time=start_time_period,
                                       end_time=end_time_period,
                                       work_calendar_parameters=work_calendar_parameters,
                                       company_vacation=company_vacation)

    def test_get_possible_production_time(self):
        possible_production_time = self.calendar.get_possible_production_time(
            end_time=t.pd.Timestamp(2021, 3, 30, 6, 0, 0),
            duration=t.pd.Timedelta(hours=8))

        self.assertEqual(str(possible_production_time), '7:30:00', "INCORRECT Production Time")

    def check_clash(self):
        clash1 = self.calendar.check_clash(start_time=t.datetime(2021, 3, 29, 22, 0, 0),
                                           end_time=t.datetime(2021, 3, 30, 6, 0, 0))

        clash2 = self.calendar.check_clash(start_time=t.datetime(2021, 3, 30, 5, 0, 0),
                                           end_time=t.datetime(2021, 3, 30, 6, 0, 0))

        self.assertEqual(clash1, True, "INCORRECT WorkCalendar Clashing1")
        self.assertEqual(clash2, False, "INCORRECT WorkCalendar Clashing2")


class TestProcessExecutionPlanner(unittest.TestCase):

    def test_check_clash(self):
        datetime_now = t.datetime.now()
        time_planner = t.ProcessExecutionPlan(name="Test", start_time=datetime_now)
        time_planner.block_period(start_time=datetime_now + t.timedelta(minutes=2),
                                  end_time=datetime_now + t.timedelta(minutes=4),
                                  blocker_name="Herbert",
                                  process_execution_id=42,
                                  production_order_id=32)
        clashing_blocker_names_before, clashing_process_execution_ids_before = \
            time_planner._check_clash(start_time=datetime_now,
                                      end_time=datetime_now + t.timedelta(minutes=2))
        clashing_blocker_names_after, clashing_process_execution_ids_after = \
            time_planner._check_clash(start_time=datetime_now + t.timedelta(minutes=4),
                                      end_time=datetime_now + t.timedelta(minutes=6))
        clashing_blocker_names_in, clashing_process_execution_ids_in = \
            time_planner._check_clash(start_time=datetime_now + t.timedelta(minutes=1),
                                      end_time=datetime_now + t.timedelta(minutes=5))
        clashing_blocker_names_before_overlap, clashing_process_execution_ids_before_overlap = \
            time_planner._check_clash(start_time=datetime_now + t.timedelta(minutes=1),
                                      end_time=datetime_now + t.timedelta(minutes=3))
        clashing_blocker_names_after_overlap, clashing_process_execution_ids_after_overlap = \
            time_planner._check_clash(start_time=datetime_now + t.timedelta(minutes=3),
                                      end_time=datetime_now + t.timedelta(minutes=5))

        self.assertEqual(clashing_blocker_names_before, [None], "INCORRECT clashing before")
        self.assertEqual(clashing_blocker_names_after, [None], "INCORRECT clashing after")
        self.assertEqual(clashing_blocker_names_in, ["Herbert"], "INCORRECT clashing in")
        self.assertEqual(clashing_blocker_names_before_overlap, ["Herbert"], "INCORRECT clashing before_overlap")
        self.assertEqual(clashing_blocker_names_after_overlap, ["Herbert"], "INCORRECT clashing after_overlap")

    def test_block_period(self):
        time_planner = t.ProcessExecutionPlan(name="Test", start_time=t.datetime.now())
        # creation
        time_planner.block_period(start_time=t.datetime.now(),
                                  end_time=t.datetime.now() + t.timedelta(minutes=1),
                                  blocker_name="Herbert",
                                  process_execution_id=42,
                                  production_order_id=32)

        self.assertEqual(time_planner.__time_schedule.shape[0], 1, "INCORRECT Time slot blocking creation")

        # second creation
        time_planner.block_period(start_time=t.datetime.now() + t.timedelta(minutes=1),
                                  end_time=t.datetime.now() + t.timedelta(minutes=2),
                                  blocker_name="Cassandra",
                                  process_execution_id=43,
                                  production_order_id=32)

        self.assertEqual(time_planner.__time_schedule.shape[0], 2, "INCORRECT Time slot blocking second creation")

        # overlap
        time_planner.block_period(start_time=t.datetime.now(),
                                  end_time=t.datetime.now() + t.timedelta(minutes=3),
                                  blocker_name="Dieter",
                                  process_execution_id=44,
                                  production_order_id=32)

        self.assertEqual(time_planner.__time_schedule.shape[0], 2, "INCORRECT Time slot blocking overlap")

        # same id
        time_planner.block_period(start_time=t.datetime.now() + t.timedelta(minutes=3),
                                  end_time=t.datetime.now() + t.timedelta(minutes=4),
                                  blocker_name="Sandra",
                                  process_execution_id=42,
                                  production_order_id=32)

        self.assertEqual(time_planner.__time_schedule.shape[0], 2,
                         "INCORRECT Time slot blocking same process_execution_id")

    def test_unblock_period(self):
        time_planner = t.ProcessExecutionPlan(name="Test",
                                              start_time=t.datetime.now())

        time_planner.block_period(start_time=t.datetime.now(),
                                  end_time=t.datetime.now() + t.timedelta(minutes=1),
                                  blocker_name="Herbert",
                                  process_execution_id=42,
                                  production_order_id=32)

        self.assertEqual(time_planner.__time_schedule.shape[0], 1, "INCORRECT Time slot blocking creation")

        successful = time_planner.unblock_period("Herbert", 43)

        self.assertEqual(successful, False,
                         "INCORRECT Time slot unblocking false process_execution_id")

        successful = time_planner.unblock_period("Herbert", 42)

        self.assertEqual(successful, True, "INCORRECT Time slot unblocking")

    def test_get_next_possible_period(self):
        time_planner = t.ProcessExecutionPlan(name="Test",
                                              start_time=t.datetime.now())
        start_time1 = t.datetime.now()
        end_time1 = start_time1 + t.timedelta(minutes=1)
        time_planner.block_period(start_time=start_time1,
                                  end_time=end_time1,
                                  blocker_name="Herbert",
                                  process_execution_id=42,
                                  production_order_id=32)
        period_length = t.timedelta(minutes=1)
        start_time, end_time = time_planner.get_next_possible_period(period_length=period_length)
        end_time2 = start_time + period_length
        self.assertEqual(start_time, end_time1, "INCORRECT start_time next_possible_period")
        self.assertEqual(end_time, end_time2, "INCORRECT end_time next_possible_period")

    def test_get_blocked_periods_calendar_extract(self):
        time_planner = t.ProcessExecutionPlan(name="Test",
                                              start_time=t.datetime.now())
        start_time1 = t.datetime.now()
        end_time1 = start_time1 + t.timedelta(minutes=1)
        time_planner.block_period(start_time=start_time1,
                                  end_time=end_time1,
                                  blocker_name="Herbert",
                                  process_execution_id=42,
                                  production_order_id=32)
        duration = t.timedelta(minutes=1)
        calendar_extract = time_planner._get_blocked_periods_calendar_extract(duration=duration)
        self.assertEqual(type(calendar_extract), t.np.ndarray, "INCORRECT type")
        self.assertEqual(calendar_extract.shape, (1, 2), "INCORRECT length")

    def test_get_free_periods_calendar_extract(self):
        start_time = t.datetime.now()
        end_time = start_time + t.timedelta(minutes=15)
        time_planner = t.ProcessExecutionPlan(name="Test",
                                              start_time=start_time)


        start_time1 = start_time + t.timedelta(minutes=10)
        end_time1 = start_time1 + t.timedelta(minutes=1)
        time_planner.block_period(start_time=start_time1,
                                  end_time=end_time1,
                                  blocker_name="Herbert",
                                  process_execution_id=42,
                                  production_order_id=32)

        start_time2 = end_time1 + t.timedelta(seconds=30)
        end_time2 = start_time2 + t.timedelta(minutes=2)
        time_planner.block_period(start_time=start_time2,
                                  end_time=end_time2,
                                  blocker_name="Dieter",
                                  process_execution_id=43,
                                  production_order_id=33)

        start_time3 = end_time2 + t.timedelta(minutes=1)
        end_time3 = start_time3 + t.timedelta(minutes=5)
        time_planner.block_period(start_time=start_time3,
                                  end_time=end_time3,
                                  blocker_name="Hans-Peter",
                                  process_execution_id=44,
                                  production_order_id=34)

        time_slot_duration = t.timedelta(minutes=1)
        calendar_extract = time_planner.get_free_periods_calendar_extract(start_time=start_time, end_time=end_time,
                                                                          time_slot_duration=time_slot_duration)

        self.assertEqual(type(calendar_extract), t.np.ndarray, "INCORRECT type")
        self.assertEqual(calendar_extract.shape, (2, 2), "INCORRECT length")
        print(calendar_extract[0,0])


if __name__ == '__main__':
    unittest.main()
