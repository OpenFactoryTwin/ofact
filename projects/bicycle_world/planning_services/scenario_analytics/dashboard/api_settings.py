API_SET_CURRENT_STATE_AVAILABLE = {"available": True,
                                   "consideration_period": 8}

parameter_agents = {"name": "agents",
                    "display_type": "dropdown",
                    "display_items": [],
                    "default_item": None,
                    "file_content": True,
                    "folder_path": "/scenarios/current/models/agents",
                    "accepted_file_types": ["xlsx", "pkl", "xlsm"]}

parameter_twin = {"name": "twin",
                  "display_type": "dropdown",
                  "display_items": [],
                  "default_item": None,
                  "file_content": True,
                  "folder_path": "/scenarios/current/models/twin",
                  "accepted_file_types": ["xlsx", "pkl", "xlsm"]}

parameter_main_agv_s = {"name": "Main part AGV's",
                        "display_type": "slider",
                        "display_items": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        "default_item": 18,
                        "file_content": False}

parameter_material_agv_s = {"name": "Individual Part AGV's",
                            "display_type": "slider",
                            "display_items": [3, 4, 5],
                            "default_item": 3,
                            "file_content": False}

parameter_simulation_time = {"name": "simulation_time",
                             "display_type": "slider",
                             "display_items": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,
                                               9.5, 10],
                             "default_item": 1,
                             "file_content": False}

API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME = [parameter_agents, parameter_twin,
                                                parameter_main_agv_s, parameter_material_agv_s,
                                                parameter_simulation_time]
