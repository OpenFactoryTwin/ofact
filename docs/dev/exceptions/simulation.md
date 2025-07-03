> riotana | UserWarning: Conditional Formatting extension is not supported and will be removed </br>
> riotana | ERROR:DigitalTwin.port_interfaces.frontend.dashboard_controller:Exception on /api/v1/simulation/start [POST] </br>
> riotana | Traceback (most recent call last): </br>
> riotana |   File "/usr/local/lib/python3.10/site-packages/ofrestapi/base.py", line 60, in _submit_request </br>
> riotana |     exception = r.json()['exception'] </br>
> riotana | KeyError: 'exception' </br>
> riotana | </br>
> riotana | During handling of the above exception, another exception occurred: </br>
> riotana | </br>
> riotana | Traceback (most recent call last): </br>
> riotana |   File "/usr/local/lib/python3.10/site-packages/flask/app.py", line 1484, in full_dispatch_request </br>
> riotana |     rv = self.dispatch_request() </br>
> riotana |   File "/usr/local/lib/python3.10/site-packages/flask/app.py", line 1469, in dispatch_request </br>
> riotana |     return self.ensure_sync(self.view_functions[rule.endpoint])(**view_args) </br>
> riotana |   File "/usr/local/lib/python3.10/site-packages/flask_restful/__init__.py", line 489, in wrapper </br>
> riotana |     resp = resource(*args, **kwargs) </br>
> riotana |   File "/usr/local/lib/python3.10/site-packages/flask/views.py", line 109, in view </br>
> riotana |     return current_app.ensure_sync(self.dispatch_request)(**kwargs) </br>
> riotana |   File "/usr/local/lib/python3.10/site-packages/flask_restful/__init__.py", line 604, in dispatch_request </br>
> riotana |     resp = meth(*args, **kwargs) </br>
> riotana |   File "/usr/src/app/DigitalTwin/port_interfaces/frontend/dashboard_controller.py", line 638, in post </br>
> riotana |     build_simulation_response(simulation_func=simulation_func, args=validated_args, </br>
> riotana |   File "/usr/src/app/DigitalTwin/port_interfaces/frontend/simulation.py", line 181, in build_simulation_response </br>
> riotana |     asyncio.run(simulation_func(digital_twin=digital_twin_model, </br>
> riotana |   File "/usr/local/lib/python3.10/asyncio/runners.py", line 44, in run </br>
> riotana |     return loop.run_until_complete(main) </br>
> riotana |   File "/usr/local/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete </br>
> riotana |     return future.result() </br>
> riotana |   File "/usr/src/app/DigitalTwin/projects/Schmaus/agent_control/simulate.py", line 44, in simulate </br>
> riotana |     return await super().simulate(digital_twin=digital_twin, agents_file_name=agents_file_name, </br>
> riotana |   File "/usr/src/app/DigitalTwin/application/agent_control/simulate.py", line 72, in simulate </br>
> riotana |     xmpp_user_manager, all_agents = await self._register_agents(environment, agents_model, instance=scenario_name) </br>
> riotana |   File "/usr/src/app/DigitalTwin/application/agent_control/simulate.py", line 182, in _register_agents </br>
> riotana |     await start_agent_control(mode=self.mode, </br>
> riotana |   File "/usr/src/app/DigitalTwin/application/agent_control/administration.py", line 46, in start_agent_control </br>
> riotana |     all_old_xmpp_users = xmpp_user_manager.get_users() </br>
> riotana |   File "/usr/local/lib/python3.10/site-packages/ofrestapi/users.py", line 37, in get_users </br>
> riotana |     return self._submit_request(get, self.endpoint, params=params) </br>
> riotana |   File "/usr/local/lib/python3.10/site-packages/ofrestapi/base.py", line 63, in _submit_request </br>
> riotana |     raise InvalidResponseException(r.status_code) </br>
> riotana | ofrestapi.exception.InvalidResponseException: 403 </br>
> riotana | INFO:werkzeug:192.168.1.32 - - [09/Nov/2023 10:52:01] "POST /api/v1/simulation/start HTTP/1.1" 500 - </br>

Openfire does not work! - check the connection to the Openfire Server!

