"""
The administration is used to create agents and connect them with the xmpp server after deleting the old ones.
In addition, it is ensured that all agents are stopped at the end of the simulation.
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
# import ofrestapi

# Imports Part 3: Project Imports
if TYPE_CHECKING:
    from ofact.twin.agent_control.basic import DigitalTwinAgent


async def create_agents(agents: dict) -> list[DigitalTwinAgent]:
    """
    The method to initialize a given dict of Agents.
    :param agents: dict which contains information of the agent_control.
    :param async_start: whether the agent_control should be created now or asynchronously
    :return: List of agent_control (/ futures if agent_control are created async)
    """
    created_agents = []
    for agent_type, agent_object_lst in agents.items():
        for agent_object in agent_object_lst:
            print(f"[Agent Administration] Start agent {agent_object.name} \n")
            agent_object: DigitalTwinAgent
            # create XMPP account
            # xmpp_user_manager.add_user(username=agent_object.name, password=agent_object.password)

            agent_future = agent_object.start()
            try:
                await agent_future
            except TimeoutError as te:
                error_possibility_one = "1. Check internet connection (required for the agent communication)!"
                error_possibility_two = "2. Check OpenFire xmpp Server settings!"
                raise TimeoutError(te, "\n", error_possibility_one, "\n", error_possibility_two)

            created_agents.append(agent_future)
    return created_agents


async def start_agent_control(mode, host, secret, endpoint, agent_names, agents_model):
    # ==== initialize the REST-API of the Openfire-XMPP Server =========================================================
    # xmpp_user_manager = ofrestapi.Users(host=host, secret=secret, endpoint=endpoint)

    # ==== cleanup the xmpp server =====================================================================================
    print(f"...[agent administration] removing all existing users from the xmpp-server ({host})")
    # all_old_xmpp_users = xmpp_user_manager.get_users()
    # print("all_old_xmpp_users", all_old_xmpp_users)
    # for user in all_old_xmpp_users["users"]:
    #     current_username = user["username"]
    #     if current_username != "admin" and current_username in agent_names:
    #         print(f"\t[main] User {current_username} delete")
    #         xmpp_user_manager.delete_user(current_username)

    # ==== create agent_control ========================================================================================
    print(f"...[agent administration] Create Agents")
    if mode == "SIMULATION":
        # agents_model.add_xmpp_user_manager(xmpp_user_manager)
        all_agents = await create_agents(agents_model.agents)
        print("all_agents", len(all_agents))

    else:
        raise NotImplementedError

    return all_agents


def stop_agent_control(all_agents):
    # === stop the program =============================================================================================
    print("[agent administration] stopping Agents...")
    for agent in all_agents:
        if agent.cr_running:
            # xmpp_user_manager.delete_user(username=agent.name)
            print(f">stopping agent '{agent.name}'...")
            agent.stop()

    print("[agent administration] all Agents stopped, closing!")
