"""
The main is the start point of the "Personal Einsatzplanung Tool"/ workforce planning for the Schmaus company.
@last update: ?.?.2022
"""
import asyncio

from ofact.twin.agent_control.behaviours.basic import DigitalTwinCyclicBehaviour
from ofact.twin.agent_control.basic import DigitalTwinAgent
from ofact.twin.agent_control.helpers.communication_objects import CommunicationObject

print("STATUS: PIP install finished")
print("STATUS: docker build finished")

# Imports Part 1: Standard Imports
import sys
import time

print("STATUS: Import Part 1 finished")
print("STATUS DEBUG - sys.path:", sys.path)

# Imports Part 2: PIP Imports
import ofrestapi
import spade

print("STATUS: Import Part 2 finished")

# Imports Part 3: Project Imports
# from projects.bicycle_world.business.dashboard import app
from ofact.twin.agent_control.organization import Agents
from projects.Schmaus.settings import XMPP_SERVER_IP_ADDRESS, XMPP_SERVER_REST_API_PORT, \
    XMPP_SERVER_SHARED_SECRET, XMPP_SERVER_REST_API_USERS_ENDPOINT

print("STATUS: Import Part 3 finished")


def create_agents(xmpp_user_manager, agents: dict, async_start: bool = True) -> object:
    """
    The method to initialize a given dict of Agents.
    :param agents: dict which contains information of the agents.
    :param async_start: whether the agents should be created now or asynchronously
    :return: List of agents (/ futures if agents are created async)
    """
    created_agents = []
    for agent_type, agent_object_lst in agents.items():
        for agent_object in agent_object_lst:
            print(f"[main] Start agent {agent_object.name} \n")

            # create XMPP account
            xmpp_user_manager.add_user(username=agent_object.name, password=agent_object.password)

            agent_future = agent_object.start()
            if async_start:
                agent_future.result()
            created_agents.append(agent_future)
    return created_agents


class DigitalTwinTestAgent(DigitalTwinAgent):

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str):
        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server,
                         ip_address_xmpp_server=ip_address_xmpp_server)

    class ReceiverBehaviour(DigitalTwinCyclicBehaviour):

        def __init__(self):
            super().__init__()
            self.t = 0
            self.t1 = None

        async def run(self):
            await super().run()
            if self.t == 0:
                self.t1 = time.process_time()

            metadata_conditions_lst = [{"performative": "request", "ontology": "PROPOSAL"}]

            msg = await self.agent.receive_msg(self, timeout=10, metadata_conditions_lst=metadata_conditions_lst)
            self.t += 1
            if self.t == 100:
                t2 = time.process_time()
                print("Receiver time", t2 - self.t1)

    class SenderBehaviour(DigitalTwinCyclicBehaviour):

        async def run(self):
            await super().run()
            await asyncio.sleep(5)

            t1 = time.process_time()
            receivers = ["one", "two"]
            receivers.remove(self.agent.name)
            for i in range(100):
                communication_object = CommunicationObject()
                msg_content = communication_object

                await self.agent.send_msg(behaviour=self, receiver_list=receivers, msg_body=msg_content,
                                          message_metadata={"performative": "request",
                                                            "ontology": "PROPOSAL",
                                                            "language": "OWL-S"})
            t2 = time.process_time()
            print("Sender time", t2 - t1)

            await asyncio.sleep(1000)

    async def setup(self):
        print(f"[{self.name}] Hello World! I'm agent {str(self.jid)} \n {type(self).__name__} \n")
        templates = [{"metadata": {"performative": "request", "ontology": "PROPOSAL", "language": "OWL-S"}},
                     {"metadata": {"performative": "response", "ontology": "PROPOSAL", "language": "OWL-S"}}]

        templates = [self.create_template(template) for template in templates]

        self.ReceiverBehaviour = type(self).ReceiverBehaviour()
        self.SenderBehaviour = type(self).SenderBehaviour()

        self.add_behaviour(self.ReceiverBehaviour, templates[0] ^ templates[1])
        self.add_behaviour(self.SenderBehaviour)


def main():
    agents_model = Agents()
    agent_names = ["one", "two"]  # , "three", "four", "six", "seven", "eight", "nine", "ten"]
    agent_objects = [DigitalTwinTestAgent(name=agent_name, organization=agents_model, change_handler=None,
                                          password_xmpp_server="LCtBjPge9y6fCyjb",
                                          ip_address_xmpp_server="127.0.0.1")
                     for agent_name in agent_names]
    agents = {}
    for agent_object in agent_objects:
        agents.setdefault(agent_object.__class__, []).append(agent_object)
    agents_model.agents = agents

    # ==== initialize the REST-API of the Openfire-XMPP Server =========================================================
    xmpp_user_manager = ofrestapi.Users(host="http://" + XMPP_SERVER_IP_ADDRESS + XMPP_SERVER_REST_API_PORT,
                                        secret=XMPP_SERVER_SHARED_SECRET,
                                        endpoint=XMPP_SERVER_REST_API_USERS_ENDPOINT)

    # ==== cleanup the xmpp server =====================================================================================
    print("...[main] removing all existing users from the xmpp-server")
    all_old_xmpp_users = xmpp_user_manager.get_users()
    print("all_old_xmpp_users", all_old_xmpp_users)
    for user in all_old_xmpp_users["users"]:
        current_username = user["username"]
        if current_username != "admin":
            print(f"\t[main] User {current_username} delete")
            xmpp_user_manager.delete_user(current_username)

    # ==== create agents ===============================================================================================

    all_agents = create_agents(xmpp_user_manager, agents_model.agents)
    print("all_agents", all_agents)

    # ==== wait until user stops the execution =========================================================================
    print("[main] Running until user interrupts with ctrl+C")

    while True:
        try:
            time.sleep(1)  # TODO Questionable
        except KeyboardInterrupt:
            # if the program is exited we want to create sink agents to write the data to the database
            sink_agent = create_agents(xmpp_user_manager, agents_model.sink_agent)
            all_agents.append(sink_agent)
            break

    # === stop the program =============================================================================================
    print("[main] stopping Agents...")
    for agent in all_agents:
        xmpp_user_manager.delete_user(username=agent.name)
        print(f">stopping agent '{agent.name}'...")
        agent.stop()

    print("[main] all Agents stopped, closing!")
    spade.quit_spade()


if __name__ == "__main__":
    print("STATUS: Starting Execution")
    main()
