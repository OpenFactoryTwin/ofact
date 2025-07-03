"""
This file encapsulates the behaviours used for the agents and provides their basic functionalities.
The main_purpose are the prints at the beginning of the behaviour, respectively on the end, etc.
see https://spade-mas.readthedocs.io/en/latest/behaviours.html
@last update: 21.08.2023
"""

# Imports Part 1: Standard Imports
import logging
from datetime import datetime

# Imports Part 2: PIP Imports
try:
    from spade.behaviour import CyclicBehaviour, State, FSMBehaviour
except ModuleNotFoundError:
    class CyclicBehaviour:
        pass
    class State:
        pass
    class FSMBehaviour:
        pass
logger = logging.getLogger("basic-behaviour")
# Imports Part 3: Project Imports
# none


def get_templates_tuple(template_lst):
    """convert the template list into the needed format (tuple with ^ links in between)"""
    if not template_lst:
        templates = ()
    elif len(template_lst) == 1:
        templates = (template_lst[0])
    elif len(template_lst) == 2:
        templates = (template_lst[0] ^ template_lst[1])
    elif len(template_lst) == 3:
        templates = (template_lst[0] ^ template_lst[1] ^ template_lst[2])
    elif len(template_lst) == 4:
        templates = (template_lst[0] ^ template_lst[1] ^ template_lst[2] ^ template_lst[3])
    elif len(template_lst) == 5:
        templates = \
            (template_lst[0] ^ template_lst[1] ^ template_lst[2] ^ template_lst[3] ^ template_lst[4])
    else:
        raise NotImplementedError

    return templates


class DigitalTwinCyclicBehaviour(CyclicBehaviour):
    """
    Standard behaviour that is called in a cyclic way. Every time, the run method is ended,
    the run method is called another time
    Mainly the behaviour provides the prints (starting, end).
    """

    async def on_start(self):
        await super().on_start()

        logging.debug(f"{datetime.now()} | [{self.agent.name:35}] Starting behaviour {self.__class__.__name__} "
                      f"from the {self.agent.__class__.__name__}")


    async def run(self):
        # print(f"\n The behavior {__class__.__name__} from the {self.agent.__class__.__name__} is opened")
        pass

    async def on_end(self):
        await super().on_end()
        logging.debug(f"{datetime.now()} | [{self.agent.name:35}] Behaviour {self.__class__.__name__} finished "
                      f"with exit code {self.exit_code}.")
        # await self.agent.stop()


class OrderFSMBehaviour(FSMBehaviour):
    """
    The states are organized in a hierarchy as classes. Via transitions, the flow through the states is organized.
    """

    def __init__(self):
        super().__init__()

    async def on_start(self):
        logging.debug(f"{datetime.now()} | [{self.agent.name:35}] FSM starting at initial state {self.current_state}")

    async def on_end(self):
        logging.debug(f"{datetime.now()} | [{self.agent.name:35}] FSM finished at state {self.current_state}")
        await self.agent.stop()


class DigitalTwinState(State):
    def __init__(self):
        super().__init__()

    async def run(self):
        logging.debug(f"{datetime.now()} | [{self.agent.name:35}] The {self.agent.__class__.__name__} "
                      f"is in state: {__class__.__name__}")
