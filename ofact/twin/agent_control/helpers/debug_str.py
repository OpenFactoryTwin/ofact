from datetime import datetime


def get_debug_str(agent_name, behaviour_name):
    return f"{datetime.now().time()} [{agent_name:35} | {behaviour_name:35}]"
