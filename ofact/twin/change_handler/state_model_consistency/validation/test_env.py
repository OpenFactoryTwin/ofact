import os
from pathlib import Path

from ofact.twin.change_handler.state_model_consistency.validation.process_chain_validation import ProcessChainValidation
from ofact.twin.repository_services.persistence import deserialize_state_model

os.getcwd()
state_model_pkl = Path(os.getcwd().split("ofact")[0], f'projects/Schmaus/data/raw_dt/test.pkl')

state_model = deserialize_state_model(source_file_path=state_model_pkl)

ProcessChainValidation(state_model)
