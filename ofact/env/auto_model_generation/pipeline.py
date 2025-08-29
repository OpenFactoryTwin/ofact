"""
Auto Model Generation Pipeline used to execute the modules in the right sequence.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ofact.env.data_integration.admin_pipeline.pipeline_settings import PipelineType
from ofact.env.data_integration.pipeline import ModelGenerationAndIntegrationPipeline

if TYPE_CHECKING:
    from ofact.env.data_integration.preprocessing.preprocessing.preprocessing import Preprocessing
    from ofact.twin.state_model.model import StateModel


class AutoModelGenerationPipeline(ModelGenerationAndIntegrationPipeline):

    def _get_pipeline_type(self):
        return PipelineType.MODEL_GENERATION

    def get_state_model(self,
                        store_interim_results: bool = False) -> StateModel:
        """
        Handle one to n data sources in a pipeline to get one model as output ...

        Parameters
        ----------
        store_interim_results: a boolean to indicate if the interim results should be stored
        """
        return self._execute_generation_or_integration(store_interim_results)



if __name__ == "__main__":
    pass
    # SourceStateModelGeneration()
    # AutoModelGenerationPipeline()
