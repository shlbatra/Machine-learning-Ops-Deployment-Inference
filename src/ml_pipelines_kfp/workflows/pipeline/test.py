import google.cloud.aiplatform as aip

from kfp import dsl, compiler

from src.workflows.configs.config import get_base_pipeline_config