import logging
import os

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define default paths, these might be overridden by environment variables
DEFAULT_LLM_CONFIG_PATH = "config/llm_config.yaml"
DEFAULT_PROMPT_PATH = "prompts/langchain/negotiation_coach.yaml"


def load_prompt_template(prompt_file_path: str = DEFAULT_PROMPT_PATH) -> str:
    """
    Loads a prompt template from a YAML file.

    Args:
        prompt_file_path: The path to the prompt YAML file.

    Returns:
        The prompt template string.
    """
    try:
        with open(prompt_file_path, "r") as f:
            prompt_config = yaml.safe_load(f)
        template = prompt_config.get("template", "").strip()
        if not template:
            logging.warning(f"Template not found or empty in {prompt_file_path}")
        return template
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {prompt_file_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {prompt_file_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading prompt: {e}")
        raise


def load_llm_configurations(config_path: str = DEFAULT_LLM_CONFIG_PATH) -> dict:
    """
    Loads LLM configurations from a YAML file.

    Args:
        config_path: The path to the LLM configuration YAML file.

    Returns:
        A dictionary containing LLM parameters.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("llm_parameters", {})
    except FileNotFoundError:
        logging.error(f"LLM configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing LLM configuration YAML file {config_path}: {e}")
        return {}
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading LLM configurations: {e}")
        return {}
