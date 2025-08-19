import os
from typing import Any, Dict, List

import yaml

# Define the base directory for prompts
PROMPTS_DIR = "prompts"


def load_prompt(prompt_id: str) -> Dict[str, Any]:
    """
    Loads a prompt from the prompt registry based on its ID.
    The prompt ID is expected to be in the format 'framework/prompt_name_vX'.
    """
    try:
        # Split the prompt_id to get the framework and the actual prompt name
        parts = prompt_id.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid prompt_id format: {prompt_id}. Expected 'framework/prompt_name'.")

        framework, prompt_name = parts

        # Construct the expected file path
        # We assume prompt files are in YAML format for now
        file_path = os.path.join(PROMPTS_DIR, framework, f"{prompt_name}.yaml")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prompt file not found at: {file_path}")

        with open(file_path, "r") as f:
            prompt_data = yaml.safe_load(f)

        # Basic validation of loaded prompt data
        if not all(key in prompt_data for key in ["id", "role", "template"]):
            raise ValueError(f"Prompt file {file_path} is missing required keys (id, role, template).")

        return prompt_data

    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        print(f"Error loading prompt '{prompt_id}': {e}")
        return {}


def get_prompt_template(prompt_id: str, **kwargs) -> str:
    """
    Loads a prompt and applies templating with provided variables.
    Returns the formatted prompt as a string.
    """
    prompt_data = load_prompt(prompt_id)
    if not prompt_data:
        return ""

    template = prompt_data.get("template", "")

    # Simple string formatting for placeholders.
    # For more complex templating, consider Jinja2.
    try:
        formatted_template = template.format(**kwargs)
        return formatted_template
    except KeyError as e:
        print(f"Error formatting prompt '{prompt_id}': Missing variable {e}")
        return template  # Return raw template if formatting fails


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Assuming 'prompts/langchain/negotiation_coach.yaml' exists
    # Example of loading a prompt
    prompt_config = load_prompt("langchain/negotiation_coach_v1")
    if prompt_config:
        print("Loaded Prompt Config:")
        print(prompt_config)

        # Example of getting a formatted template
        formatted_prompt = get_prompt_template(
            "langchain/negotiation_coach_v1", context="Some retrieved context here.", question="What is BATNA?"
        )
        print("\nFormatted Prompt:")
        print(formatted_prompt)
    else:
        print("Failed to load prompt.")
