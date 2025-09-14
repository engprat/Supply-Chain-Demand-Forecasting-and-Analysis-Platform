'''src/utils/schema_utils.py'''
import yaml

def load_schema(path):
    """
    Loads a YAML schema file and returns the parsed dictionary.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
