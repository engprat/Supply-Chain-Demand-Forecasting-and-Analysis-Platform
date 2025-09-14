'''src/utils/schema_registry_utils.py'''
import yaml
import difflib
import os

REGISTRY_DIR = os.path.join(os.path.dirname(__file__), '../../schema_registry')


def list_schema_versions():
    return sorted([f for f in os.listdir(REGISTRY_DIR) if f.startswith('schema_v') and f.endswith('.yaml')])


def load_schema_version(filename):
    with open(os.path.join(REGISTRY_DIR, filename), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def diff_schemas(schema1, schema2):
    """Return a unified diff of two schema dicts as a string."""
    s1 = yaml.dump(schema1, sort_keys=True).splitlines()
    s2 = yaml.dump(schema2, sort_keys=True).splitlines()
    return '\n'.join(difflib.unified_diff(s1, s2, fromfile='schema1', tofile='schema2'))


def detect_schema_drift(current_schema_path='../../configs/schema.yaml'):
    """Compare current schema to latest registry version."""
    versions = list_schema_versions()
    if not versions:
        print('No schema versions in registry.')
        return None
    latest = versions[-1]
    current = load_schema_version(os.path.basename(current_schema_path))
    latest_schema = load_schema_version(latest)
    diff = diff_schemas(latest_schema, current)
    if diff:
        print('Schema drift detected!')
        print(diff)
    else:
        print('No schema drift detected.')
    return diff
