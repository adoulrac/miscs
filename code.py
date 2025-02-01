import fnmatch

def extract_related_data_optimized(d: dict, key_to_extract: str, exclude_pattern: str = None) -> dict:
    """
    Optimized version of extract_related_data function with exclusion of keys based on patterns.
    
    :param d: The dictionary to extract data from
    :param key_to_extract: The key whose related data should be extracted
    :param exclude_pattern: The pattern to match keys that should be excluded (default is None)
    :return: A new dictionary with extracted related data and excluded keys
    """
    result = {}

    for key, value in d.items():
        # Skip keys that match the exclude pattern
        if exclude_pattern and fnmatch.fnmatch(key, exclude_pattern):
            continue
        
        # Extract the relevant key-value pair
        if key_to_extract in key:
            result[key] = value

        # Recursively handle nested dictionaries and lists
        if isinstance(value, dict):
            nested_result = extract_related_data_optimized(value, key_to_extract, exclude_pattern)
            if nested_result:
                result[key] = nested_result

        elif isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, dict):
                    new_item = extract_related_data_optimized(item, key_to_extract, exclude_pattern)
                    if new_item:
                        new_list.append(new_item)
                else:
                    new_list.append(item)
            if new_list:
                result[key] = new_list
        else:
            result[key] = value

    return result

# Example of use
data = {
    "user.name": "Alice",
    "user.age": 30,
    "user.city": "Paris",
    "product.name": "Laptop",
    "ignore.this": "should be excluded",
    "user.preferences.color": "blue"
}

# Extract related data for "user" and exclude any keys that match "ignore.*"
result = extract_related_data_optimized(data, "user", "ignore.*")
print(result)




import fnmatch

def remove_prefix_and_exclude_keys(d: dict, prefix: str, exclude_pattern: str = None) -> dict:
    """
    Remove a prefix from all keys in the dictionary and optionally exclude keys matching a pattern.
    
    :param d: The dictionary to process
    :param prefix: The prefix to remove from the keys
    :param exclude_pattern: The pattern to match keys that should be excluded (default is None)
    :return: A new dictionary with modified keys
    """
    result = {}

    for key, value in d.items():
        # Skip keys that match the exclude pattern
        if exclude_pattern and fnmatch.fnmatch(key, exclude_pattern):
            continue
        
        # Remove the prefix if the key starts with it
        new_key = key[len(prefix):] if key.startswith(prefix) else key
        
        # Recursively process the value if it's a dictionary or list
        if isinstance(value, dict):
            result[new_key] = remove_prefix_and_exclude_keys(value, prefix, exclude_pattern)
        elif isinstance(value, list):
            result[new_key] = [remove_prefix_and_exclude_keys(v, prefix, exclude_pattern) if isinstance(v, dict) else v for v in value]
        else:
            result[new_key] = value
    
    return result

# Example of use
data = {
    "user.name": "Alice",
    "user.age": 30,
    "user.city": "Paris",
    "product.name": "Laptop",
    "ignore.this": "should be excluded"
}

# Remove 'user.' prefix and exclude keys that match 'ignore.*' pattern
result = remove_prefix_and_exclude_keys(data, "user.", "ignore.*")
print(result)




from collections import deque, defaultdict

def extract_related_data(data: dict, start_key: str) -> dict:
    """
    Extrait un dictionnaire filtré en conservant la structure d'origine et en récupérant
    toutes les entrées liées selon un système de préfixes (avant le point '.').
    
    Optimisations :
    - Pré-indexe les clés selon leur préfixe pour accélérer les recherches.
    - Utilise une deque pour améliorer la gestion des clés à traiter.

    :param data: Dictionnaire source.
    :param start_key: Clé de départ pour l'extraction.
    :return: Dictionnaire filtré avec toutes les dépendances résolues.
    """

    # Étape 1 : Construire un index des clés par préfixe
    prefix_map = defaultdict(set)
    for key in data:
        prefix = key.split('.')[0]
        prefix_map[prefix].add(key)

    # Étape 2 : Parcourir les clés liées
    if start_key not in data:
        return {}

    extracted = {}
    seen_keys = set()
    keys_to_process = deque([start_key])

    while keys_to_process:
        key = keys_to_process.popleft()
        if key in seen_keys:
            continue
        seen_keys.add(key)

        if key in data:
            value = data[key]
            extracted[key] = value  # Conserver la structure

            # Si la valeur est une chaîne, vérifier si elle correspond à un préfixe existant
            if isinstance(value, str) and value in prefix_map:
                keys_to_process.extend(prefix_map[value])

            # Si la valeur est une liste, traiter chaque élément
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item in prefix_map:
                        keys_to_process.extend(prefix_map[item])

    return extracted



from collections.abc import Iterable, Mapping

def extract_related_data(data: dict, start_key: str) -> dict:
    """
    Extracts a filtered dictionary that maintains the exact input structure.
    It resolves subdependencies by looking at values and matching keys based on their prefix (before the dot).

    :param data: The input dictionary.
    :param start_key: The key to start extraction from.
    :return: A filtered dictionary with all related key-value pairs.
    """
    def get_related_keys(value):
        """Finds all keys where the prefix before the dot matches the given value."""
        return {k for k in data if k.split('.')[0] == value}

    def resolve_keys(keys_to_process, seen_keys):
        """Recursively processes keys, adding related keys dynamically."""
        while keys_to_process:
            key = keys_to_process.pop()
            if key in seen_keys:
                continue
            seen_keys.add(key)

            if key in data:
                value = data[key]
                extracted[key] = value  # Keep the exact structure

                # Check if value references another key prefix
                if isinstance(value, str):
                    keys_to_process.update(get_related_keys(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            keys_to_process.update(get_related_keys(item))

    if start_key not in data:
        return {}

    extracted = {}
    resolve_keys({start_key}, set())
    return extracted





from collections.abc import Iterable, Mapping

def extract_related_dict(data: dict, key: str) -> dict:
    """
    Extracts a filtered dictionary that includes only the relevant key and all its related sub-objects.
    Maintains the original dictionary structure.

    :param data: The input dictionary.
    :param key: The key to start extraction from.
    :return: A filtered dictionary containing the key and its related sub-objects.
    """
    def collect_keys(value, collected):
        """Recursively collect all keys that are referenced in the dictionary."""
        if isinstance(value, str) and value in data and value not in collected:
            collected.add(value)
            collect_keys(data[value], collected)
        elif isinstance(value, list):
            for item in value:
                collect_keys(item, collected)
        elif isinstance(value, dict):
            for v in value.values():
                collect_keys(v, collected)

    if key not in data:
        return {}

    # Step 1: Find all related keys
    related_keys = {key}
    collect_keys(data[key], related_keys)

    # Step 2: Filter the original dictionary to keep only related keys
    filtered_dict = {k: v for k, v in data.items() if k in related_keys}

    return filtered_dict




from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd

class MyDataModel(BaseModel):
    data: Dict[str, Any]  # Accepts a dictionary with any key-value pairs

def extract_prefixed_keys(model: MyDataModel, prefix: str) -> pd.DataFrame:
    # Extract relevant keys and values
    filtered_items = {k[len(prefix):]: v for k, v in model.data.items() if k.startswith(prefix)}
    
    # Convert to DataFrame
    df = pd.DataFrame([filtered_items])  # Wrap in list to create single-row DataFrame
    
    return df

# Example dictionary
large_dict = {
    "price_AAPL": 150,
    "price_GOOG": 2800,
    "price_MSFT": 299,
    "volume_AAPL": 1000,
    "volume_GOOG": 2000
}

# Convert to Pydantic model
model = MyDataModel(data=large_dict)

# Extract DataFrame for "price_" prefix
df_prices = extract_prefixed_keys(model, "price_")
print(df_prices)






import ast

def parse_dictionary(input_dict):
    def parse_value(value):
        """Convert string representations of lists into actual lists."""
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
        return value

    def build_nested_structure(keys, value, structure):
        """Recursively build nested dictionaries from a list of keys."""
        if len(keys) == 1:
            structure[keys[0]] = value
        else:
            if keys[0] not in structure:
                structure[keys[0]] = {}
            build_nested_structure(keys[1:], value, structure[keys[0]])

    def resolve_prefixes(parsed_dict):
        """Resolve prefixes by checking if a value (or list item) is a prefix of another key."""
        final_dict = {}
        for key, value in parsed_dict.items():
            if isinstance(value, list):
                # Check if any item in the list is a prefix for another key
                for item in value:
                    if isinstance(item, str):
                        # Check if this item is a prefix for another key
                        for other_key in parsed_dict:
                            if other_key != key and other_key.startswith(item + '.'):
                                # Create a nested structure for the prefix
                                if item not in final_dict:
                                    final_dict[item] = {}
                                build_nested_structure(
                                    other_key[len(item) + 1:].split('.'),
                                    parsed_dict[other_key],
                                    final_dict[item]
                                )
            elif isinstance(value, str):
                # Check if this value is a prefix for another key
                for other_key in parsed_dict:
                    if other_key != key and other_key.startswith(value + '.'):
                        # Create a nested structure for the prefix
                        if value not in final_dict:
                            final_dict[value] = {}
                        build_nested_structure(
                            other_key[len(value) + 1:].split('.'),
                            parsed_dict[other_key],
                            final_dict[value]
                        )
            # Add the current key-value pair to the final dictionary
            if key not in final_dict:
                build_nested_structure(key.split('.'), value, final_dict)
        return final_dict

    # First pass: Parse values and build the initial dictionary
    parsed_dict = {}
    for key, value in input_dict.items():
        parsed_dict[key] = parse_value(value)

    # Second pass: Resolve prefixes and build the final nested structure
    final_dict = resolve_prefixes(parsed_dict)
    return final_dict

# Example usage:
input_dict = {
    'user.name': 'John Doe',
    'user.age': '30',
    'user.hobbies': '["reading", "traveling", "coding"]',
    'address.city': 'New York',
    'address.zip': '10001',
    'reading.property': 'value for reading',
    'coding.difficulty': 'high',
    'traveling.destination': 'Japan'
}

parsed_dict = parse_dictionary(input_dict)
print(parsed_dict)