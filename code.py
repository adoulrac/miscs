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