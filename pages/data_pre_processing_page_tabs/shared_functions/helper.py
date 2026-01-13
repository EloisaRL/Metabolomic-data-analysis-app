import os, json, ast

def read_study_details_dpp(folder):
    """Reads study details for a given study, contains info of the study name and dataset source"""
    details_path = os.path.join(folder, "study_details.txt")
    details = {}
    if os.path.exists(details_path):
        with open(details_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key, value = parts
                        details[key.strip()] = value.strip()
    return details

def get_flow_steps(flow_name):
    """Return a list of preprocessing steps from data_preprocessing_flows/{flow_name}.txt."""
    path = os.path.join("data_preprocessing_flows", f"{flow_name}.txt")
    if os.path.exists(path):
        with open(path) as f:
            # Parse the file as JSON
            data = json.load(f)
            # Extract steps ensuring order: missing_values, transformation, standardisation
            steps = [
                data.get("missing_values", ""),
                data.get("transformation", ""),
                data.get("standardisation", "")
            ]
            # Remove any empty strings if a key was missing.
            steps = [step for step in steps if step]
            return steps
    else:
        return []
    
def filter_data_groups(data, filter):
    """Flatten filter values into a single list of allowed groups."""
    allowed_groups = []
    for key, val in filter.items():
        if isinstance(val, list):
            allowed_groups.extend(val)
        else:
            allowed_groups.append(val)
    # Filter rows where 'Group' is in the allowed groups.
    data = data[data['Group'].isin(allowed_groups)]
    return data

def remove_outliers(data, outliers):
    """Drop sample outliers"""
    if outliers:
        data = data.drop(outliers)
    return data

def get_group_value(val):
    # If the value is already a list, join the items.
    if isinstance(val, list):
        return ', '.join(val)
    # If it's a string that might represent a list, try to parse it.
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return ', '.join(parsed)
    except Exception:
        pass
    # Otherwise, assume it’s a simple string and return as is.
    return val