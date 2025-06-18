# pages/data_pre_processing_page_tabs/data_summary.py
from dash import html, dcc, callback, Input, Output, State, dash_table, no_update, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import os
import json
import glob
import requests
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import numpy as np
import re
import numbers
import ast
import pandas as pd
import time
import logging
logger = logging.getLogger(__name__)


UPLOAD_FOLDER = "pre-processed-datasets"
# Path to the temporary file that holds the selected studies
SELECTED_STUDIES_FILE = os.path.join(UPLOAD_FOLDER, "selected_studies_temp.txt")

###### REMOVE THE RELIANCE ON BELOW #######
default_metadata = pd.DataFrame({
    "Sample Name": ["Sample1", "Sample2", "Sample3"],
    "Group": ["Control", "Treatment", "Control"]
})
default_md_filter = {"Group": ["Control", "Treatment"]}

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

# ================================= #
# Missing Values Imputation Options #
# ================================= #
def missing_values_knn_impute(data):
    """Uses KNNImputer""" 
    #print(data)
    if 'Group' in data.columns:
        group = data['Group']
        data_numeric = data.drop('Group', axis=1)
    else:
        data_numeric = data
    imputer = KNNImputer(n_neighbors=2, weights="uniform").set_output(transform="pandas")
    imputed_numeric = imputer.fit_transform(data_numeric)
    if 'Group' in data.columns:
        imputed_numeric['Group'] = group
    return imputed_numeric

def missing_values_mean_impute(data):
    """Uses SimpleImputer with mean strategy"""
    if 'Group' in data.columns:
        group = data['Group']
        data_numeric = data.drop('Group', axis=1)
    else:
        data_numeric = data
    imputer = SimpleImputer(strategy='mean')
    imputed_array = imputer.fit_transform(data_numeric)
    imputed_numeric = pd.DataFrame(imputed_array, columns=data_numeric.columns, index=data_numeric.index)
    if 'Group' in data.columns:
        imputed_numeric['Group'] = group
    return imputed_numeric

def missing_values_iterative_impute(data):
    """Uses IterativeImputer"""
    if 'Group' in data.columns:
        group = data['Group']
        data_numeric = data.drop('Group', axis=1)
    else:
        data_numeric = data
    imputer = IterativeImputer(random_state=0)
    imputed_array = imputer.fit_transform(data_numeric)
    imputed_numeric = pd.DataFrame(imputed_array, columns=data_numeric.columns, index=data_numeric.index)
    if 'Group' in data.columns:
        imputed_numeric['Group'] = group
    return imputed_numeric

# ====================== #
# Transformation Options #
# ====================== #
def log_transform(data):
    """Log transformation: np.log(data+1)"""
    if 'Group' in data.columns:
        group = data['Group']
        data_numeric = data.drop('Group', axis=1)
    else:
        data_numeric = data
    data_log = np.log(data_numeric + 1)
    if 'Group' in data.columns:
        data_log['Group'] = group
    return data_log

def cube_root_transform(data):
    """Cube root transformation using np.cbrt"""
    if 'Group' in data.columns:
        group = data['Group']
        data_numeric = data.drop('Group', axis=1)
    else:
        data_numeric = data
    data_cube = np.cbrt(data_numeric)
    if 'Group' in data.columns:
        data_cube['Group'] = group
    return data_cube

# ======================= #
# Standardisation Options #
# ======================= #
def standardise_standard_scaler(data):
    """Uses Standard Scaler"""
    if 'Group' in data.columns:
        group = data['Group']
        data_numeric = data.drop('Group', axis=1)
    else:
        data_numeric = data
    scaler = StandardScaler().set_output(transform="pandas")
    scaled = scaler.fit_transform(data_numeric)
    if 'Group' in data.columns:
        scaled['Group'] = group
    return scaled

def standardise_min_max_scaler(data):
    """Uses Min Max Scaler"""
    if 'Group' in data.columns:
        group = data['Group']
        data_numeric = data.drop('Group', axis=1)
    else:
        data_numeric = data
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(data_numeric)
    scaled = pd.DataFrame(scaled_array, columns=data_numeric.columns, index=data_numeric.index)
    if 'Group' in data.columns:
        scaled['Group'] = group
    return scaled

def standardise_robust_scaler(data):
    """Uses Robust Scaler"""
    if 'Group' in data.columns:
        group = data['Group']
        data_numeric = data.drop('Group', axis=1)
    else:
        data_numeric = data
    scaler = RobustScaler()
    scaled_array = scaler.fit_transform(data_numeric)
    scaled = pd.DataFrame(scaled_array, columns=data_numeric.columns, index=data_numeric.index)
    if 'Group' in data.columns:
        scaled['Group'] = group
    return scaled

def standardise_max_abs_scaler(data):
    """Uses MaxAbsSacler"""
    if 'Group' in data.columns:
        group = data['Group']
        data_numeric = data.drop('Group', axis=1)
    else:
        data_numeric = data
    scaler = MaxAbsScaler()
    scaled_array = scaler.fit_transform(data_numeric)
    scaled = pd.DataFrame(scaled_array, columns=data_numeric.columns, index=data_numeric.index)
    if 'Group' in data.columns:
        scaled['Group'] = group
    return scaled

# ======================================================================================================== #
# Determining if in the different datasets uploaded there are difference in how they refer to the patients #
# ======================================================================================================== #

def get_removal_info(example_extra_id, ref_ids):
    """
    Determines if by removing a number of letters from the beginning (prefix) or 
    from the end (suffix) of the example_extra_id we can obtain a string that 
    is one of the reference IDs in ref_ids.

    Parameters
    ----------
    example_extra_id : str
        One metadata ID that contains extra letters.
    ref_ids : list or set of str
        The reference IDs that are considered correct.

    Returns
    -------
    tuple : (removal_type, removal_amount)
        removal_type is either 'prefix' or 'suffix', and removal_amount is the number
        of characters to remove. If no match is found, (None, 0) is returned.
    """
    ref_set = set(ref_ids)
    
    best_prefix_removal = None
    best_suffix_removal = None

    # Check for prefix removal: try removing 0 to len(example_extra_id) characters from the start.
    for k in range(0, len(example_extra_id) + 1):
        candidate = example_extra_id[k:]
        if candidate in ref_set:
            best_prefix_removal = k
            break  # minimal removal found.

    # Check for suffix removal: try removing 0 to len(example_extra_id) characters from the end.
    for k in range(0, len(example_extra_id) + 1):
        # When k==0 no removal occurs.
        candidate = example_extra_id[:-k] if k > 0 else example_extra_id
        if candidate in ref_set:
            best_suffix_removal = k
            break  # minimal removal found.

    # Decide which removal works best.
    if best_prefix_removal is None and best_suffix_removal is None:
        return None, 0  # No valid removal found.
    elif best_prefix_removal is not None and best_suffix_removal is not None:
        if best_prefix_removal <= best_suffix_removal:
            return 'prefix', best_prefix_removal
        else:
            return 'suffix', best_suffix_removal
    elif best_prefix_removal is not None:
        return 'prefix', best_prefix_removal
    else:
        return 'suffix', best_suffix_removal

def get_removal_info_for_combining(example_extra_id, ref_ids):
    """
    Try removing k chars from front or back of both example_extra_id and
    each ref_id so they line up.  If trimming k leaves a trailing/leading
    '-' or '_' in both strings, we treat that as “not clean” and return k+1.
    Otherwise we return the minimal k that yields an exact, punctuation‐free match.
    """
    punct = {'-', '_'}
    refs = list(ref_ids)

    # quick check: no trim needed?
    if example_extra_id in refs:
        return None, 0

    max_k = min(len(example_extra_id), *(len(r) for r in refs))

    for k in range(1, max_k + 1):
        # --- suffix removal ---
        core_ex = example_extra_id[:-k]
        for ref in refs:
            if len(ref) < k:
                continue
            core_ref = ref[:-k]

            # 1) punctuation‐leftover case: both end in '-' or '_'
            if core_ex and core_ref \
               and core_ex[-1] in punct and core_ref[-1] in punct \
               and core_ex[:-1] == core_ref[:-1]:
                return 'suffix', k + 1

            # 2) clean exact match (no trailing punctuation)
            if core_ex == core_ref \
               and (not core_ex or core_ex[-1] not in punct):
                return 'suffix', k

        # --- prefix removal ---
        core_ex = example_extra_id[k:]
        for ref in refs:
            if len(ref) < k:
                continue
            core_ref = ref[k:]

            # 1) punctuation‐leftover at the start
            if core_ex and core_ref \
               and core_ex[0] in punct and core_ref[0] in punct \
               and core_ex[1:] == core_ref[1:]:
                return 'prefix', k + 1

            # 2) clean exact match (no leading punctuation)
            if core_ex == core_ref \
               and (not core_ex or core_ex[0] not in punct):
                return 'prefix', k

    # nothing aligned
    return None, 0

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

# ========================= #
# Data processing functions #
# ========================= #

def static_preprocess_workbench(folder, preprocessing_steps=None, outliers=None, filter=None, selected_group=None, database_source=None):
    """Processes datasets with RefMet names as ids"""

    identifier_name = "database_identifier" 
    # Get study details so we can obtain the study name
    details = read_study_details_dpp(folder)
    study_name = details.get("Study Name", "")
    
    # Find all CSV files in the study folder.
    files = glob.glob(os.path.join(folder, "*.csv"))
    if len(files) == 0:
        #raise Exception("No CSV files found in the folder.")
        logger.error("Data summary tab - No CSV files found in the folder.")
        return None
    
    def preprocess(df):
        # Workbench CSVs are assumed to have at least the following columns:
        # 'Samples' and 'Class'.

        # only do the merge logic if at least one Samples value has _NEG or _POS
        if (df['Samples']
            .fillna('')           # turn NaN → ""
            .astype(str)          # ensure string dtype
            .str.contains(r'(_NEG|_POS)$')
            ).any():
            # create a "base" sample ID without the trailing suffix
            df['base_id'] = df['Samples'].str.replace(r'(_NEG|_POS)$', '', regex=True)

            # build an aggregation dict:
            #   - first() for Samples, Class (and our helper base_id)
            #   - mean() for everything else
            agg = {}
            for col in df.columns:
                if col in ('Samples', 'Class', 'base_id'):
                    agg[col] = 'first'
                else:
                    agg[col] = 'mean'

            # group & aggregate
            df = df.groupby('base_id', as_index=False).agg(agg)

            # restore Samples to the cleaned base_id, drop helper
            df['Samples'] = df['base_id']
            df = df.drop(columns=['base_id'])

        # … now continue with the rest of the preprocessing …
        data_filt = df.copy()

        if 'Samples' not in data_filt.columns or 'Class' not in data_filt.columns:
            #raise Exception("CSV file must contain 'Samples' and 'Class' columns.")
            logger.error("Data summary tab - CSV file must contain 'Samples' and 'Class' columns.")
            return None
        data_filt[identifier_name] = data_filt['Samples']
        data_filt.index = data_filt[identifier_name]
        
        # --- Group Extraction Logic with Filter Support ---
        # Before dropping the metadata columns, process the 'Class' column.
        # If there are multiple groups (separated by " | ") and a selected_group is provided,
        # determine its position from the first row and keep only that element for all rows.
        if "Class" in data_filt.columns:
            first_class = str(data_filt.iloc[0]["Class"])
            groups_first = [grp.strip() for grp in first_class.split("|") if grp.strip()]
            
            if len(groups_first) > 1:
                # If filter is provided, determine sel_index from allowed groups.
                if filter is not None:
                    allowed_groups = []
                    for key, vals in filter.items():
                        allowed_groups.extend(vals)
                    sel_index = None
                    for i, grp in enumerate(groups_first):
                        if grp in allowed_groups:
                            sel_index = i
                            break
                    if sel_index is None:
                        sel_index = 0
                # Else, if a selected_group is provided, use that.
                elif selected_group is not None:
                    try:
                        sel_index = groups_first.index(selected_group)
                    except ValueError:
                        sel_index = 0  # Default to the first group if the chosen one isn’t found.
                else:
                    sel_index = 0
                # For each row, split the Class value and take the element at the chosen index.
                data_filt["Group"] = data_filt["Class"].apply(
                    lambda s: ([grp.strip() for grp in str(s).split("|") if grp.strip()][sel_index]
                               if len([grp.strip() for grp in str(s).split("|") if grp.strip()]) > sel_index 
                               else [grp.strip() for grp in str(s).split("|") if grp.strip()][0])
                )
            else:
                # Either only one group exists or no selection was made;
                # in this case, just copy the original Class value.
                data_filt["Group"] = data_filt["Class"]

        # --- End Group Extraction Logic ---

        if filter is not None and filter != {}:
            data_filt = filter_data_groups(data_filt, filter)

        # Remove the metadata columns
        data_filt = data_filt.drop(columns=['Class', 'Samples', identifier_name])

        # Convert metabolite names to RefMet IDs using the Workbench API.
        if database_source == "metabolomics workbench":
            mets_url = 'https://www.metabolomicsworkbench.org/rest/study/study_id/repl/metabolites'
            try:
                mets = requests.get(mets_url.replace('repl', study_name)).text
                mets_df = pd.read_json(mets).T
                mets_dict = dict(zip(mets_df['metabolite_name'], mets_df['refmet_name']))
                # make sure 'Group' is identity-mapped
                mets_dict['Group'] = 'Group'
                data_filt.columns = data_filt.columns.map(mets_dict)  

            except Exception:
                logger.exception("Data summary tab - Error converting to RefMet IDs")

        # Drop any columns with missing names and try to drop empty column names.
        data_filt = data_filt.loc[:, data_filt.columns.notna()]
        try:
            data_filt = data_filt.drop(columns=[''])
        except KeyError:
            pass

        # Remove outliers if provided.
        if outliers is not None and outliers != '':
            data_filt = remove_outliers(data_filt, outliers)

        # Missingness checks:
        # Replace empty strings, single spaces, and 0 with NaN.
        data_filt = data_filt.replace(['', ' ', 0], np.nan)
        # Drop rows and columns where all values are missing.
        data_filt = data_filt.dropna(axis=0, how='all')
        data_filt = data_filt.dropna(axis=1, how='all')

        # Remove rows/columns that are entirely 0.
        data_filt = data_filt.loc[:, (data_filt != 0).any(axis=0)]
        data_filt = data_filt.loc[(data_filt != 0).any(axis=1), :]


        # Define the subset of columns to check (i.e., all columns except 'Group')
        non_group_cols = [col for col in data_filt.columns if col != 'Group' and col != 'group_type' and col != 'Samples']

        # Drop rows only if all non‑Group columns are NaN (re added)
        #data_filt = data_filt.dropna(axis=0, how='all', subset=non_group_cols)
        
        # Drop columns with more than 50% missing data.
        data_filt = data_filt.dropna(axis=1, thresh=0.5 * data_filt.shape[0])
        missing_pct = data_filt.isnull().sum().sum() / (data_filt.shape[0] * data_filt.shape[1]) * 100
        #print(f"Missingness: {missing_pct:.2f}%")

        return data_filt

    proc_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            logger.exception(f"Data summary tab - Error reading file {f}")
            continue
        proc_data = preprocess(df)
        if proc_data is None:
            continue

        # ----------------------------
        # Apply optional preprocessing steps:
        # ----------------------------
        if preprocessing_steps is not None:
            # Missing values imputation options:
            if any(opt in preprocessing_steps for opt in ['knn_imputer', 'mean_imputer', 'iterative_imputer']):
                if 'knn_imputer' in preprocessing_steps:
                    proc_data = missing_values_knn_impute(proc_data)
                elif 'mean_imputer' in preprocessing_steps:
                    proc_data = missing_values_mean_impute(proc_data)
                elif 'iterative_imputer' in preprocessing_steps:
                    proc_data = missing_values_iterative_impute(proc_data)

            # delete cols where all values are the same
            proc_data = proc_data[[i for i in proc_data if len(set(proc_data[i]))>1]]
            
            # Transformation options:
            if any(opt in preprocessing_steps for opt in ['log_transform', 'cube_root']):
                if 'log_transform' in preprocessing_steps:
                    proc_data = log_transform(proc_data)
                elif 'cube_root' in preprocessing_steps:
                    proc_data = cube_root_transform(proc_data)
            
            # Standardisation options:
            if any(opt in preprocessing_steps for opt in ['standard_scaler', 'min_max_scaler', 'robust_scaler', 'max_abs_scaler']):
                if 'standard_scaler' in preprocessing_steps:
                    proc_data = standardise_standard_scaler(proc_data)
                elif 'min_max_scaler' in preprocessing_steps:
                    proc_data = standardise_min_max_scaler(proc_data)
                elif 'robust_scaler' in preprocessing_steps:
                    proc_data = standardise_robust_scaler(proc_data)
                elif 'max_abs_scaler' in preprocessing_steps:
                    proc_data = standardise_max_abs_scaler(proc_data)

        proc_dfs.append(proc_data)
    
    if len(proc_dfs) == 0:
        logger.error("Data summary tab - No valid processed data from CSV files")
        return None 
        #raise Exception("No valid processed data from CSV files.")
    # If more than one CSV file was processed, combine them.
    if len(proc_dfs) > 1:
        # Concatenate along columns. raw_data_combined = pd.concat(proc_dfs, axis=1, join='inner')
        combined = pd.concat(proc_dfs, axis=1)
        # Remove duplicate columns.
        combined = combined.loc[:, ~combined.columns.duplicated()]

        processed_data = combined
    else:
        processed_data = proc_dfs[0]

    return processed_data

def static_preprocess(folder, metadata, preprocessing_steps=None, outliers=None, filter=None, selected_group=None):
    """Processes datasets with ChEBI id as ids"""
    def preprocess(df):
        identifier_name = "database_identifier" 
        data = df.copy()
        try:
            data['mass_to_charge'] = data['mass_to_charge'].round(2)
            data['mass_to_charge'] = data['mass_to_charge'].astype('str').apply(lambda x: re.sub(r'\.', '_', x))
        except KeyError:
            pass

        data = data[data[identifier_name].notna()]

        if data.shape[0] == 0:
            #print('No CHEBIS for assay')
            return None
        else:
            data = data[data[identifier_name] != 'unknown']
            data.index = data[identifier_name]
            
            # First, obtain the sample list from metadata and filter data columns.
            samples = metadata['Sample Name'].tolist()
            data_filtered = data.iloc[:, data.columns.isin(samples)]

            if data_filtered.shape[1] == 0:
                # No matching sample columns found using original 'Sample Name'
                print("No samples found using the original 'Sample Name'. Trying to fix extra letters...")

                # Get the reference IDs from data.columns.
                reference_ids = data.columns.tolist()

                found_removal = False
                removal_type = None
                removal_amount = 0

                # Iterate through all sample names in metadata to see if any yield a valid removal.
                for extra_id_example in metadata['Sample Name']:
                    extra_id_example = str(extra_id_example)
                    removal_type, removal_amount = get_removal_info(extra_id_example, reference_ids)
                    if removal_type is not None and removal_amount > 0:
                        print(f"Found removal using sample '{extra_id_example}': {removal_type} removal of {removal_amount}")
                        found_removal = True
                        break
                    else:
                        print(f"No valid removal found for sample '{extra_id_example}'.")
                
                # Apply the found removal if any, else keep the original names.
                if found_removal:
                    if removal_type == 'prefix':
                        metadata['Fixed Sample Name'] = metadata['Sample Name'].astype(str).apply(lambda s: s[removal_amount:])
                    elif removal_type == 'suffix':
                        metadata['Fixed Sample Name'] = metadata['Sample Name'].astype(str).apply(lambda s: s[:-removal_amount])
                else:
                    print("No removal of letters generated a match for any sample. Using original sample names.")
                    metadata['Fixed Sample Name'] = metadata['Sample Name']
                
                # Use the fixed sample names to filter data.
                # before filtering:
                orig_n = data.shape[1]
                samples = metadata['Fixed Sample Name'].tolist()
                data = data.iloc[:, data.columns.isin(samples)]

                # after filtering:
                new_n = data.shape[1]

                #print(f"Filtered out {orig_n - new_n} samples; {new_n} remain (out of {orig_n}).")
                md_dict = dict(zip(metadata['Fixed Sample Name'], metadata[selected_group]))
            else:
                # If there were sample columns found using the original sample names, no removal is done.
                data = data_filtered
                md_dict = dict(zip(metadata['Sample Name'], metadata[selected_group]))
                #print("Mapping dictionary using original sample names:", md_dict)



            data = data.apply(pd.to_numeric, errors='coerce')
            data = data.T
            
            data['Group'] = data.index.map(lambda sample: get_group_value(md_dict.get(sample, '')))

            if outliers is not None and outliers != "":
                data = remove_outliers(data, outliers)
            if filter is not None and filter != {}:
                data = filter_data_groups(data, filter)

            data = data.replace(['', ' '], np.nan)
            data = data.dropna(axis=0, how='all')
            data = data.dropna(axis=1, how='all')
            data = data.loc[:, (data != 0).any(axis=0)]
            data = data.loc[(data != 0).any(axis=1), :]

            # Define the subset of columns to check (i.e., all columns except 'Group')
            non_group_cols = [col for col in data.columns if col != 'Group' and col != 'group_type' and col != 'Samples']

            # Drop rows only if all non‑Group columns are NaN
            data = data.dropna(axis=0, how='all', subset=non_group_cols)

            data = data.dropna(axis=1, thresh=0.5 * data.shape[0])
            
            return data

    files = glob.glob(f"{folder}/*maf.tsv")
    
    if len(files) == 0:
        logger.error("Data summary tab - No assay files found in the folder.")
        return None 
        #raise Exception("No assay files found in the folder.")

    proc_dfs = []
    for f in files:
        df = pd.read_csv(f, sep='\t')
        proc_data = preprocess(df)
        if proc_data is None:
            continue
        
        # ----------------------------
        # Apply optional preprocessing steps:
        # ----------------------------
        if preprocessing_steps is not None:
            # Missing values imputation options:
            if any(opt in preprocessing_steps for opt in ['knn_imputer', 'mean_imputer', 'iterative_imputer']):
                if 'knn_imputer' in preprocessing_steps:
                    proc_data = missing_values_knn_impute(proc_data)
                elif 'mean_imputer' in preprocessing_steps:
                    proc_data = missing_values_mean_impute(proc_data)
                elif 'iterative_imputer' in preprocessing_steps:
                    proc_data = missing_values_iterative_impute(proc_data)
            
            # Transformation options:
            if any(opt in preprocessing_steps for opt in ['log_transform', 'cube_root']):
                if 'log_transform' in preprocessing_steps:
                    proc_data = log_transform(proc_data)
                elif 'cube_root' in preprocessing_steps:
                    proc_data = cube_root_transform(proc_data)
            
            # Standardisation options:
            if any(opt in preprocessing_steps for opt in ['standard_scaler', 'min_max_scaler', 'robust_scaler', 'max_abs_scaler']):
                if 'standard_scaler' in preprocessing_steps:
                    proc_data = standardise_standard_scaler(proc_data)
                elif 'min_max_scaler' in preprocessing_steps:
                    proc_data = standardise_min_max_scaler(proc_data)
                elif 'robust_scaler' in preprocessing_steps:
                    proc_data = standardise_robust_scaler(proc_data)
                elif 'max_abs_scaler' in preprocessing_steps:
                    proc_data = standardise_max_abs_scaler(proc_data)
        
        proc_dfs.append(proc_data)

    if len(proc_dfs) == 0:
        logger.error("Data summary tab - No valid processed data from assay files.")
        return None 
        #raise Exception("No valid processed data from assay files.")

    if len(proc_dfs) > 1:
        # 1) try the normal inner‐join
        raw_data_combined = pd.concat(proc_dfs, axis=1, join='inner')

        # 2) if that produced nothing, try to auto‐align by trimming sample IDs
        if raw_data_combined.empty:
            print("No overlap on sample IDs—trying to align via get_removal_info…")
            # pick an example “extra” ID from the first df’s index
            example_extra_id = str(proc_dfs[0].index[0])
            removal_type = None
            removal_amount = 0

            # compare it to each of the *other* df’s indices until we find a match
            for other_df in proc_dfs[1:]:
                rt, amt = get_removal_info_for_combining(example_extra_id,
                                           other_df.index.tolist())
                if rt is not None and amt > 0:
                    removal_type, removal_amount = rt, amt
                    print(f"  → will remove {amt} chars as a {rt}")
                    break

            # if we found a valid trimming rule, apply it to all proc_dfs
            if removal_type:
                for i, df in enumerate(proc_dfs):
                    if removal_type == 'prefix':
                        proc_dfs[i].index = df.index.map(lambda s: str(s)[removal_amount:])
                    else:  # suffix
                        proc_dfs[i].index = df.index.map(lambda s: str(s)[:-removal_amount])

                # and rebuild the inner-join
                raw_data_combined = pd.concat(proc_dfs, axis=1, join='inner')
                print("  → after trimming, overlap size:", raw_data_combined.shape)
            else:
                print("  ✗ could not find any trimming that creates an overlap")

        # 3) now proceed as before, whether trimmed or not
        combined_proc = raw_data_combined.groupby(by=raw_data_combined.columns, axis=1).apply(
            lambda g: g.mean(axis=1) if isinstance(g.iloc[0, 0], numbers.Number) else g.iloc[:, 0]
        )
        combined_proc = combined_proc.loc[:, ~combined_proc.columns.duplicated()]
        processed_data = combined_proc
    else:
        processed_data = proc_dfs[0].groupby(by=proc_dfs[0].columns, axis=1).apply(
            lambda g: g.mean(axis=1) if isinstance(g.iloc[0, 0], numbers.Number) else g.iloc[:, 0]
        )
    return processed_data

# ================================== #
# Layout of the Data summary tab #
# ================================== #

layout = html.Div([
                    html.H2("Data Summary", style={"fontFamily": "Arial"}),
                    dcc.Interval(
                        id="processed-file-check-interval_dpp",
                        interval=2000,  # 2000ms = 2 seconds
                        n_intervals=0,
                        disabled=True  # We'll enable it when processing is complete
                    ),
                    # Dropdown to select a study
                    dcc.Dropdown(
                        id="selected-studies-dropdown-summary_dpp",
                        placeholder="Select a study",
                        options=[],  # Updated by a callback
                        value=None,
                        style={"width": "300px", "margin": "1rem auto"}
                    ),
                    dbc.Row([
                        # Left column: processed data preview wrapped in Collapse
                        dbc.Col(
                            [
                                # 1) smaller, left-aligned button
                                dbc.Button(
                                    "Process Data",
                                    id="process-data-btn_dpp",
                                    color="primary",
                                    size="md",              # md or sm will shrink the padding/font
                                    className="mt-3 mb-3",  # spacing
                                    style={"width": "180px"}  # or "auto" if you prefer
                                ),

                                # 2) your collapse
                                dbc.Collapse(
                                    [
                                        html.H4("Processed Data Preview"),
                                        html.Div(id="processed-data-table_dpp"),
                                        html.Div(
                                            id="process-data-progress-bar_dpp",
                                            style={
                                                "display": "flex",
                                                "justifyContent": "center",
                                                "alignItems": "center",
                                            },
                                        ),
                                    ],
                                    id="processed-data-collapse_dpp",
                                    is_open=False,
                                ),
                            ],
                            width=9,
                        ),
                        # Right column: Preprocessing sidebar
                        dbc.Col(
                            html.Div(
                                [
                                    # 1) Study details sidebar (read-only)
                                    html.Div(
                                        [
                                            # header (no button here, since it’s summary only)
                                            html.H4("Study details", style={"margin": 0, "marginBottom": "1rem"}),

                                            # Outliers as a disabled text input
                                            html.Div(
                                                [
                                                    dbc.Label("Outliers"),
                                                    dbc.Input(
                                                        id="summary-side-outliers_dpp",
                                                        type="text",
                                                        disabled=True,
                                                        style={"backgroundColor": "#f9f9f9"}
                                                    ),
                                                ],
                                                className="mb-3"
                                            ),

                                            # Control group as a disabled multi-dropdown
                                            html.Div(
                                                [
                                                    dbc.Label("Control group"),
                                                    dcc.Dropdown(
                                                        id="summary-side-control-group_dpp",
                                                        multi=True,
                                                        disabled=True,
                                                        style={"backgroundColor": "#f9f9f9"}
                                                    ),
                                                ],
                                                className="mb-3"
                                            ),

                                            # Case group as a disabled multi-dropdown
                                            html.Div(
                                                [
                                                    dbc.Label("Case group"),
                                                    dcc.Dropdown(
                                                        id="summary-side-case-group_dpp",
                                                        multi=True,
                                                        disabled=True,
                                                        style={"backgroundColor": "#f9f9f9"}
                                                    ),
                                                ],
                                                className="mb-3"
                                            ),
                                        ],
                                        style={
                                            "padding": "1rem",
                                            "border": "1px solid #ccc",
                                            "borderRadius": "5px",
                                        },
                                    ),

                                    html.Br(),

                                    # 2) Data processing sidebar (read-only)
                                    html.Div(
                                        [
                                            html.H4("Data Processing", style={"margin": 0, "marginBottom": "1rem"}),

                                            html.H6("Missing Values Imputation", className="mt-4"),
                                            dbc.Checklist(
                                                id="summary-missing-values-checklist_dpp",
                                                options=[
                                                    {"label": "KNN Imputer",      "value": "knn_imputer"},
                                                    {"label": "Mean Imputer",     "value": "mean_imputer"},
                                                    {"label": "Iterative Imputer","value": "iterative_imputer"},
                                                ],
                                                value=[],  # will be set by callback
                                                inline=False,
                                                style={"pointerEvents": "none", "paddingLeft": "1rem"}
                                            ),

                                            html.H6("Transformation", className="mt-3"),
                                            dbc.Checklist(
                                                id="summary-transformation-checklist_dpp",
                                                options=[
                                                    {"label": "Log Transform",       "value": "log_transform"},
                                                    {"label": "Cube Root Transform", "value": "cube_root"},
                                                ],
                                                value=[],  # will be set by callback
                                                inline=False,
                                                style={"pointerEvents": "none", "paddingLeft": "1rem"}
                                            ),

                                            html.H6("Standardisation", className="mt-3"),
                                            dbc.Checklist(
                                                id="summary-standardisation-checklist_dpp",
                                                options=[
                                                    {"label": "Standard Scaler", "value": "standard_scaler"},
                                                    {"label": "Min-Max Scaler",  "value": "min_max_scaler"},
                                                    {"label": "Robust Scaler",   "value": "robust_scaler"},
                                                    {"label": "Max Abs Scaler",  "value": "max_abs_scaler"},
                                                ],
                                                value=[],  # will be set by callback
                                                inline=False,
                                                style={"pointerEvents": "none", "paddingLeft": "1rem"}
                                            ),
                                        ],
                                        style={
                                            "padding": "1rem",
                                            "border": "1px solid #ccc",
                                            "borderRadius": "5px",
                                            "marginTop": "1.5rem",
                                        },
                                    ),
                                ],
                                style={"display": "flex", "flexDirection": "column"},
                            ),
                            width=3,
                        )
                    ])
                ], style={"padding": "1rem"})

def register_callbacks():
    # Callback to enable the data summary tab if all details have been given for all studies
    @callback(
        [
            Output("summary-tab_dpp",             "disabled"),
            Output("summary-check-interval_dpp",  "disabled"),
            Output("summary-check-interval_dpp",  "n_intervals"),
        ],
        [
            Input("confirm-study-details_dpp",   "n_clicks"),
            Input("confirm-data-processing_dpp", "n_clicks"),
            Input("summary-check-interval_dpp",  "n_intervals"),
        ],
        State("selected-study-store_dpp", "data"),
        prevent_initial_call=True
    )
    def delayed_enable_summary(confirm_details, confirm_processing, n_intervals, selected_studies):
        triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

        # 1) if user just clicked *either* confirm button ➞ kick off the 2s timer
        if triggered_id in ("confirm-study-details_dpp", "confirm-data-processing_dpp"):
            # leave the tab as-is (still disabled), enable the interval, reset its counter
            return no_update, False, 0

        # 2) if the interval just fired (n_intervals == 1) ➞ do the real check
        if triggered_id == "summary-check-interval_dpp" and n_intervals == 1:
            # default: keep disabled unless all studies are good
            disabled = True

            # no studies? bail
            if selected_studies:
                try:
                    with open(SELECTED_STUDIES_FILE, "r", encoding="utf-8") as f:
                        payload = json.load(f).get("studies", {})
                except Exception:
                    logger.exception("Data summary tab - Error reading SELECTED_STUDIES_FILE")
                    payload = {}

                # check every selected study
                ok = True
                for study in selected_studies:
                    info = payload.get(study, {})
                    gf      = info.get("group_filter", {})
                    control = gf.get("Control") or []
                    case    = gf.get("Case")    or []
                    prep    = info.get("preprocessing") or []
                    if not (control and case and isinstance(prep, list) and prep):
                        ok = False
                        break
                disabled = not ok

                if ok:
                    logger.info("Data summary tab - All selected studies have complete details — enabling Data Summary tab")

            # after checking, disable the interval so it doesn't fire again
            return disabled, True, no_update

        # fallback — do nothing
        raise PreventUpdate

    # Callback to hide the process data button once clicked so the data isn't processed twice
    @callback(
        Output("process-data-btn_dpp", "style"),
        Input("process-data-btn_dpp", "n_clicks"),
        prevent_initial_call=True
    )
    def hide_process_button(n_clicks):
        if n_clicks:
            # Hide the button entirely
            return {"display": "none"}
        # Should never get here, but just in case
        return no_update

    # Callback to process the data 
    @callback(
        [Output("process-data-status_dpp", "children"),
        Output("processing-complete-store_dpp", "data")],
        [Input("process-data-btn_dpp", "n_clicks")],
        [State("selected-study-store_dpp", "data"),
        State("project-folder-store_dpp", "data")],
        prevent_initial_call=True
    )
    def process_data_for_all_studies(n_clicks, selected_studies, project_folder):
        if not n_clicks:
            raise PreventUpdate

        if not selected_studies:
            raise PreventUpdate

        # Use project_folder to build the save path, or default to "processed-datasets".
        if project_folder:
            base_save = os.path.join(project_folder, "Processed-datasets")
        else:
            base_save = "processed-datasets"

        final_save_folder = base_save
        os.makedirs(final_save_folder, exist_ok=True)

        # Load preprocessing steps and other study-specific parameters from the selected studies file.
        steps_map = {}
        temp = {}
        if os.path.exists(SELECTED_STUDIES_FILE):
            try:
                with open(SELECTED_STUDIES_FILE) as f:
                    temp = json.load(f)
            except Exception:
                logger.exception("Data summary tab - Error reading SELECTED_STUDIES_FILE")

        # Iterate over all selected studies.
        for study in selected_studies:
            folder = os.path.join(UPLOAD_FOLDER, study)
            if not os.path.isdir(folder):
                logger.error(f"Data summary tab - No folder for study: {study}")
                continue

            details = read_study_details_dpp(folder)
            dataset_source = details.get("Dataset Source", "").lower()
            study_name = details.get("Study Name", "")
            # Retrieve outliers from the SELECTED_STUDIES_FILE payload.
            outliers = temp.get("studies", {}).get(study, {}).get("outliers")

            # If outliers is a string, convert it to a list by splitting on commas.
            if isinstance(outliers, str):
                # Split by comma and remove any extra whitespace from each entry
                outliers = [value.strip() for value in outliers.split(",") if value.strip()]
            try:
                md_filter_local = temp.get("studies", {}).get(study, {}).get("group_filter")
            except Exception:
                logger.exception(f"Data summary tab - Error parsing metadata filter for study {study}")
                #md_filter_local = default_md_filter
                continue

            # Retrieve preprocessing steps.
            preprocessing_steps = temp.get("studies", {}).get(study, {}).get("preprocessing")
            flows = [os.path.splitext(f)[0] for f in os.listdir("data_preprocessing_flows") if f.endswith(".txt")]
            if isinstance(preprocessing_steps, list) and len(preprocessing_steps) == 1 and preprocessing_steps[0] in flows:
                flow_name_for_file = preprocessing_steps[0]
                preprocessing_steps = []  # Replace with actual flow steps as needed.
            else:
                flow_name_for_file = "_".join(preprocessing_steps) if preprocessing_steps else "Untitled"

            # Build a safe filename.
            filename = f"processed_{details.get('Study Name', study)}_{flow_name_for_file}.csv"
            path = os.path.join(final_save_folder, filename)

            # ↪ record the filename back into the SELECTED_STUDIES_FILE payload
            temp["studies"][study]["filename"] = filename
            try:
                with open(SELECTED_STUDIES_FILE, "w", encoding="utf-8") as f:
                    json.dump(temp, f, indent=2)
            except Exception:
                logger.exception("Data summary tab - Error writing filename back into SELECTED_STUDIES_FILE")

            # Retrieve the confirmed group type from the temp file.
            saved_group = temp.get("studies", {}).get(study, {}).get("group_type")
            if not saved_group:
                logger.error(f"Data summary tab - Group not confirmed for study: {study}")
                return html.Div("Group not confirmed for one or more studies."), False

            group_selection = saved_group

            try:
                if dataset_source in (
                    "metabolomics workbench",
                    "original data - refmet ids",
                    "original data - chebi ids",
                ):
                    processed_df = static_preprocess_workbench(
                        folder,
                        preprocessing_steps=preprocessing_steps,
                        outliers=outliers,
                        filter=md_filter_local,
                        selected_group=group_selection, 
                        database_source=dataset_source
                    )
                    group_mapping = {g: group_type for group_type, groups in md_filter_local.items() for g in groups}
                    processed_df['group_type'] = processed_df['Group'].map(group_mapping)
                    if processed_df['group_type'].isnull().any():
                        missing_groups = processed_df.loc[processed_df['group_type'].isnull(), 'Group'].unique()
                        logger.error(f"Data summary tab - The following group names were not found in the metadata filter: {missing_groups}")

                    # 1) Drop the old Group column
                    processed_df = processed_df.drop(columns=["Group"], errors="ignore")

                    # 2) Reset index (this may create “Identifier” if your old index was named that)
                    processed_df = processed_df.reset_index()

                    # 3) Rename or drop whatever columns you need
                    processed_df = (
                        processed_df
                        .rename(columns={"index": "database_identifier"})      # in case you have an unnamed index
                        .drop(columns=["Identifier"], errors="ignore")         # drop the unwanted one
                    )

                    # 4) Reorder
                    cols = processed_df.columns.tolist()
                    rest = [c for c in cols if c not in ("database_identifier", "group_type")]
                    processed_df = processed_df[["database_identifier", "group_type"] + rest]

                    # 5) Save
                    processed_df.to_csv(path, index=False)
                else:
                    # 1) build the pattern
                    pattern = os.path.join(folder, "s_*.txt")

                    # 2) expand the pattern into actual files
                    matches = glob.glob(pattern)

                    # 3) handle zero or many matches, and pick one
                    if not matches:
                        logger.error(f"Data summary tab - No metadata file found matching pattern: {pattern!r}")
                        raise PreventUpdate
                    elif len(matches) > 1:
                        # you could choose the newest, the first, or raise an error
                        matches.sort()  # alphabetical; or sort by os.path.getmtime for newest
                    meta_filepath = matches[0]
                    #meta_filepath = os.path.join(folder, "s_*.txt")
                    if os.path.exists(meta_filepath):
                        try:
                            metadata_df = pd.read_csv(meta_filepath, sep="\t", encoding="unicode_escape")
                        except Exception:
                            logger.exception(f"Data summary tab - Error reading metadata file for study {study}")
                            #metadata_df = default_metadata
                            continue
                    else:
                        metadata_df = default_metadata

                    processed_df = static_preprocess(
                        folder, metadata_df,
                        preprocessing_steps, outliers, md_filter_local,
                        selected_group=group_selection
                    )
                    group_mapping = {g: group_type for group_type, groups in md_filter_local.items() for g in groups}
                    processed_df['group_type'] = processed_df['Group'].map(group_mapping)
                    if processed_df['group_type'].isnull().any():
                        missing_groups = processed_df.loc[processed_df['group_type'].isnull(), 'Group'].unique()
                        logger.error(f"Data summary tab - The following group names were not found in the metadata filter: {missing_groups}")
                    
                    # 1) Drop the old Group column
                    processed_df = processed_df.drop(columns=["Group"], errors="ignore")

                    # 2) Reset index (this may create “Identifier” if your old index was named that)
                    processed_df = processed_df.reset_index()

                    # 3) Rename or drop whatever columns you need
                    processed_df = (
                        processed_df
                        .rename(columns={"index": "database_identifier"})      # in case you have an unnamed index
                        .drop(columns=["Identifier"], errors="ignore")         # drop the unwanted one
                    )

                    # 4) Reorder
                    cols = processed_df.columns.tolist()
                    rest = [c for c in cols if c not in ("database_identifier", "group_type")]
                    processed_df = processed_df[["database_identifier", "group_type"] + rest]

                    # 5) Save
                    processed_df.to_csv(path, index=False)

                logger.info(f"Data summary tab - Saved {filename} -> {final_save_folder}")
            except Exception:
                logger.exception(f"Data summary tab - Error processing study {study}")
                continue

        # After processing all studies, save the SELECTED_STUDIES_FILE into the project folder 
        # under the new name "project_details_file.json".

        # Determine destination path
        if project_folder:
            dest_path = os.path.join(project_folder, "project_details_file.json")
        else:
            dest_path = "project_details_file.json"

        try:
            # 1) Load the incoming payload
            with open(SELECTED_STUDIES_FILE, "r", encoding="utf-8") as f:
                new_payload = json.load(f)
            new_studies = new_payload.get("studies", {})

            # 2) Load existing details if present, otherwise start fresh
            if os.path.exists(dest_path):
                with open(dest_path, "r", encoding="utf-8") as f:
                    existing_payload = json.load(f)
                existing_studies = existing_payload.get("studies", {})
            else:
                existing_payload = {}
                existing_studies = {}

            # 3) Merge: replace or append each new study
            for study_name, details in new_studies.items():
                if study_name in existing_studies:
                    logger.info(f"Data summary tab - Updated details for study {study_name} in {project_folder}")
                else:
                    logger.info(f"Data summary tab - Added new study {study_name} in {project_folder}")
                existing_studies[study_name] = details

            # 4) Write back the merged payload
            merged = {"studies": existing_studies}
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)


        except Exception:
            logger.exception(f"Data summary tab - Error saving details for study {study_name} in {project_folder}")


        processing_complete = True
        logger.info("Data summary tab - All studies have been pre-processed")
        return None, processing_complete 

    # Callback to delay check of if files have being processed
    @callback(
        Output("processed-file-check-interval_dpp", "disabled"),
        Input("processing-complete-store_dpp", "data")
    )
    def toggle_interval(processing_complete):
        # If processing is complete, enable the interval (disabled=False)
        return not processing_complete

    # Callback to display the processed dataset for the study selected in the dropdown
    @callback(
        Output("processed-data-table_dpp", "children"),
        [
        Input("selected-studies-dropdown-summary_dpp", "value"),
        Input("processing-complete-store_dpp", "data")],
        State("project-folder-store_dpp", "data"),  # Added project folder store state
        prevent_initial_call=True
    )
    def display_processed_data_from_file(selected_study, processing_complete, project_folder):
        if not processing_complete:
            return no_update

        # Build the base path using the project folder if available.
        if project_folder:
            base = os.path.join(project_folder, "Processed-datasets")
        else:
            base = "processed-datasets"

        # Determine the final save folder based on the folder choice.
        save_folder = base

        # Load preprocessing steps and determine the flow for the file name.
        steps_map = {}
        if os.path.exists(SELECTED_STUDIES_FILE):
            try:
                with open(SELECTED_STUDIES_FILE) as f:
                    steps_map = json.load(f)
            except Exception:
                logger.exception("Data summary tab - Error reading SELECTED_STUDIES_FILE")

        preprocessing_steps = steps_map.get("studies", {}).get(selected_study, {}).get("preprocessing")
        flows = [os.path.splitext(f)[0] for f in os.listdir("data_preprocessing_flows") if f.endswith(".txt")]
        if isinstance(preprocessing_steps, list) and len(preprocessing_steps) == 1 and preprocessing_steps[0] in flows:
            flow_name_for_file = preprocessing_steps[0]
            # Optionally, assign the actual flow steps if needed, e.g., preprocessing_steps = get_flow_steps(flow_name_for_file)
        else:
            flow_name_for_file = "_".join(preprocessing_steps) if preprocessing_steps else "Untitled"

        # Build the filename and file path.
        filename = f"processed_{selected_study}_{flow_name_for_file}.csv"
        filepath = os.path.join(save_folder, filename)

        """ if not os.path.exists(filepath):
            return html.Div("Processing data, please wait...") """

        try:
            df = pd.read_csv(filepath)
        except Exception:
            logger.exception("Data summary tab - Error reading processed file")
            return html.Div("Error reading processed file")

        #df_head = df.head(100).reset_index().rename(columns={"index": "database_identifier"})
        df_head = df.head(100)
        # now just reflect the DataFrame’s columns *in order*
        columns = [{"name": col, "id": col} for col in df_head.columns]
        fixed_width = "150px"
        processed_table = dash_table.DataTable(
            data=df_head.to_dict("records"),
            columns=columns,
            page_size=10,
            style_table={
                "overflowX": "auto",
                "marginRight": "50px",
                "border": "1px solid #ccc",
                "borderRadius": "5px",
                "boxShadow": "2px 2px 5px rgba(0, 0, 0, 0.1)"
            },
            style_header={
                "backgroundColor": "#f2f2f2",
                "fontFamily": "Arial",
                "fontSize": "16px",
                "fontWeight": "bold",
                "textAlign": "left",
                "border": "1px solid #ddd",
                "padding": "10px"
            },
            style_cell={
                "fontFamily": "Arial",
                "fontSize": "14px",
                "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
                "overflow": "hidden",
                "textAlign": "left",
                "border": "1px solid #ddd",
                "padding": "10px",
                "minWidth": fixed_width,
                "width": fixed_width,
                "maxWidth": fixed_width
            },
            markdown_options={"html": True},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"}
            ]
        )
        return processed_table
    
    # Callback that controls the progress bar updating
    @callback(
        Output("start-ts-store",             "data"),
        Output("folder-interval",            "disabled"),
        Output("hide-progress-interval",     "disabled"),
        Output("process-data-progress-bar_dpp","children"),
        Input("process-data-btn_dpp",        "n_clicks"),
        Input("folder-interval",             "n_intervals"),
        Input("hide-progress-interval",      "n_intervals"),
        State("start-ts-store",              "data"),
        State("project-folder-store_dpp",    "data"),
        State("selected-study-store_dpp",    "data"),
        prevent_initial_call=True
    )
    def progress_bar_control(btn, folder_ticks, hide_ticks, start_store, project_folder, selected_studies):
        # figure out which Input fired
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        total = len(selected_studies or [])
        base  = os.path.join(project_folder, "Processed-datasets") if project_folder else "processed-datasets"

        # 1) Button‐click: seed timestamp, enable folder‐interval, disable hide‐interval, show 0/total
        if trigger == "process-data-btn_dpp":
            if total == 0:
                raise PreventUpdate
            ts = time.time()
            bar0 = dbc.Progress(
                value=0,
                label=f"0/{total} studies",
                striped=True, animated=True,
                style={"width":"250px","height":"20px","marginBottom":"1rem"}
            )
            return {"start": ts}, False, True, bar0

        # 2) Folder‐interval tick: count new files → update bar
        if trigger == "folder-interval":
            if not start_store or "start" not in start_store:
                raise PreventUpdate
            cutoff = start_store["start"]

            # count only .csv created after click, matching our studies
            count = 0
            if os.path.isdir(base):
                seen = {
                    fn for fn in os.listdir(base)
                    if fn.endswith(".csv")
                    and os.path.getmtime(os.path.join(base, fn)) > cutoff
                    and any(fn.startswith(f"processed_{s}_") for s in selected_studies)
                }
                count = len(seen)

            pct = int(count/total*100) if total else 0
            bar = dbc.Progress(
                value=pct,
                label=f"{count}/{total} studies",
                striped=True, animated=True,
                style={"width":"250px","height":"20px","marginBottom":"1rem"}
            )

            done = (count >= total)
            # when done: stop polling (disable folder), start hide timer (enable hide)
            return start_store, done, not done, bar

        # 3) Hide‐interval tick: clear bar & disable hide
        if trigger == "hide-progress-interval":
            # simply clear out the progress bar and stop this timer
            return start_store, True, True, ""

        # fallback
        raise PreventUpdate

    # Callback to display the processed data
    @callback(
        Output("processed-data-collapse_dpp", "is_open"),
        Input("process-data-btn_dpp", "n_clicks"),
        prevent_initial_call=True
    )
    def show_processed_data(n_clicks):
        # As soon as the button is clicked once, open the collapse
        return bool(n_clicks)

    # Callback to ensure that the correct studies are displayed in the dropdown
    @callback(
        [Output("selected-studies-dropdown-summary_dpp", "options"),
        Output("selected-studies-dropdown-summary_dpp", "value")],
        Input("selected-study-store_dpp", "data")
    )
    def update_summary_dropdown(selected_studies):
        if selected_studies:
            options = [{"label": study, "value": study} for study in selected_studies]
            return options, options[0]["value"]
        return [], None
    
    # Callback to populate the sidebars with the options chosen for the study in the dropdown
    @callback(
        [
            # study‐details sidebar
            Output("summary-side-outliers_dpp",      "value"),
            Output("summary-side-control-group_dpp", "options"),
            Output("summary-side-control-group_dpp", "value"),
            Output("summary-side-case-group_dpp",    "options"),
            Output("summary-side-case-group_dpp",    "value"),
            # preprocessing summary
            Output("summary-missing-values-checklist_dpp", "value"),
            Output("summary-transformation-checklist_dpp", "value"),
            Output("summary-standardisation-checklist_dpp","value"),
        ],
        [
            Input("data_pre_process_tabs",                 "active_tab"),
            Input("selected-studies-dropdown-summary_dpp", "value"),
        ],
        prevent_initial_call=True
    )
    def populate_summary_sidebars(active_tab, selected_study):
        if active_tab != "summary" or not selected_study:
            raise PreventUpdate

        # 1) load JSON
        try:
            with open(SELECTED_STUDIES_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            logger.exception("Data summary tab - Error reading SELECTED_STUDIES_FILE")
            raise PreventUpdate

        study = payload.get("studies", {}).get(selected_study, {})

        # ─── study details ─────────────────────────────
        outliers = study.get("outliers") or []

        group_filter = study.get("group_filter", {})
        control = group_filter.get("Control", []) or []
        case    = group_filter.get("Case",   []) or []

        control_options = [{"label": g, "value": g} for g in control]
        case_options    = [{"label": g, "value": g} for g in case]

        # ─── preprocessing summary ─────────────────────
        saved = study.get("preprocessing") or []

        # detect single‐flow case
        flows = [
            os.path.splitext(f)[0]
            for f in os.listdir("data_preprocessing_flows")
            if f.endswith(".txt")
        ]
        if isinstance(saved, list) and len(saved) == 1 and saved[0] in flows:
            steps = get_flow_steps(saved[0])
        else:
            steps = saved

        missing_vals   = [s for s in steps if s in ["knn_imputer", "mean_imputer", "iterative_imputer"]]
        transformation = [s for s in steps if s in ["log_transform", "cube_root"]]
        standardisation= [s for s in steps if s in ["standard_scaler", "min_max_scaler", "robust_scaler", "max_abs_scaler"]]

        return (
            # study details
            outliers,
            control_options, control,
            case_options,    case,
            # preprocessing
            missing_vals,
            transformation,
            standardisation,
        )
    
    
