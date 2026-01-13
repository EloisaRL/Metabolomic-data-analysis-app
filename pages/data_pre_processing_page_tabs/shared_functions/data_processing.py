import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from .helper import (read_study_details_dpp,
                    filter_data_groups,
                    remove_outliers,
                    get_group_value)
import glob, os, requests, logging, re
from io import StringIO
logger = logging.getLogger(__name__)

# ================================= #
# Missing Values Imputation Options #
# ================================= #
def missing_values_knn_impute(data):
    """Uses KNNImputer""" 
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


"""
If multiple datasets are uploaded for the same study, 
due the samples being analysed in different analytical conditions
then these two functions are used. If in each dataset there is a 
slightly different naming of the same patients e.g. 'P_XXX' and 'N_XXX'
due to positive and negative states of Mass Spectrometery machine. This 
code tries to identify this to remove the 'P_' and 'N_' parts (same 
investigation is done for trailing suffix).
"""

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
        logger.error("Data processing - No CSV files found in the folder.")
        return None
    
    def preprocess(df):
        # Workbench CSVs are assumed to have at least the following columns:
        # 'Samples' and 'Class'. Set the index to 'Samples'.

        # only do the merge logic if at least one Samples value has _NEG or _POS
        if (df['Samples']
            .fillna('')           # turn NaN → ""
            .astype(str)          # ensure string dtype
            .str.contains(r'(?:_NEG|_POS)$')
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
            logger.error("Data processing - CSV file must contain 'Samples' and 'Class' columns.")
            return None
        data_filt[identifier_name] = data_filt['Samples']
        data_filt.index = data_filt[identifier_name]

        # --- Group Extraction Logic with Filter Support --- #
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
        # --- End New Group Extraction Logic ---

        if filter is not None and filter != {}:
            data_filt = filter_data_groups(data_filt, filter)

        # Remove the metadata columns
        data_filt = data_filt.drop(columns=['Class', 'Samples', identifier_name])

        # Convert metabolite names to RefMet IDs using the Workbench API.
        if database_source == "metabolomics workbench":
            mets_url = 'https://www.metabolomicsworkbench.org/rest/study/study_id/repl/metabolites'
            try:
                mets = requests.get(mets_url.replace('repl', study_name)).text
                mets_df = pd.read_json(StringIO(mets)).T
                mets_dict = dict(zip(mets_df['metabolite_name'], mets_df['refmet_name']))
                # make sure 'Group' is identity-mapped
                mets_dict['Group'] = 'Group'
                data_filt.columns = data_filt.columns.map(mets_dict)  

            except Exception:
                logger.exception("Data processing - Error converting metabolie names to RefMet IDs")

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

        # Drop rows only if all non‑Group columns are NaN (re added - DOUBLE CHECK)
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
            logger.exception(f"Data processing - Error reading file {f}")
            continue
        proc_data = preprocess(df)
        if proc_data is None:
            continue

        # ----------------------------------- #
        # Apply optional preprocessing steps: #
        # ----------------------------------- #
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


    files = glob.glob(f"{folder}/*.tsv")
    
    if len(files) == 0:
        logger.error("Data processing - No assay files found in the folder.")
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
        logger.error("Data processing - No valid processed data from assay files.")
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
        # Numeric columns: average duplicates
        num_cols = raw_data_combined.select_dtypes(include=np.number).columns
        avg_num   = raw_data_combined[num_cols].groupby(level=0, axis=1).mean()

        # Non-numeric (e.g., 'Group'): take the first non-null across duplicates
        non_cols = raw_data_combined.columns.difference(num_cols)
        non_num  = (
            raw_data_combined[non_cols]
            .groupby(level=0, axis=1)
            .agg(lambda df: df.bfill(axis=1).ffill(axis=1).iloc[:, 0])
        )

        processed_data = pd.concat([avg_num, non_num], axis=1)
    else:
        """ processed_data = (
            proc_dfs[0].T
            .groupby(level=0)
            .apply(lambda g: g.mean(axis=0) if isinstance(g.iloc[0, 0], numbers.Number) else g.iloc[:, 0])
            .T
        ) """
        df0 = proc_dfs[0]  # just for clarity

        # --- Split numeric vs non-numeric columns ---
        num_cols = df0.select_dtypes(include=np.number).columns
        non_cols = df0.columns.difference(num_cols)

        # --- 1) For numeric columns: average duplicates ---
        avg_num = df0[num_cols].groupby(level=0, axis=1).mean()

        # --- 2) For non-numeric columns (like "Group"): take the first non-null value ---
        non_num = (
            df0[non_cols]
            .groupby(level=0, axis=1)
            .agg(lambda df: df.bfill(axis=1).ffill(axis=1).iloc[:, 0])
        )

        # --- 3) Combine back together ---
        processed_data = pd.concat([avg_num, non_num], axis=1)
    return processed_data