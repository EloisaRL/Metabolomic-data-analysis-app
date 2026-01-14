# Metabolomic Data Analysis App

**User guide:** https://doi.org/10.5281/zenodo.15720017

An interactive Dash application for visualising and analysing differential metabolites and differential pathways in a single study or across multiple studies.  
The app automatically harmonises metabolite annotations (RefMet names or ChEBI IDs), letting you seamlessly explore your own data or public datasets from MetaboLights and Metabolomics Workbench in one place.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/f28ff8f3-4e41-4baf-9c69-7901be25da60"
    alt="Workflow overview"
    width="600"
  />
</p>



---

## Table of Contents

- [Getting Started](#getting-started)  
  - [Option A — Run with Docker (Recommended)](#option-a--run-with-docker-recommended)
  - [Option B — Run Locally with Python](#option-b--run-locally-with-python) 
  - [Data Requirements](#data-requirements)
- [Usage](#usage)  
  - [Single-Study Analysis](#single-study-analysis)  
  - [Multi-Study Analysis](#multi-study-analysis)  
- [Example Data (“Demo Project”)](#example-data-demo-project)  
- [Contributing](#contributing)  
- [License](#license)  


---

## Getting Started

There are **two ways** to run the application:

- **Option A (Recommended): Run with Docker (no Python setup required)**
- **Option B: Run locally with Python (for development or contributors)**


### Option A — Run with Docker (Recommended)
This is the **simplest and most reproducible way** to run the app.

**Prerequisites**
- Docker
- Docker Compose

#### Steps
```bash
git clone https://github.com/EloisaRL/Metabolomic-data-analysis-app.git
cd Metabolomic-data-analysis-app
docker compose up
```
Docker will pull the published image from Docker Hub and start the app.

Once running, open your browser at:
```arduino
http://localhost:8050
```

To stop the app:
```bash
docker compose down
```


### Option B — Run Locally with Python
This option is useful if you want to **develop or modify the code**.

**Prerequisites**
- Python **3.12.9**

#### Steps

Clone the repository:
```bash
git clone https://github.com/EloisaRL/Metabolomic-data-analysis-app.git
cd Metabolomic-data-analysis-app
```

Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate      # On macOS/Linux: source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
python index.py
```

You should see output similar to:
```bash
Dash is running on http://127.0.0.1:8050/
```

Open the app in your browser at:
```arduino
http://localhost:8050
```


### Data Requirements

Before uploading a study dataset, make sure your data follow these guidelines:

1. **File Format & Structure**  
   - Dataset file type - Use CSV with a single header row (no blank rows).  
   - Dataset contents - One column for patient ids, one column for patient group (e.g. contains Covid-19, Healthy) and all other columns are for metabolites (either RefMet names or ChEBI ids as column names).
   - Study folder contents - One study can have multiple dataset csv files this happens because each file corresponds to the same patient samples profiled under different analytical conditions (e.g. ion mode, column chemistry, collision energy). When you click the upload new study button (in the Data Pre-Processing tab) you must upload all of these files in the pop up before clicking confirm study upload. 
   - (If the dataset originates from MetaboLights) Supply a separate sample metadata file (it will begin with 's_....txt') containing the patient group information needed for the app to function. When you click the upload new study button (in the Data Pre-Processing tab) you must included this metadata data file in the pop up before clicking confirm study upload, also ensure that you specifiy that the study originates from MetaboLights. 

2. **Organism**  
   - If you plan to run the differential pathway analysis studies must be from human (Homo sapiens) paitents since only the human reactome file is used in this app currrently.

3. **Sample & Phenotype Recommendations**  
   - Include enough samples per group to support reliable statistical testing. (Recommended more than 10 per group) 
   - Have at least two discrete phenotypes or outcome groups—if your outcome is continuous (e.g. BMI), discretise it (e.g. “high” vs. “low”).

4. **Metabolite Coverage**  
   - Aim for broad coverage of identified metabolites that map successfully to ChEBI IDs. (Recommended more than 20)
   - If a metabolite has more than 50% missing values across all patients that metabolite will be removed during data preprocessing in the app.

5. **Cross-Study Overlap (multi-study mode)**  
   - Ensure there is some overlap of differential metabolite identifiers across studies to enable comparative analyses.  

6. **Pathway Analysis Considerations**  
   - Provide 2 or more metabolites mapping to each reactome pathway to allow for differential pathway testing (if all reactome pathways have only one metabolite mapping to it then differential pathway testing cannot be performed).

> **Tip:** These are general guidelines. Tailor them to your specific experimental design, data quality, and scientific goals.
> 
> **Step-by-step help in uploading data:** Please refer to the user guide.

---

## Usage

The app allows users to perform two types of analysis: single-study analysis and multi-study analysis. Both types of analysis perform differential testing to identify either differential metabolites or differential pathways, but they differ in the way these results are visualised.

### Single-Study Analysis

Single-study analysis produces a box plot of the top 10 differntial metabolites/pathways and a csv table with the all the differential metabolites/pathways for the selected study.

#### Differential metabolites tab
<img width="1918" height="867" alt="Differential_metabolites_SSA" src="https://github.com/user-attachments/assets/2307cb01-b38e-485c-b03c-9d33ec2b973b" />


#### Differential pathways tab
<img width="1918" height="866" alt="Differential_pathways_SSA" src="https://github.com/user-attachments/assets/de55795a-0d85-48d7-9bde-b09330ccfbb4" />


### Multi-Study Analysis

Multi-study analysis produces upset plots of the co-occuring metabolites and differential metabolites (metabolites are matched based on the metabolite name, **not** the ChEBI id) for the selected studies. Also, cytoscape-based network graphs are produced as either differential metabolites (metabolites are matched based on the metabolite name) or differential pathways as the nodes, for the selected studies. 

#### Upset plots tab
<img width="1918" height="866" alt="Upset_plots" src="https://github.com/user-attachments/assets/9c688757-2757-4f26-baeb-4c713565881e" />


#### Network plots tab - differential metabolites network graph (pie chart nodes)
<img width="1918" height="866" alt="Network_diff_met_pie_charts" src="https://github.com/user-attachments/assets/3764309d-3c8b-484d-b0af-949dea2d2448" />

#### Network plots tab - differential metabolites network graph (t-statistic nodes)
<img width="1918" height="866" alt="Network_diff_met_t_stat" src="https://github.com/user-attachments/assets/c6521338-3d2c-4bad-b419-cce78d4cd93c" />

#### Network plots tab - differential metabolites network graph (bipartite)
<img width="1917" height="867" alt="Network_diff_met_bipartite" src="https://github.com/user-attachments/assets/8ca6ce59-1093-4d5d-8cbe-542f1ebff1df" />


#### Network plots tab - differential pathways network graph
Differential pathways shown must be found differential in 2 or more studies:
<img width="1918" height="867" alt="Network_pathway_min_2" src="https://github.com/user-attachments/assets/eda59b42-9525-4335-acd3-d4cd9013ed42" />

Differential pathways shown must be found differential in 3 or more studies:
<img width="1918" height="868" alt="Network_pathway_min_3" src="https://github.com/user-attachments/assets/4f8d107e-ac0d-4517-94f5-37ed8cf041ff" />


---

## Example Data (“Demo Project”)

To help you get started, we’ve included a **Demo_project** folder containing preprocessed files originating from MetaboLights and Metabolomics Workbench studies:

```bash
/Demo_project
├── /Processed-datasets
    ├── processed_MTBLS1866_knn_imputer_log_transform_standard_scaler.csv
    ├── processed_MTBLS2014_knn_imputer_log_transform_standard_scaler.csv
    ├── processed_ST000041_knn_imputer_log_transform_standard_scaler.csv
    └── ...
├── project_details_file.json
└── /Plots
    ├── /Preprocessing-analysis
        ├── /PCA-plots
        ├── /Residual-plots
        └── /Box-plots
    ├── /Single-study-analysis
        ├── /Differential-metabolites-box-plots
        ├── /Differential-metabolites-table-plots
        ├── /Differential-pathway-box-plots
        └── /Differential-pathway-table-plots
    └── /Multi-study-analysis
        ├── /Co-occurring-metabolites-upset-plots
        ├── /Differential-co-occurring-metabolites-upset-plots
        ├── /Differential-metabolites-network-plots
        └── /Differential-pathway-network-plots
```


### 1. Processed Datasets

- **Filename convention**:  
  `processed_<STUDY_ID>_<preprocessing_steps>.csv`  
  e.g. `processed_MTBLS1866_knn_imputer_log_transform_standard_scaler.csv`
- **Metabolite columns headers**:  
  - For MetaboLights studies (e.g. `MTBLS1866`), column headers are **ChEBI IDs**.  
  - For Metabolomics Workbench studies (e.g. `ST000041`), column headers are **RefMet names**.
  - For your own studies uploaded with ChEBI ids or RefMet names the column headers will not change.
- **Usage**:  
  - Processed datasets are in saved in the conventional format as those accepted by MetaboAnalyst (https://www.metaboanalyst.ca/MetaboAnalyst/home.xhtml). However, when you load these CSVs into MetaboAnalyst, those same IDs/names will appear on all plots and tables.

**Studies included in this Demo Project**
| Study ID      | Title                                                                                                                                                                       | DOI                                                                                          |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **MTBLS1866** | Large-Scale Plasma Analysis Revealed New Mechanisms and Molecules Associated with the Host Response to SARS-CoV-2                                                           | [https://doi.org/10.3390/ijms21228623](https://doi.org/10.3390/ijms21228623)                 |
| **MTBLS2014** | Integrative Modeling of Quantitative Plasma Lipoprotein, Metabolic, and Amino Acid Data Reveals a Multiorgan Pathological Signature of SARS-CoV-2 Infection                 | 10.1021/acs.jproteome.0c00519                                                                |
| **MTBLS2224** | Sex-specific metabolic shifts in plasma of COVID-19 patients after cure                                                                                                     | 10.1016/j.csbj.2021.03.039                                                                   |
| **MTBLS2291** | Metabolome and exposome profiling of the biospecimens from COVID-19 patients in India (Blood plasma assay)                                                                  | [https://doi.org/10.36233/0372-9311-161](https://doi.org/10.36233/0372-9311-161)             |
| **MTBLS2336** | Metabolomic/lipidomic profiling of COVID-19 and individual response to tocilizumab                                                                                          | [https://doi.org/10.1371/journal.ppat.1009243](https://doi.org/10.1371/journal.ppat.1009243) |
| **MTBLS2542** | The trans-omics landscape of COVID-19                                                                                                                                       | [https://doi.org/10.1038/s41467-021-24482-1](https://doi.org/10.1038/s41467-021-24482-1)     |
| **MTBLS3852** | Amino Acid Metabolism is Significantly Altered at the Time of Admission in Hospital for Severe COVID-19 Patients: Findings from Longitudinal Targeted Metabolomics Analysis | [https://doi.org/10.1128/spectrum.00338-21](https://doi.org/10.1128/spectrum.00338-21)       |
| **MTBLS6739** | Targeted plasma metabolomics combined with machine learning for the diagnosis of acute SARS-CoV-2                                                                           | [https://doi.org/10.3389/fmicb.2022.1059289](https://doi.org/10.3389/fmicb.2022.1059289)     |
| **ST000041**  | High PUFA diet in humans                                                                                                                                                    | doi:10.21228/M8X59D                                                                          |
| **ST000284**  | Colorectal Cancer Detection Using Targeted Serum Metabolic Profiling                                                                                                        | doi:10.21228/M8FG61                                                                          |
| **ST000899**  | Alterations in Lipid, Amino Acid, and Energy Metabolism Distinguish Crohn Disease from Ulcerative Colitis and Control Subjects by Serum Metabolomic Profiling               | doi:10.21228/M8W983                                                                          |
| **ST000974**  | GC6-74 metabolomic of TB (Part 1: Plasma)                                                                                                                                   | doi:10.21228/M8KQ3J                                                                          |
| **ST001412**  | Metabolomics study in Plasma of Obese Patients with Neuropathy Identifies Potential Metabolomics Signatures                                                                 | doi:10.21228/M8K688                                                                          |
| **ST001420**  | Metabolomic analysis of patients with recurrent angina                                                                                                                      | doi:10.21228/M8SQ55                                                                          |
| **ST001736**  | The COVIDome Explorer Researcher Portal (blood plasma)                                                                                                                      | doi:10.21228/M8739H                                                                          |
| **ST001933**  | Absolute quantification of plasma cytokines and metabolome reveals the glycylproline regulating antibody-fading in convalescent COVID-19 patients                           | doi:10.21228/M8RH8X                                                                          |
| **ST002016**  | Metabolomics of COVID patients                                                                                                                                              | doi:10.21228/M88715                                                                          |
| **ST002100**  | Functional metabolomics-based molecular profiling of acute and chronic hepatitis (Human Serum Metabolomics)                                                                 | doi:10.21228/M8ST3F                                                                          |
| **ST002301**  | Serum metabolomics profiling identifies new predictive biomarkers for disease severity in COVID-19 patients                                                                 | doi:10.21228/M86998                                                                          |
| **ST002428**  | Mass Spectrometry-based Proteomic and Metabolomic profiling of serum samples for discovery and validation of Tuberculosis diagnostic biomarker signature                    | doi:10.21228/M8SM54                                                                          |
| **ST002498**  | Plasma Metabolomics Profiling of 580 Patients from the Weill Cornell Medicine Early Detection Research Network Prostate Cancer Cohort                                       | doi:10.21228/M86H7K                                                                          |
| **ST002829**  | Nucleotide, phospholipid, and kynurenine metabolites are robustly associated with COVID-19 severity and time of plasma sample collection in a prospective cohort study      | doi:10.21228/M8SM6H                                                                          |


### 2. Project Details (`project_details_file.json`)

This JSON file contains, for each study:

| Field                  | Description                                                                                               |
|------------------------|-----------------------------------------------------------------------------------------------------------|
| `filename`             | Name of the processed CSV in `Processed-datasets/` (e.g. `processed_MTBLS1866_knn_imputer_….csv`).        |
| `study_id`             | The original study identifier (e.g. `MTBLS1866`, `ST000041`).                                             |
| `group_type`           | Metadata field used for grouping (e.g. `Factor Value[Medical case]`).                                     |
| `preprocessing`        | Ordered list of preprocessing steps applied (e.g. `["knn_imputer", "log_transform", "standard_scaler"]`). |
| `outliers`             | Sample ID(s) flagged as outliers (e.g. `"SA 8"` or `["SA 3","SA 8"]`).                                    |
| `group_filter`         | Maps “Control” and “Case” labels to the raw metadata values:                                              |
| `Control`              | List of values interpreted as the control group (e.g. ["HEALTHY"]).                                       |
| `Case`                 | List of values interpreted as the case group (e.g. ["COVID-19"]).                                         |

> **Note on metadata conventions**  
> - **MetaboLights** uses metadata columns prefixed with `Factor_Value` to indicate sample groups.  
> - **Metabolomics Workbench** embeds multiple group labels in the `Class` column, separated by `|`.  
>  
> In `project_details_file.json`, we parse these into a clean `group_type` object so you can switch between different patient stratifications (e.g., control vs. case, male vs. female, treated vs. untreated).


### 3. Plots

All plots you generate will be saved under the matching subfolders in `/Demo_project/Plots/` using the filename you provide at export. **Exception**: network plots (both metabolite and pathway networks) are automatically downloaded to your computer’s default Downloads folder. After running your multi-study analysis, please move those network plot files into the appropriate `/Demo_project/Plots/Multi-study-analysis/…` folders to keep everything organized.

> **Note:** Each subfolder already contains a few example plots generated by the app, so you can see the typical output format and file naming conventions before you start.


🚀 **Quick Start**  
After you’ve browsed the folders and seen the example outputs above, you’re all set—just open the app, select **Demo Project** in the project selection dropdown (in the Single-Study or Multi-Study tab), and start exploring immediately!

---

## Contributing

We welcome contributions! Here’s how to get started:

**Report issues**  
   - 🔍 Found a bug? Open an issue with steps to reproduce.  
   - 💡 Have an idea? Open an issue describing the feature.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
