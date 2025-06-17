# Metabolomic Data Analysis App

An interactive Dash application for visualizing and analyzing differential metabolites and pathways across single- and multi-study datasets. By leveraging harmonized metabolite annotations, the app enables you to:

- Explore metabolic data from individual studies
- Integrate large-scale public repositories (e.g., MetaboLights, MetabolomicsWorkbench) or your own data (using RefMet or ChEBI IDs)
- Uncover multi-study signatures and trends
- Maximize reusability and reproducibility of metabolomics data

With a focus on standardization and integration, this tool supports scientific discovery through comprehensive, accessible data visualization and analysis.

![image](https://github.com/user-attachments/assets/7b7ffe0e-f397-4394-9858-11038453f6c3)


---

## Table of Contents

- [Features](#features)  
- [Getting Started](#getting-started)  
  - [Clone the Repository](#clone-the-repository)
  - [Set up the Environment](#set-up-the-environment) 
  - [Install Dependencies](#install-dependencies)  
  - [Run the App](#run-the-app)  
- [Usage](#usage)  
  - [Single-Study Analysis](#single-study-analysis)  
  - [Multi-Study Analysis](#multi-study-analysis)  
- [Example Data (“Dummy Project”)](#example-data-dummy-project)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Features

1. **Interactive Visualizations**  
   - Dynamic plots of differential metabolites, differential pathways, clustering, and more.  

2. **Single-Study Analysis**  
   - Upload your own processed data (RefMet/ChEBI IDs).  
   - Filter, sort, and visualize metabolites and pathways from a single experiment.

3. **Multi-Study Analysis**  
   - Integrate data from multiple sources (e.g., public repositories or your own studies).  
   - Harmonize metabolite annotations across studies to find common signatures.  
   - Compare study-specific vs. cross-study trends in a unified view.

4. **Public Repository Integration**  
   - Pull annotated metabolomic data directly from MetaboLights or MetabolomicsWorkbench.  
   - Harmonize annotations using RefMet or ChEBI IDs under the hood.

5. **Reproducibility and Reusability**  
   - Built-in support for standardized formats and nomenclature.  
   - Easy export of (high resolution) plots, tables, and filtered datasets for downstream analysis or publication.

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/EloisaRL/Metabolomic-data-analysis-app.git
cd metabolight-data-analysis-app
```

### Set up the Environment
Ensure you are using **Python 3.12.9** (or create a virtual environment with it).    

**Create and activate a virtual environment (optional but recommended):**

```bash
python -m venv venv
venv\Scripts\activate      # On macOS/Linux: source venv/bin/activate
```
### Install Dependencies
Install all required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Run the App 

```bash
python index.py
```
You’ll see output similar to:

```console
Dash is running on http://127.0.0.1:8050/  (Press CTRL+C to quit)
```
**Ctrl+Click** (or **Cmd+Click** on macOS) the link to open the app in your browser.

## Usage

The app allows users to perform two types of analysis: single-study analysis and multi-study analysis. Both types of analysis perform differential testing to identify either differential metabolites or differential pathways, but they differ in the way these results are visualised.

### Single-Study Analysis

Single-study analysis produces a box plot of the top 10 metabolites/pathways (the number of metabolites/pathways visualised can be determined by the user) and a csv table with the all the differential metabolites/pathways for the selected study.

#### Differential metabolites tab
![Differential-metabolites-tab-ssa-page](https://github.com/user-attachments/assets/8b208f0e-ec04-4007-b4c9-2c640aa35198)

#### Differential pathways tab
![Differential-pathways-tab-ssa-page](https://github.com/user-attachments/assets/493cc472-b44d-409e-8221-e4dd288f1e1a)


### Multi-Study Analysis

Multi-study analysis produces upset plots of the co-occuring metabolites and differential metabolites (metabolites are matched based on the metabolite name, **not** the ChEBI id) for the selected studies. Also, cytoscape-based network graphs are produced as either differential metabolites (metabolites are matched based on the metabolite name) or differential pathways as the nodes, for the selected studies. 

#### Upset plots tab
![Upset-plots-tab-msa-page](https://github.com/user-attachments/assets/4900b142-2d4e-44db-b2da-a0e4aa385b00)

#### Network plots tab - differential metabolites network graph
![Network-plots-tab-msa-page](https://github.com/user-attachments/assets/3caadc60-d008-4df3-868b-d21429a96b4e)

#### Network plots tab - differential pathways network graph
![Network-plots-tab-msa-page-pathways](https://github.com/user-attachments/assets/253fc0c2-f06b-4272-a49b-fd5b351d762b)


## Example Data (“Dummy Project”)

To start using the single-study and multi-study analysis tabs now, you can use the data in the 'Dummy project' which contains processed data from all studies that currently uploaded. Or you can upload the processed data directly from the 'Dummy project' on to Metaboanalyst for data analysis.
