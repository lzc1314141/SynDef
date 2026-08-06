# SynDef: Syntax-Constrained Semantic Decoding for Line-Level Defect Prediction

This repository contains the complete pipeline to reproduce the experiments described in our work. The workflow covers data preprocessing, DFA data generation, model training, and final evaluation.

## Prerequisites

- **Dataset**: The raw dataset is available at [https://github.com/lzc1314141/Datasets](https://github.com/lzc1314141/Datasets)
- **Supplementary files**: All additional required files are provided in the official source code of the SPLICE paper. Please download them from the SPLICE repository before starting.

## Environment Setup

Download the repository, then set up the Python environment using Conda:

```bash
cd SynDef

# Create and activate conda environment
conda create -n syndef_env python=3.9.5
conda activate syndef_env

# Install dependencies
pip install -r requirements.txt
```


## Pipeline Overview

The whole process consists of four main stages:

1. Generate DFA data – run three scripts in the `line-automat` folder.
2. Organise the generated datasets – separate first-version files for training and the rest for source data.
3. Train models – train both the automat and transformer models.
4. Obtain final results – run the SynDef script.

## Step-by-Step Instructions

### A. Generate DFA Data

Navigate to the `line-automat` directory and execute the following scripts **in the given order**:

```bash
cd line-automat
python to_java.py                # Step 1a
python clean_sourcedata.py       # Step 1b
python lineautomat_sourcedata.py # Step 1c
```

After these steps, the DFA data will be successfully created.

> **Note**: The `to_java.py` script produces multiple dataset files of different versions. These files will be used in the next step.

### B. Organise the Generated Datasets

After running `to_java.py`, you will obtain several dataset files.

- Move all **first-version** dataset files into `SynDef/X--traindata/`
- Move all **remaining** dataset files (other versions) into `SynDef/sourcedata/`

Make sure the files are placed directly under their respective folders (no subdirectories needed).

### C. Train the Models

Now you can train both models. Run the following scripts (order does not matter):

```bash
python automat.py      # Train the automat model
python transformer.py  # Train the transformer model
```

These scripts are typically located in the project root. Training will produce model checkpoints and intermediate outputs.

### D. Obtain Final Results

Finally, execute the SynDef script to generate the evaluation results:

```bash
python Syndef.py
```

The final metrics and predictions will be printed to the console or saved to output files.

## Important Notes

- **File paths**: All scripts contain hard-coded or relative paths. Please modify them to match your local directory structure before running any script.
- **Missing files**: If you encounter errors about missing dependencies or data, refer to the official SPLICE source code repository to download the required supplementary materials.

## Project Structure

Below is an overview of the file structure to help you understand the organization of the repository:

```
.
├─ README.md
├─ line-automat/
│  ├─ clean_sourcedata.py
│  ├─ lineautomat_sourcedata.py
│  └─ to_java.py
└─ SynDef/
   ├─ RQ/
   │  ├─ RQ1/
   │  │  ├─ Hit_and_Over.R
   │  │  ├─ TN_Upset.R
   │  │  └─ TP_Upset.R
   │  └─ RQ2/
   │     ├─ RQ2_Compare.R
   │     └─ SynDef.R
   ├─ Syndef.py
   ├─ automat.py
   └─ transformer.py
```

> **Note**: Directories such as `File-level/`, `automat-traindata/`, `transform-traindata/`, `sourcedata/`, `n_gram_result/`, and `line-automat/sourcefile/` are created automatically when the scripts run, so they are not listed here.

## Contact

For questions or issues, please open an issue in this repository or reach out to the authors directly.

Happy reproducing!
