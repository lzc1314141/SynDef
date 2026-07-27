README
This repository contains the complete pipeline to reproduce the experiments described in our work. The workflow covers data preprocessing, DFA data generation, model training, and final evaluation.

Prerequisites
Dataset: The raw dataset is available at [DATASET_LINK] (to be filled in by the user).

Supplementary files: All additional required files (e.g., helper scripts, pre‑trained components) are provided in the official source code of the SPLICE paper. Please download them from the SPLICE repository before starting.

Pipeline Overview
The whole process consists of four main stages:

Generate DFA data – run three scripts in the line-automat folder.

Organise the generated datasets – separate first‑version files for training and the rest for source data.

Train models – train both the automat and transformer models.

Obtain final results – run the SynDef script.

Step‑by‑Step Instructions
1. Generate DFA Data
Navigate to the line-automat directory and execute the following scripts in the given order:

bash
cd line-automat
python to_java.py                # Step 1a
python clean_sourcedata.py       # Step 1b
python lineautomat_sourcedata.py # Step 1c
After these steps, the DFA data will be successfully created.

Note: The to_java.py script produces multiple dataset files of different versions. These files will be used in the next step.

2. Organise the Generated Datasets
After running to_java.py, you will obtain several dataset files.

Move all first‑version dataset files into:

text
SynDef/X--traindata/
Move all remaining dataset files (other versions) into:

text
SynDef/sourcedata/
Make sure the files are placed directly under their respective folders (no subdirectories needed).

3. Train the Models
Now you can train both models. Run the following scripts (order does not matter):

bash
python automat.py      # Train the automat model
python transformer.py  # Train the transformer model
These scripts are typically located in the project root. Training will produce model checkpoints and intermediate outputs.

4. Obtain Final Results
Finally, execute the SynDef script to generate the evaluation results:

bash
python Syndef.py
The final metrics and predictions will be printed to the console or saved to output files.

Important Notes
File paths: All scripts contain hard‑coded or relative paths. Please modify them to match your local directory structure before running any script.

Missing files: If you encounter errors about missing dependencies or data, refer to the official SPLICE source code repository to download the required supplementary materials.


Contact
For questions or issues, please open an issue in this repository or reach out to the authors directly.

Happy reproducing!
