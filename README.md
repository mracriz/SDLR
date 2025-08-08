# SDLR: Knowledge Distillation for Learning to Rank 🚀

This repository provides an implementation of the Knowledge Distillation technique for Learning to Rank models, based on the `allRank` framework. The primary workflow, **SDLR**, involves training a "Teacher" model to guide the learning of a smaller, more efficient "Student" model.

The project is fully integrated with **MLflow** for robust experiment tracking, comparison, and management. It is designed to be highly configurable through a central control panel (`experiments.yaml`) and executed via a single, unified command script (`main.py`).

## acknowledgments and Origin

This work is an implementation and adaptation of the original SDLR research. For a complete understanding of the methodology, please refer to the original work.

  * **Original SDLR Repository:** [https://github.com/sanazkeshvari/Papers/tree/main/SDLR](https://github.com/sanazkeshvari/Papers/tree/main/SDLR)

Both this project and the original are built upon the excellent `allRank` framework.

  * **allRank Framework:** [https://github.com/allegro/allRank](https://github.com/allegro/allRank)

## 🛠️ Prerequisites and Installation

Before you begin, ensure you have **Python 3.10** or higher installed.

### Step 1: Create and Activate a Virtual Environment (`venv`)

It is highly recommended to use a virtual environment to isolate project dependencies.

```bash
# From the project's root directory (SDLR), create the venv
python3 -m venv venv

# Activate the environment (on macOS/Linux)
source venv/bin/activate
```

> *On Windows, the activation command is `venv\Scripts\activate`*

### Step 2: Install Dependencies

The installation is a two-part process: first, we install the standard packages from the root `requirements.txt`, and then we "link" the local Teacher and Student projects to our environment.

```bash
# 1. Install the main library dependencies from the root file
pip install -r requirements.txt

# 2. Make the Teacher's code "discoverable" by Python
pip install -e Teacher/allRank-master/

# 3. Make the Student's code "discoverable" by Python
pip install -e Student/allRank-master/
```

> **Important Note ⚠️:** The `pip install -e` commands are **essential**. They create a link to your local projects, which resolves the `ModuleNotFoundError` and `pkg_resources.DistributionNotFound` errors, ensuring that Python and MLflow can find the training scripts.

## ⚙️ Experiment Configuration

All experiments are controlled from a single "control panel" file: **`experiments.yaml`**, located in the root of the project. **This is the only file you need to edit to define and configure your experiments.**

The internal `.json` files (e.g., in `configs/`) should be treated as **templates** for model architecture and loss functions. The user-specific settings, like data paths, are all handled in `experiments.yaml`.

### `experiments.yaml` Structure

Here is an example of how to define experiments:

```yaml
# This is the control panel for all your experiments.
# Run an experiment by its key, e.g., `python3 main.py sdlr_ips_run`

# --- An SDLR (Teacher -> Student) Experiment ---
sdlr_ips_run:
  experiment_name: "SDLR_IPS_Experiment"  # The name that will appear in the MLflow UI
  name: "sdlr"                         # The workflow type: "sdlr" or "single"
  teacher:
    config_template: "configs/teacher/sdlr_ips_teacher.json"
    data_path: "/path/to/your/teacher/data_folder"
  student:
    config_template: "configs/student/sdlr_ips_student.json"
    data_path: "/path/to/your/student/data_folder"
  inference_data: "/path/to/your/final_test_set.txt" # Set to null to skip evaluation
  data_options:
    noise_percent: 0.2
    max_noise: -1.0
    data_percent: 0.8

# --- A Single Model (Baseline) Experiment ---
neural_ndcg_baseline:
  experiment_name: "NeuralNDCG_Baseline" # The name that will appear in the MLflow UI
  name: "single"
  model:
    base_dir: "Teacher/allRank-master" # Which codebase to use (usually Teacher's)
    config_template: "configs/baselines/neural_ndcg_config.json"
    data_path: "/path/to/your/baseline/data_folder"
  inference_data: "/path/to/your/final_test_set.txt" # Set to null to skip evaluation
  data_options: # "Clean" settings for a standard baseline
    noise_percent: 0.0
    max_noise: 0.0
    data_percent: 1.0
```

  * **`experiment_name`**: The name displayed in the MLflow UI.
  * **`name`**: The workflow type. Use `sdlr` for Teacher/Student distillation or `single` for baselines.
  * **`config_template`**: The path to the JSON file that defines the model architecture and loss function.
  * **`data_path`**: The absolute path to the data folder. This folder **must** contain `train.txt`, `vali.txt`, and `test.txt`.
  * **`inference_data`**: The absolute path to the final test set for evaluation. Set this to `null` or remove the line to only train the model without evaluating it.
  * **`data_options`**: Controls data sampling and noise injection for your custom data loaders.

## 💾 Data Preparation (CSV to SVM Rank)

All training, validation, and test files (`train.txt`, `vali.txt`, `test.txt`) **MUST BE IN THE SVM Rank FORMAT**.

This repository includes a utility to convert your data from `.csv` to the required SVM Rank format. The script is located at `utils/csv_to_svm.py`.

### How to Use the Converter

Imagine you have a `my_data.csv` file. You can convert it using the following command from the `SDLR` root directory:

```bash
python3 utils/csv_to_svm.py \
  --input-csv /path/to/my_data.csv \
  --output-svm /path/to/converted_data.txt \
  --qid-col query_id_column_name \
  --relevance-col relevance_column_name \
  --feature-cols feature_A feature_B feature_C
```

## ⚡ How to Run Experiments

All experiments are executed through the unified `main.py` script. You just need to provide the name of the experiment you want to run (the key from your `experiments.yaml` file).

**To run the SDLR experiment defined above:**

```bash
python3 main.py sdlr_ips_run
```

**To run the NeuralNDCG baseline experiment:**

```bash
python3 main.py neural_ndcg_baseline
```

## 📊 Viewing Results with MLflow

All experiments are logged with MLflow. To view the results:

1.  **Start the MLflow UI:** From the `SDLR` root directory, run:

    ```bash
    mlflow ui
    ```

2.  **Open in Browser:** Open your web browser and go to `http://127.0.0.1:5000`.

There you will find all your experiments, neatly organized. You can compare metrics (like `final_ndcg_at_k`), view parameters, and explore the artifacts of each run (models, parameter files, etc.). Enjoy your analysis\!
