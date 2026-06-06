# PNN_S
Probablistic neural network implementation in Python, with a write-up in Typst. The project includes data preprocessing, model training, and evaluation on the Spambase dataset. The write-up covers the methodology, results, and conclusions drawn from the experiments.

The write up in polish is available in the `WriteUp` directory.

# Requirements
- Python 3.14
- Typst 0.14.2
- uv 

# Running the project
## Running the main script
To run the main script, use the following command in your terminal:
```bash
uv run main.py
```
## Building the write-up
To build the write-up, use the following command:
```bash
typst compile --root . WriteUp/main.typ
```
