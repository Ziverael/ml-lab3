# Getting Started with Anaconda Environments
---
This guide provides a brief introduction to using Anaconda environments, including setup in popular IDEs.

## Pre-requisites

Before you begin, ensure you have the following:

1. Anaconda or Miniconda installed on your system. You can download it from the [official Anaconda website](https://www.anaconda.com/products/distribution).

2. An `environment.yml` file in your project directory. This file defines the environment and its dependencies.

3. Basic familiarity with using the command line interface (CLI).

## Steps

1. Open a terminal and navigate to the directory containing your environment file.

2. Create the environment by running the following command:
    ```bash
    conda env create -f environment.yml
    ```

3. Once created, activate the environment using:
    ```bash
    conda activate wust-ml-lab-1
    ```

Note: The environment name (wust-ml-lab-1) may vary depending on your specific configuration.

## Setting Up Conda Environments in IDEs

### PyCharm

1. Open your project in PyCharm.
2. Go to File > Settings (on Windows/Linux) or PyCharm > Preferences (on macOS).
3. Navigate to Project: YourProjectName > Python Interpreter.
4. Click on the gear icon next to the Python Interpreter dropdown and select "Add".
5. Choose "Conda Environment" from the left sidebar.
6. Select "Existing environment" and click on the folder icon to browse for your conda environment.
7. Navigate to your Anaconda installation directory, then to `envs/wust-ml-lab-1/bin/python` (adjust the path as necessary).
8. Click "OK" to confirm and apply the changes.

### Visual Studio Code

1. Open your project folder in VS Code.
2. Press Ctrl+Shift+P (or Cmd+Shift+P on macOS) to open the Command Palette.
3. Type "Python: Select Interpreter" and select it from the list.
4. From the list of available interpreters, choose the one corresponding to your Conda environment (it should include "conda" and "wust-ml-lab-1" in its name).
5. VS Code will automatically activate this environment when you open a new terminal.

To verify the active environment, open a new terminal in VS Code (Terminal > New Terminal) and you should see the environment name in the prompt.

Remember to install the Python extension for VS Code if you haven't already, as it provides enhanced support for working with Python and Conda environments.
