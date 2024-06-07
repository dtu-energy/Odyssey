

# Getting Started with Odyssey: Installation Guide
Follow these steps to install Odyssey on your system:

## Step 1: Create a Python Virtual Environment

A virtual environment is a self-contained directory tree that contains a Python installation for a particular version of Python, plus a number of additional packages. To create a virtual environment, open your terminal and run the following command:

```bash
python3 -m venv odyssey_venv
```
This command creates a new directory named `odyssey_venv` in your current directory, which contains a copy of the Python interpreter, the standard library, and various supporting files.

## Step 2: Activate the Virtual Environment

Before you can start installing or using packages in your virtual environment, you’ll need to activate it. Activating a virtual environment will put the virtual environment-specific `python` and `pip` executables into your shell’s PATH.

On Windows, run:

```bash
odyssey_venv\Scripts\activate
```

On macOS, run:

```bash
source odyssey_venv/bin/activate
```

## Step 3: Install Odyssey

With your virtual environment activated, you can now install Odyssey. Use the following command to install Odyssey from its GitLab repository:

```bash
pip3 install git+https://gitlab.com/auto_lab/odyssey
```

Congratulations! You have successfully installed Odyssey. You're now ready to start optimizing with Odyssey.
