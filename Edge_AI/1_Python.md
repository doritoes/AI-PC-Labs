# Set Up Python for our AI Lab

## Download the installer
- Go to the Python.org Windows Downloads page
- Look for Python 3.11.9 (this is currently the "sweet spot" for stability with AI libraries like OpenVINO)
- Select the Windows installer (64-bit)

## Installation
Once the .exe file is downloaded, run it. Stop at the first screen.

Check the box: Add python.exe to PATH..

Check the box: Use admin privileges when installing py.exe (recommended for Windows 11 Pro labs).

Click Customize installation.

On the Optional Features screen, ensure pip, tcl/tk and IDLE, and Python test suite are all checked. Click Next.

On the Advanced Options screen:

Check Install Python 3.11 for all users.

Check Precompile standard library.

Click Install.

3. Disable Path Length Limit
At the very end of the installation, you will see a successful setup message.

Look for the option: "Disable path length limit".

Click it. (AI libraries often have deep folder structures that exceed the standard Windows 260-character limit).



## Verify
Open PowerShell (not as Admin this time, just a regular user) and type:
python --version
# Expected Output: Python 3.11.9

pip --version
# Expected Output: pip 24.x.x from ... (python 3.11)


## Create the Virtual Environment
To keep your student labs clean, never install AI libraries globally. Always use a virtual environment.

Navigate to your project folder:
cd C:\Path\To\Your\AI-PC-Labs\Edge_AI

Create the environment:
python -m venv nputest_env

Activate it:
.\nputest_env\Scripts\Activate

You should now see (nputest_env) at the start of your PowerShell prompt.

