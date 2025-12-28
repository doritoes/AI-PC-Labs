# Set Up Python for our AI Lab

## Download the installer
Download Python installer from Python.org Windows Downloads page
- https://www.python.org/downloads/windows/
- This Lab uses: Python 3.14.2 - Dec. 5, 2025
- Download Windows installer (64-bit) https://www.python.org/ftp/python/3.14.2/python-3.14.2-amd64.exe
- "Look for Python 3.11.9 (this is currently the "sweet spot" for stability with AI libraries like OpenVINO)"

## Installation
1. Run the downloaded file, similar to `python-3.14.2-amd64.exe`
2. **STOP** at the first screen and make the following selections
    - Check the box: Add python.exe to PATH
    - Check the box: Use admin privileges when installing py.exe (recommended for Windows 11 Pro labs)
    - Click Customize installation
        - On the Optional Features screen, ensure pip, tcl/tk and IDLE, and Python test suite are all checked
        - Click **Next**
3. On the **Advanced Options** screen:
    - Check **Install Python 3.14 for all users**
    - Check **Precompile standard library**
    - Click **Install**
5. Disable Path Length Limit
    - At the very end of the installation, you will see a successful setup message
    - STOP and look for the option: "Disable path length limit"
    - Click **Disable path length limit** (AI libraries often have deep folder structures that exceed the standard Windows 260-character limit)
    - Click **CLose**

## Verify
Open a new PowerShell windows (not as Admin this time, just a regular user) and type:
1. `python --version`
    - Expected Output: Python 3.14.2
2. `pip --version`
    - Expected Output: pip 25.3 from ... (python 3.14)

## Create the Virtual Environment
To keep your student labs clean, never install AI libraries globally. Always use a virtual environment.
1. Create project folder
    - `mkdir "$env:USERPROFILE\Captcha-AI"`
2. Navigate to your project folder:
    - `cd "$env:USERPROFILE\Captcha-AI"`
2. Create the environment:
    - `python -m venv nputest_env`
3. Activate it:
    - `.\nputest_env\Scripts\Activate`
    - You should now see (nputest_env) at the start of your PowerShell prompt
