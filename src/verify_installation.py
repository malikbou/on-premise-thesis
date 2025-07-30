#!/usr/bin/env python3
"""
A script to verify and correct the installation of PyMuPDF.

This script forcefully reinstalls the correct PDF library (PyMuPDF) to resolve
potential conflicts with other libraries that might also use the 'fitz' name.
It then attempts a simple import to verify that the environment is correct.

Usage:
------
python -m src.verify_installation
"""
import subprocess
import sys

def run_pip(args):
    """Runs a pip command using the current Python interpreter."""
    subprocess.check_call([sys.executable, "-m", "pip"] + args)

def verify_installation():
    """Forcefully reinstalls PyMuPDF and verifies the installation."""
    print("--- Step 1: Force-reinstalling libraries ---")
    try:
        print("Uninstalling conflicting 'fitz' library...")
        run_pip(["uninstall", "-y", "fitz"])
    except subprocess.CalledProcessError:
        print("Could not find a 'fitz' package to uninstall. That's fine.")

    try:
        print("Uninstalling existing 'PyMuPDF' to ensure a clean slate...")
        run_pip(["uninstall", "-y", "PyMuPDF"])
    except subprocess.CalledProcessError:
        print("Could not find a 'PyMuPDF' package to uninstall. That's fine.")

    print("\nInstalling the correct 'PyMuPDF' library...")
    run_pip(["install", "--force-reinstall", "PyMuPDF"])

    print("\n--- Step 2: Verifying the installation ---")
    try:
        import fitz
        print("Successfully imported 'fitz'.")
        print(f"Fitz documentation: {fitz.__doc__}")
        if 'PyMuPDF' in fitz.__doc__:
            print("\nSUCCESS: The correct 'fitz' (from PyMuPDF) is installed and working.")
        else:
            print("\nERROR: The imported 'fitz' is not from PyMuPDF. Environment issue persists.")
    except ImportError as e:
        print(f"\nERROR: Could not import 'fitz' after installation: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    verify_installation()
