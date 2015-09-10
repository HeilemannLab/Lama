import sys
from cx_Freeze import setup, Executable


package_list = ["os", "pylab", "matplotlib", "numpy", "scipy", "PIL"]
exclude_list = ["tkinter"]
include_list = ["mainwindow.ui", "gpl_v3.txt"]
icon_path = "lama.png"
app_name = "Lama"
ver = "0.9.4"
des = "LAMA!"


# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": package_list, "excludes": exclude_list, "include_files":include_list, "icon": icon_path}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
#if sys.platform == "win32":
    #base = "Win32GUI"

setup(name = app_name,
      version = ver,
      description = des,
      options = {"build_exe": build_exe_options},
      executables = [Executable("main.py", base=base)])