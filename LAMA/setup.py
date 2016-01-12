##  setup_win.py
##  LAMA
##
##  Created by Sebastian Malkusch on 10.06.15.
##  <malkusch@chemie.uni-frankfurt.de>
##  Copyright (c) 2015 Single Molecule Biophysics. All rights reserved.
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
from cx_Freeze import setup, Executable

#package_list = []
package_list = ["os", "pylab", "matplotlib", "numpy", "scipy", "PIL"]
exclude_list = ["tkinter", "Tkinter"]
include_list = ["mainwindow.ui", "gpl_v3.txt"]
icon_path = "lama.icns"
app_name = "Lama"
ver = "16.01"
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