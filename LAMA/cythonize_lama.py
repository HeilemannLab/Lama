##
##  cythonize_lama.py
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

'''
    Translates slow python code into efficient C-source code and compiles it into a python extension module.
    For further Information read:
   <http://cython.org>
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules=[
             Extension("lama_communicate", ["lama_communicate_source.pyx"],
                       include_dirs=[numpy.get_include()]),
             Extension("lama_app", ["lama_app_source.pyx"],
                       include_dirs=[numpy.get_include()]),
             ]

setup(
      name = 'lama',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      )