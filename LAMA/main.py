#!/usr/bin/env python
##
##  main.py
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


# import PyQt4 QtCore and QtGui modules
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from mainwindow import MainWindow
ver = 1510
print ('\nSingle Molecule Biophysics\nUniversity of Frankfurt\nloading the Lama...\n')
if __name__ == '__main__':

    # create application
    app = QApplication( sys.argv )
    app.setApplicationName( 'lama' )

    # create widget
    w = MainWindow()
    w.setWindowTitle( 'lama' )
    w.show()
    print ('The Lama (v.%i) is saddled and ready to pronk through your SMLM coordinate lists.\n'%(ver))

    # connection
    QObject.connect( app, SIGNAL( 'lastWindowClosed()' ), app, SLOT( 'quit()' ) )

    # execute application
    sys.exit( app.exec_() )
