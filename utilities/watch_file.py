
"""
Implementation of a "file watcher" class, using PyQt5's QFileSystemWatcher.

.. warning::
    This file is not documented. Use at your own risk!

# todo: add doc
"""

from PyQt5.QtCore import QFileSystemWatcher


class FileWatcher(object):

    def __init__(self, files=None):
        self._watcher = QFileSystemWatcher()
        self._files = list() # List of file(s) to watch
        self.isWatching = False
        if files is not None:
            self.addFile(files)

    def test(self):
        file = ['/Users/bitzer/hudat.spec']

        self.addFile(file)

    def addFile(self, files):
        # Add a file(s) to the watch list
        # Files needs to be a list, even if a single file

        # Do it while actively watching?

        for _f in files:
            print(_f)
            self._files.append(_f)

    def removeFile(self):
        # Remove a file
        pass

    def replaceFile(self, file):
        # In the (usual) case of a watching a single file, replace it
        if len(self._files) != 1:
            print('Only allowed if one file is currently watched')

        self._files[0] = file

    def startWatch(self):
        # Start watching the files

        self._watcher.addPaths(self._files)

        self._watcher.fileChanged.connect(self.onChange)
        self.isWatching = True

    def stopWatch(self):
        # Stop Watching the folder

        self._watcher.removePaths(self._files)

        self._watcher.fileChanged.disconnect(self.onChange)

        self.isWatching = False

    def onChange(self, file):
        # When a file changes, do something
        # Which file changed?
        print('changed ' + file)
