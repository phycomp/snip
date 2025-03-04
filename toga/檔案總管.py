import os
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from toga import Box, SplitContainer, Label, Tree, MainWindow, App, Icon, ScrollContainer, WebView
from toga.constants import COLUMN, ROW
from toga.sources import Source
from toga.style import Pack
#from toga.import HORIZONTAL
from toga import SplitContainer

# This is a slightly less toy example of a tree view to display
# items in your home directory. To avoid loading everything
# in advance, the content of a folder is loaded dynamically,
# on access. This typically happens when the tree view asks for
# the number of children to decide if it should display an
# expandable row.
#
# In practice, one can use a similar concept of on-access
# loading to fetch data from an API or online resource but
# perform the loading asynchronously.


class LoadingFailedNode:
    """A node to represent failed loading of children"""

    def __init__(self, parent):
        self._parent = parent
        self.children = []
        self.name = 'loading failed'
        self.date_modified = ''

    # Methods required for the data source interface
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise StopIteration

    @staticmethod
    def can_have_children():
        return False


class Node:
    """A node which loads its children on-demand."""

    def __init__(self, path, parent):
        super().__init__()
        self._parent = parent
        self._children = []

        self.path = path
        self._mtime = self.path.stat().st_mtime
        if self.path.is_file():
            self._icon = Icon('resources/file')
        else:
            self._icon = Icon('resources/folder')
        self._did_start_loading = False

    def __repr__(self):
        return "<Node {0}>".format(self.path)

    # Methods required for the data source interface
    def __len__(self):
        return len(self.children)

    def __getitem__(self, index):
        return self.children[index]

    def can_have_children(self):
        # this will trigger loading of children, if not yet done
        return len(self.children) > 0

    # Property that returns the first column value as (icon, label)
    @property
    def name(self):
        return self._icon, self.path.name

    # Property that returns modified date as str
    @property
    def date_modified(self):
        return datetime.fromtimestamp(self._mtime).strftime('%d %b %Y at %H:%M')

    # on-demand loading of children
    @property
    def children(self):
        if not self._did_start_loading:
            self._did_start_loading = True
            self.load_children()
        return self._children

    def load_children(self):
        try:
            sub_paths = [p for p in self.path.iterdir()]
            self._children = [Node(p, self) for p in sub_paths]
        except NotADirectoryError:
            self._children = []
        except OSError:
            self._children = [LoadingFailedNode(self)]

    def index(self, node):
        if node._parent:
            return node._parent._children.index(node)
        else:
            return self.children.index(node)


class FileSystemSource(Node, Source):
    def __init__(self, path):
        super().__init__(path, parent=None)


class treeSourceApp(App):
    def slctnHndlr(self, widget, node):
        # A node is a dictionary of the last item that was clicked in the tree.
        # node['node'].path would get you the file path to only that one item.
        # self.label.text = f'Selected {node["node"].path}'

        # If you iterate over widget.selection, you can get the names and the
        # paths of everything selected (if multiple_select is enabled.)
        # filepaths = [node.path for node in widget.selection]
        if isinstance(widget.selection, list):
            files = len(widget.selection)
        else:
            files = 0 if widget.selection is None else 1
        if files == 0:
            self.label.text = 'A view of the current directory!'
        elif files == 1:
            self.label.text = 'You selected {0} item'.format(files)
        else:
            self.label.text = 'You selected {0} items'.format(files)

    def dblClckHndlr(self, widget, node):
        # open the file or folder in the platform's default app
        self.label.text = 'You started {0}'.format(node.path)
        if platform.system() == 'Darwin':
            subprocess.call(('open', node.path))
        elif platform.system() == 'Windows':
            os.startfile(node.path)
        else:
            subprocess.call(('xdg-open', node.path))

    def startup(self):
        self.main_window = MainWindow(title=self.name)
        pwd=Path.cwd()
        self.fs_source = FileSystemSource(pwd)
        self.tree = Tree(headings=['Name', 'Date Modified'], data=self.fs_source, style=Pack(flex=1), multiple_select=True, on_select=self.slctnHndlr, on_double_click=self.dblClckHndlr)
        lblPck=Pack(padding=10)
        self.label = Label('A view of the current directory!', style=lblPck)
        pckLEFT=Pack(flex=1, direction=COLUMN)  #ROW, HORIZONTAL
        self.lftCntnr=Box(children=[self.label, self.tree], style=pckLEFT)
        self.lftCntnr.style.width='200'
        #dmmyBx=Box(style=pckClmn)
        #wbVwr=WebView()
        self.rghtCntnr=Box(style=pckLEFT)    #ScrollContainer(horizontal=False)#, content=wbVwr) #SplitContainer.HORIZONTAL)
        #self.rghtCntnr=wbVwr
        self.rghtCntnr.style.background='maroon'
        #self.scrllCntnr.content=dmmyBx
        #self.rghtCntnr=Box(children=[self.label, self.tree], style=pckClmn)
        self.spltCntnr=SplitContainer(direction=SplitContainer.VERTICAL) #
        self.spltCntnr.content=[self.lftCntnr, self.rghtCntnr]
        # Add the content on the main window
        self.main_window.content=self.spltCntnr

        # Show the main window
        self.main_window.show()


#def main():return ExampleTreeSourceApp('Tree Source', 'org.beeware.widgets.tree_source')


if __name__ == '__main__':
    app = treeSourceApp('Tree Source', 'org.beeware.widgets.tree_source')
    app.main_loop()
