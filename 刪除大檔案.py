import gi
import os
import stat
from gi.repository import Gtk

class FileCleanerApp:
    def __init__(self):
        self.window = Gtk.Window(title="刪除大檔案")
        self.window.set_default_size(400, 300)
        self.window.connect("destroy", Gtk.main_quit)

        # 創建一個VBox來放置小部件
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.window.add(vbox)

        # 創建一個ListBox來顯示檔案
        self.file_list = Gtk.ListBox()
        vbox.pack_start(self.file_list, True, True, 0)

        # 創建一個按鈕來加載大檔案
        load_button = Gtk.Button(label="加載大檔案")
        load_button.connect("clicked", self.load_large_files)
        vbox.pack_start(load_button, False, False, 0)

        # 創建一個按鈕來刪除選中的檔案
        delete_button = Gtk.Button(label="刪除選中檔案")
        delete_button.connect("clicked", self.delete_selected_file)
        vbox.pack_start(delete_button, False, False, 0)

    def load_large_files(self, button):
        # 清空目前的列表
        self.file_list.foreach(lambda widget: self.file_list.remove(widget))

        # 指定要掃描的目錄
        directory = "/path/to/your/directory"  # 請替換為你的目錄路徑
        size_limit = 100 * 1024 * 1024  # 100MB

        for root, dirs, files in os.walk(directory):
            for filename in files:
                filepath = os.path.join(root, filename)
                try:
                    if os.path.getsize(filepath) > size_limit:
                        row = Gtk.ListBoxRow()
                        label = Gtk.Label(filepath)
                        row.add(label)
                        self.file_list.add(row)
                except Exception as e:
                    print(f"無法訪問檔案 {filepath}: {e}")

        self.file_list.show_all()

    def delete_selected_file(self, button):
        selected_row = self.file_list.get_selected_row()
        if selected_row:
            filepath = selected_row.get_child().get_text()
            try:
                os.remove(filepath)
                print(f"已刪除檔案: {filepath}")
                self.file_list.remove(selected_row)
            except Exception as e:
                print(f"無法刪除檔案 {filepath}: {e}")

    def run(self):
        self.window.show_all()
        Gtk.main()

if __name__ == "__main__":
    app = FileCleanerApp()
    app.run()
