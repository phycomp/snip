from tkinter import Tk, Frame, Label, Button

'''
class SampleApp(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        # the container is where we'll pack the current page
        self.container = Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.current_frame = None
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        # destroy the old page, if there is one
        if self.current_frame: self.current_frame.destroy()
        # create the new page and pack it in the container
        cls = globals()[page_name]
        self.current_frame = cls(self.container)#, self)
        self.current_frame.pack(fill="both", expand=True)
'''

class SampleApp(Tk):
    def __init__(self):
        Tk.__init__(self)
        self._frame = None
        self.switch_frame(StartPage)

    def switch_frame(self, frame_class):
        """Destroys current frame and replaces it with a new one."""
        new_frame = frame_class(self)
        if self._frame: self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

class StartPage(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        Label(self, text="This is the start page").pack(side="top", fill="x", pady=10)
        Button(self, text="Open page one", command=lambda: master.switch_frame(PageOne)).pack()
        Button(self, text="Open page two", command=lambda: master.switch_frame(PageTwo)).pack()

class PageOne(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        Label(self, text="This is page one").pack(side="top", fill="x", pady=10)
        Button(self, text="Return to start page", command=lambda: master.switch_frame(StartPage)).pack()

class PageTwo(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        Label(self, text="This is page two").pack(side="top", fill="x", pady=10)
        Button(self, text="Return to start page", command=lambda: master.switch_frame(StartPage)).pack()

if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
