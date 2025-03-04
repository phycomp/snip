from tkinter import ttk, StringVar, Tk, Frame, Button, Label
from tkinter.filedialog import askdirectory

LARGE_FONT= ("Verdana", 12)


class Application(Tk):

    def __init__(self, *args, **kwargs):

        Tk.__init__(self, *args, **kwargs)

        Tk.wm_title(self, "Title")

        container = Frame(self, width=768, height=1000)
        container.pack(side="top", fill='both' , expand = 1)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo):

            frame = F(container, self)
            self.frames[F] = frame
            #frame.pack()
            frame.grid(row=0, column=0)
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class StartPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self,parent)
        label = ttk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        button = ttk.Button(self, text="Visit Page 1", command=lambda: controller.show_frame(PageOne))
        button.pack()
        button2 = ttk.Button(self, text="Visit Page 2", command=lambda: controller.show_frame(PageTwo))
        button2.pack()
class PageOne(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = ttk.Label(self, text="Page One!!!", font=LARGE_FONT)
        label.pack(pady=0,padx=100)
        button1 = ttk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button1.pack()
        button2 = ttk.Button(self, text="Page Two", command=lambda: controller.show_frame(PageTwo))
        button2.pack()

class PageTwo(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = ttk.Label(self, text="Page Two!!!", font=LARGE_FONT)
        label.pack(pady=0,padx=0)

        button1 = ttk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page One", command=lambda: controller.show_frame(PageOne))
        button2.pack()

        phone = StringVar()
        home = ttk.Radiobutton(self, text='Home', variable=phone, value='home')
        office = ttk.Radiobutton(self, text='Office', variable=phone, value='office')
        cell = ttk.Radiobutton(self, text='Mobile', variable=phone, value='cell')
        home.pack()
        office.pack()
        cell.pack()

app = Application()
app.mainloop()
