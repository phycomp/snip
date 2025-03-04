from toga import App, Box, Image, ImageView, MainWindow
from toga.style.pack import CENTER, COLUMN


class ImageViewApp(App):
    def startup(self):
        self.main_window = MainWindow(title=self.name)
        box = Box()
        box.style.padding = 40
        box.style.update(alignment=CENTER)
        box.style.update(direction=COLUMN)
        # image from local path
        # load brutus.png from the package
        # We set the style width/height parameters for this one
        imgOBJ = Image('resources/brutus.png')
        ivOBJ = ImageView(image=imagOBJ)
        ivOBJ.style.update(height=72)
        ivOBJ.style.update(width=72)
        box.add(ivOBJ)

        # image from remote URL
        # no style parameters - we let Pack determine how to allocate
        # the space
        imgURL = Image('https://beeware.org/project/projects/libraries/toga/toga.png')
        ivURL = ImageView(image=imgURL)
        box.add(ivURL)
        self.main_window.content = box
        self.main_window.show()

if __name__ == '__main__':
    app=ImageViewApp('ImageView', 'org.beeware.widgets.imageview')
    app.main_loop()
