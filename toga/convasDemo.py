from toga import App, MainWindow, Font, Group, Key, Command, Label, Box, Button, Switch, NumberInput, Slider, Canvas, Selection
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from toga.widgets.canvas import FillRule
from travertino.constants import BLACK, BLUE, BOLD, GREEN, ITALIC, NORMAL, RED, YELLOW
from toga.fonts import CURSIVE, FANTASY, MESSAGE, MONOSPACE, SANS_SERIF, SERIF, SYSTEM

STROKE, FILL, INSTRUCTIONS = "Stroke", "Fill",  "instructions"
TRIANGLE = "triangle"
TRIANGLES = "triangles"
RECTANGLE = "rectangle"
ELLIPSE = "ellipse"
HALF_ELLIPSE = "half ellipse"
ICE_CREAM = "ice cream"
SMILE = "smile"
SEA = "sea"
STAR = "star"
CONTINUOUS = "continuous"
DASH_1_1 = "dash 1-1"
DASH_1_2 = "dash 1-2"
DASH_2_3_1 = "dash 2-3-1"

class CanvasApp(App):
    def startup(self):
        # Set up main window
        self.main_window=MainWindow(size=(750, 500), title=self.name)
        self.canvas=Canvas(style=Pack(flex=1), on_resize=self.refresh_canvas, on_press=self.on_press, on_drag=self.on_drag, on_release=self.on_release, on_alt_press=self.on_alt_press, on_alt_drag=self.on_alt_drag, on_alt_release=self.on_alt_release)
        self.context_selection = Selection(items=[STROKE, FILL], on_select=self.refresh_canvas)
        self.drawing_shape_instructions={INSTRUCTIONS:self.draw_instructions, TRIANGLE:self.draw_triangle, TRIANGLES:self.draw_triangles, RECTANGLE:self.draw_rectangle, ELLIPSE:self.draw_ellipse, HALF_ELLIPSE:self.draw_half_ellipse, ICE_CREAM:self.draw_ice_cream, SMILE:self.draw_smile, SEA:self.draw_sea, STAR:self.draw_star }
        self.shape_selection=Selection(items=list(self.drawing_shape_instructions.keys()), on_select=self.on_shape_change)
        self.color_selection = Selection(items=[BLACK, BLUE, GREEN, RED, YELLOW], on_select=self.refresh_canvas)
        self.fill_rule_selection = Selection(items=[value.name.lower() for value in FillRule], on_select=self.refresh_canvas)
        self.line_width_slider = Slider(range=(1, 10), default=1, on_change=self.refresh_canvas)
        self.dash_patterns = {DASH_1_1:[1, 1], DASH_1_2:[1, 2], DASH_2_3_1:[2, 3, 1], CONTINUOUS:None}
        self.dash_pattern_selection = Selection( items=list(self.dash_patterns.keys()), on_select=self.refresh_canvas)
        self.scale_x_slider = Slider(range=(0, 2), default=1, tick_count=10, on_change=self.refresh_canvas)
        self.scale_y_slider = Slider(range=(0, 2), default=1, tick_count=10, on_change=self.refresh_canvas)
        self.font_selection = Selection(items=[SYSTEM, MESSAGE, SERIF, SANS_SERIF, CURSIVE, FANTASY, MONOSPACE], on_select=self.refresh_canvas)
        self.font_size = NumberInput(min_value=10, max_value=72, default=20, on_change=self.refresh_canvas)
        self.italic_switch = Switch( label="italic", on_toggle=self.refresh_canvas)
        self.bold_switch = Switch( label="bold", on_toggle=self.refresh_canvas)
        label_style = Pack(font_size=10, padding_left=5)
        box = Box(
            style=Pack(direction=COLUMN),
            children=[
                #Box(style=Pack(direction=ROW, padding=5), children=[self.context_selection, self.shape_selection, self.color_selection, self.fill_rule_selection]),
                #Box(style=Pack(direction=ROW, padding=5), children=[ Label("Line Width:", style=label_style), self.line_width_slider, self.dash_pattern_selection ]),
                #Box( style=Pack(direction=ROW, padding=5), children=[ Label("X Scale:", style=label_style), self.scale_x_slider, Label("Y Scale:", style=label_style), self.scale_y_slider, Button(label="Reset transform", on_press=self.reset_transform) ]),
                #Box( style=Pack(direction=ROW, padding=5), children=[ Label("Font Family:", style=label_style), self.font_selection, Label("Font Size:", style=label_style), self.font_size, self.bold_switch, self.italic_switch ]),
                self.canvas
            ]
        )
        self.main_window.content = box
        self.main_window.show()

    @property
    def height(self):
        return self.canvas.layout.content_height
    @property
    def width(self):
        return self.canvas.layout.content_width
    @property
    def translation(self):
        return self._x_translation, self._y_translation
    @translation.setter
    def translation(self, xy_tuple):
        if not xy_tuple:
            self.x_translation = self.y_translation = 0
        else:
            self.x_translation, self.y_translation = xy_tuple
    @property
    def x_translation(self):
        return self._x_translation
    @property
    def y_translation(self):
        return self._y_translation
    @x_translation.setter
    def x_translation(self, x_translation):
        self._x_translation = x_translation

    def reset_transform(self, widget):
        self.translation = None
        self.scale_x_slider.value = 1
        self.scale_y_slider.value = 1
        self.rotation = 0
        self.refresh_canvas(widget)

    def on_shape_change(self, widget):
        self.change_shape()
        self.refresh_canvas(widget)
    def draw_triangle(self, context, factor):
        # calculate offsets to centralize drawing in the bigger axis
        dx = self.x_middle - factor / 2
        dy = self.y_middle - factor / 2
        with context.closed_path(dx + factor / 3, dy + factor / 3) as closer:
            closer.line_to(dx + 2 * factor / 3, dy + 2 * factor / 3)
            closer.line_to(dx + 2 * factor / 3, dy + factor / 3)
    def draw_triangles(self, context, factor):
        # calculate offsets to centralize drawing in the bigger axis
        triangle_size = factor / 5
        gap = factor / 12
        context.move_to(self.x_middle - 2 * triangle_size - gap, self.y_middle - 2 * triangle_size)
        context.line_to(self.x_middle - gap, self.y_middle - 2 * triangle_size)
        context.line_to(self.x_middle - triangle_size - gap, self.y_middle - triangle_size)
        context.line_to(self.x_middle - 2 * triangle_size - gap, self.y_middle - 2 * triangle_size)
        context.move_to(self.x_middle + gap, self.y_middle - 2 * triangle_size)
        context.line_to(self.x_middle + 2 * triangle_size + gap, self.y_middle - 2 * triangle_size)
        context.line_to(self.x_middle + triangle_size + gap, self.y_middle - triangle_size)
        context.line_to(self.x_middle + gap, self.y_middle - 2 * triangle_size)
        context.move_to( self.x_middle - triangle_size, self.y_middle - triangle_size + gap)
        context.line_to( self.x_middle + triangle_size, self.y_middle - triangle_size + gap)
        context.line_to(self.x_middle, self.y_middle + gap)
        context.line_to( self.x_middle - triangle_size, self.y_middle - triangle_size + gap)
    def draw_rectangle(self, context, factor):
        context.rect( self.x_middle - factor / 3, self.y_middle - factor / 6, 2 * factor / 3, factor / 3)
    def draw_ellipse(self, context, factor):
        rx = factor / 3
        ry = factor / 4
        context.ellipse(self.width / 2, self.height / 2, rx, ry)
    def draw_half_ellipse(self, context, factor):
        rx = factor / 3
        ry = factor / 4
        with context.closed_path(self.x_middle + rx, self.y_middle) as closer:
            closer.ellipse(self.x_middle, self.y_middle, rx, ry, 0, 0, math.pi)
    def draw_ice_cream(self, context, factor):
        dx = self.x_middle
        dy = self.y_middle - factor / 6
        with context.closed_path(dx - factor / 5, dy) as closer:
            closer.arc(dx, dy, factor / 5, math.pi, 2 * math.pi)
            closer.line_to(dx, dy + 2 * factor / 5)
    def draw_smile(self, context, factor):
        dx = self.x_middle
        dy = self.y_middle - factor / 5
        with context.closed_path(dx - factor / 5, dy) as closer:
            closer.quadratic_curve_to(dx, dy + 3 * factor / 5, dx + factor / 5, dy)
            closer.quadratic_curve_to(dx, dy + factor / 5, dx - factor / 5, dy)
    def draw_sea(self, context, factor):
        with context.closed_path(self.x_middle-1*factor/5, self.y_middle-1*factor/5) as closer:
            closer.bezier_curve_to(
                self.x_middle - 1 * factor / 10,
                self.y_middle,
                self.x_middle + 1 * factor / 10,
                self.y_middle - 2 * factor / 5,
                self.x_middle + 1 * factor / 5,
                self.y_middle - 1 * factor / 5)
            closer.line_to(self.x_middle+1*factor/5, self.y_middle+1*factor/5)
            closer.line_to(self.x_middle-1*factor/5, self.y_middle+1*factor/5)
    def draw_star(self, context, factor):
        sides = 5
        radius = factor / 5
        rotation_angle = 4 * math.pi / sides
        with context.closed_path(self.x_middle, self.y_middle - radius) as closer:
            for i in range(1, sides):
                closer.line_to(self.x_middle + radius * math.sin(i * rotation_angle),
                               self.y_middle - radius * math.cos(i * rotation_angle))
    def draw_instructions(self, context, factor):
        text = """Instructions:
1. Use the controls to modify the image
2. Press and drag to move the image
3. Double press to center the image at that position
4. Drag using the alternate (e.g. right) button to rotate the image
"""
        font = Font( family=self.font_selection.value, size=self.font_size.value, weight=self.get_weight(), style=self.get_style())
        width, height = self.canvas.measure_text(text, font, tight=True)
        context.write_text(text, self.x_middle - width / 2, self.y_middle, font)

    def refresh_canvas(self, widget):
        self.render_drawing()
    def on_press(self, widget, x, y, clicks):
        self.clicked_point = (x, y)
        self.render_drawing()

    def render_drawing(self):
        self.canvas.clear()
        self.canvas.translate(self.width/2+self.x_translation, self.height/2+self.y_translation)
        self.canvas.rotate(self.rotation)
        self.canvas.scale(self.scale_x_slider.value, self.scale_y_slider.value)
        self.canvas.translate(-self.width/2, -self.height/2)
        with self.get_context(self.canvas) as context:
            self.draw_shape(context)
        self.canvas.reset_transform()

    def on_drag(self, widget, x, y, clicks):
        tx = self.x_translation + x - self.clicked_point[0]
        ty = self.y_translation + y - self.clicked_point[1]
        self.translation = (tx, ty)
        self.clicked_point = (x, y)
        self.render_drawing()
    def on_release(self, widget, x, y, clicks):
        if clicks >= 2:
            self.x_translation = x - self.width / 2
            self.y_translation = y - self.height / 2
        self.clicked_point = None
        self.render_drawing()
    def on_alt_press(self, widget, x, y, clicks):
        self.clicked_point = (x, y)
        self.render_drawing()
    def on_alt_drag(self, widget, x, y, clicks):
        location_vector1 = self.get_location_vector(x, y)
        location_vector2 = self.get_location_vector(*self.clicked_point)
        self.rotation += self.get_rotation_angle(location_vector1, location_vector2)
        self.clicked_point = (x, y)
        self.render_drawing()
    def on_alt_release(self, widget, x, y, clicks):
        self.clicked_point = None
        self.render_drawing()

app=CanvasApp('Canvas', 'org.beeware.widgets.canvas')
app.main_loop()
