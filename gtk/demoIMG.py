#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hildon
import sys
import time
import locale
locale.setlocale(locale.LC_ALL, "")
import pygtk
pygtk.require("2.0")
import gtk
import gtk.keysyms
import gobject
import pango
import os
import stat
import exceptions
import gc

class TSettings(object):
	def __init__(self):
		self._auto_rotate = True

	auto_rotate = property(lambda self: self._auto_rotate)

settings = TSettings()

class TFitMode(int):
	Scale_Width = 0
	Scale_Area = 1

class GImageView(gtk.DrawingArea):
	#view_1 = gtk.Image()
	#view_1.set_from_file(path)

	__gtype_name = "GImageView"

	__gsignals__ = {
		"expose-event": "override",
		"realize": "override",
		"size-allocate": "override",
		"motion-notify-event": "override",
		"button-press-event": "override",
		#"key-press-event": "override",
	}

	def __init__(self):
		gtk.DrawingArea.__init__(self)

		self.set_flags(self.flags() | gtk.CAN_FOCUS)

		self._path = None
		self._description = None
		self._pixbuf = None
		self._gc = None
		self._scroll_offset = (0, 0)
		self._last_scroll_time = 0
		self._hot_spot = (0, 0)
		self._fit_mode = TFitMode(TFitMode.Scale_Width)
		self._rotation = 0
		self._location_layout = None
		self._caption_layout = None
		self._scaled_pixbuf = (None, None) # ((width, height), pixbuf)
		self.add_events(gtk.gdk.KEY_PRESS_MASK|gtk.gdk.KEY_RELEASE_MASK|
		                gtk.gdk.BUTTON1_MOTION_MASK|gtk.gdk.BUTTON_PRESS_MASK|
		                gtk.gdk.BUTTON_RELEASE_MASK)
		# FOCUS_CHANGE_MASK, SCROLL_MASK, POINTER_MOTION_HINT_MASK, BUTTON_MOTION_MASK, BUTTON1_MOTION_MASK
		# BUTTON_PRESS_MASK, BUTTON_RELEASE_MASK

		self._scroll_delayed = None

	def scroll_delayed_cb(self):
		#self.set_scroll_offset(scroll_offset)
		self._scroll_delayed = None
		self.queue_draw()
		return False

	def cancel_scroll_delayed(self):
		if self._scroll_delayed != None:
			gobject.source_remove(self._scroll_delayed)
			self._scroll_delayed = None

	def update_scroll_delayed(self):
		self.cancel_scroll_delayed()
		self._scroll_delayed = gobject.timeout_add(50, self.scroll_delayed_cb)

	def set_path(self, value):
		self._path = value
		self._scaled_pixbuf = (None, None)

		#gobject.GError: Bilddatei »/home/dannym/manga/Good Morning Call/GMC_Vol01_Ch01[m-s]/.« enthält keine Daten
		try:
			self._pixbuf = gtk.gdk.pixbuf_new_from_file(value)
		except gobject.GError:
			self._pixbuf = None

		self._update_location_layout()
		self.queue_resize()

	def _update_location_layout(self):
		path = self._path
		if path != None:
			self._caption_layout = None
			if os.path.isdir(path):
				font_description = pango.FontDescription("Sans 28")
				self._caption_layout = self.create_pango_layout(os.path.basename(os.path.normpath(path)))
				self._caption_layout.set_font_description(font_description)

			if len(path) > 40:
				path_parts = path.split("/")[-2:]
				path = "/".join(["..."] + path_parts)



		else:
			path = "(no path)"

		if self._description != None:
			description = "\n" + self._description
		else:
			description = ""

		self._location_layout = self.create_pango_layout(path + description)

	def do_size_allocate(self, allocation):
		self.queue_draw()
		return gtk.DrawingArea.do_size_allocate(self, allocation)

	def clamp_scroll_offset(self, scroll_offset_):
			#return scroll_offset_

			scroll_offset = scroll_offset_

			pixbuf = self._scaled_pixbuf[1]
			if pixbuf != None:
				new_width = pixbuf.get_width()
				new_height = pixbuf.get_height()

				if new_width <= self.allocation.width:
					scroll_offset = 0, scroll_offset_[1]

				if scroll_offset[1] >= new_height:
					scroll_offset = (0, 0)

			return scroll_offset

	def _scale(self):
		if self._pixbuf != None:
			previous_size = self._scaled_pixbuf[0]

			width = self._pixbuf.get_width()
			height = self._pixbuf.get_height()

			frame_width = self.allocation.width
			frame_height = self.allocation.height

			if width > 0:
				if self._fit_mode == TFitMode.Scale_Area:
					new_width = frame_width
					new_height = height * frame_width / width

					if new_height > frame_height and height > 0:
						new_height = frame_height
						new_width = width * frame_height / height
				elif self._fit_mode == TFitMode.Scale_Width:
					new_width = frame_width
					new_height = height * frame_width / width
				else:
					new_width = width
					new_height = height

				#new_width = width * height / frame_height
				#new_height = frame_height
			else:
				new_width = width
				new_height = height


			if self._scaled_pixbuf[1] != None \
			and previous_size == (new_width, new_height): # already there
				return self._scaled_pixbuf[1]

			# load new

			scaled_pixbuf = self._pixbuf.scale_simple(new_width, new_height, gtk.gdk.INTERP_BILINEAR)
			#if self._rotation != 0:
			#	scaled_pixbuf = scaled_pixbuf.rotate_simple(self._rotation)

			new_width = scaled_pixbuf.get_width()
			new_height = scaled_pixbuf.get_height()

			self._scaled_pixbuf = ((new_width, new_height), scaled_pixbuf)

			# just in case:
			scroll_offset = self.clamp_scroll_offset(self.scroll_offset)
			if scroll_offset != self.scroll_offset:
				self.scroll_offset = scroll_offset

			return scaled_pixbuf
		else:
			self._scaled_pixbuf = (None, None)

			return None

	def do_realize(self):
		gtk.DrawingArea.do_realize(self)
		self._gc = gtk.gdk.GC(self.window)

	def do_button_press_event(self, event):
		self._hot_spot = event.x, event.y

	def do_motion_notify_event(self, event):
		x,y = self._hot_spot
		relative_x = x - event.x
		relative_y = y - event.y

		self._hot_spot = event.x, event.y

		offset = self._scroll_offset

		new_offset = offset[0] + relative_x, offset[1] + relative_y

		self.set_scroll_offset(new_offset)

		#return gtk.DrawingArea.do_motion_notify_event(self, event)

	#def do_key_press_event(self, event):
	#	if event.keyval == gtk.keysyms.space and event.state == 0:
	#		path, remaining_count = find_next_path(path)
	#		self.path
	#		return True
	#
	#	return gtk.DrawingArea.do_key_press_event(self, event)

	def do_expose_event(self, event):
		gc = self._gc
		#print "expose"

		scaled_pixbuf = self._scale()
		if scaled_pixbuf != None:
			new_width, new_height = scaled_pixbuf.get_width(), scaled_pixbuf.get_height()

			x, y = -self._scroll_offset[0], -self._scroll_offset[1]
			x = int(x)
			y = int(y)
			self.window.draw_pixbuf(gc, scaled_pixbuf, 0, 0,
			                        x, y, new_width, new_height, 
			                        gtk.gdk.RGB_DITHER_NONE, 0, 0)


		if self._location_layout != None:
			self.window.draw_layout(gc, 0, 0, self._location_layout)

		if self._caption_layout != None:
			self.window.draw_layout(gc, 0, 80, self._caption_layout)


	def set_description(self, value):
		self._description = value
		self._update_location_layout()
		self.queue_resize()

	def set_scroll_offset(self, value):
		self._scroll_offset = self.clamp_scroll_offset(value)

		now = time.time()
		if now > self._last_scroll_time + 0.1:
			self.scroll_delayed_cb()
			self.cancel_scroll_delayed()
			self._last_scroll_time = now
		else:
			self.update_scroll_delayed()

		#self.queue_draw()

	def set_fit_mode(self, value):
		self._fit_mode = value
		self.queue_draw()

	def set_rotation(self, value):
		self._rotation = value
		self._scaled_pixbuf = (None, None)
		self.queue_resize()

	def go(widget, path):
		global settings

		remainder_count = get_remainder_count(path)
		widget.path = path
		widget.scroll_offset = (0, 0)
		widget.fit_mode = TFitMode.Scale_Width
		#widget.rotation = 
		if settings.auto_rotate == True:
			image_size = widget.image_size
			if image_size[0] > image_size[1]: # wide
				widget.rotation = 0
			else:
				widget.rotation = 90
			pass

		widget.description = "(%d remaining)" % (remainder_count - 1)
		gc.collect()

	def _get_image_size(self):
		if self._pixbuf == None:
			return None, None

		return self._pixbuf.get_width(), self._pixbuf.get_height()

	path = property(lambda self: self._path, set_path)
	description = property(lambda self: self._description, set_description)
	scroll_offset = property(lambda self: self._scroll_offset, set_scroll_offset)
	fit_mode = property(lambda self: self._fit_mode, set_fit_mode)
	rotation = property(lambda self: self._rotation, set_rotation) # degrees
	image_size = property(_get_image_size)

#previous_paths = []

surroundings = (None, [], None) # directory_path, file names, mtime

def update_surroundings(path):
	global surroundings
	directory_path = os.path.dirname(path)
	file_name = os.path.basename(path)
	mtime = os.stat(directory_path)[stat.ST_MTIME]

	if directory_path == surroundings[0] and mtime == surroundings[2]:
		file_names = surroundings[1]

		if file_name not in file_names:
			surroundings = (None, [], None)
	else:
		surroundings = (None, [], None)


	if surroundings[0] != directory_path:
		nodes = [x for x in os.listdir(directory_path) if not x.startswith(".")]
		nodes.sort(cmp = locale.strcoll)
		file_names = ["."] + nodes
		surroundings = (directory_path, file_names, mtime)

	assert(surroundings[0] == directory_path)
	file_names = surroundings[1]
	return file_names

# returns: path, remainder_count_in_folder
def find_next_path(path):
	directory_path = os.path.dirname(path)
	file_name = os.path.basename(path)
	if file_name == "":
		return None

	file_names = update_surroundings(path)

	try:
		i = file_names.index(file_name)
	except exceptions.ValueError:
		# not found
		#return None
		#print "name", file_name, "X"
		raise


	i = i + 1
	if i < len(file_names):
		path = os.path.join(directory_path, file_names[i])

		if os.path.isdir(path):
			return "%s/." % path
		else:
			return path

	if directory_path != "":
		return find_next_path(directory_path)

	return None

def get_surroundings(directory_path):
	global surroundings
	file_names = update_surroundings(path)

	return file_names

def get_remainder_count(path):
	file_names = update_surroundings(path)

	directory_path = os.path.dirname(path)
	file_name = os.path.basename(path)
	try:
		i = file_names.index(file_name)
	except exceptions.ValueError:
		# not found
		return 0

	remainder_node_count = len(file_names) - i

	return remainder_node_count

view_1 = GImageView()
view_1.show()

scrolled_window_1 = gtk.ScrolledWindow()
scrolled_window_1.set_policy(gtk.POLICY_NEVER, gtk.POLICY_NEVER)
scrolled_window_1.add_with_viewport(view_1)
scrolled_window_1.show()

window_1 = hildon.Window()
window_1.add(scrolled_window_1)

window_1.connect("destroy", lambda x: gtk.main_quit())

window_1.show()

previous_paths = []

path = sys.argv[1]
if os.path.isdir(path):
	path = "%s/." % path
else:
	# simulate entries to enable "back" navigation
	directory_path = os.path.dirname(path)

	for name in get_surroundings(directory_path):
		if name == os.path.basename(path): # current
			break

		previous_paths.append(os.path.join(directory_path, name))

#path = "/home/dannym/manga/Good Morning Call/GMC_Vol01_Ch01[m-s]/."
path = find_next_path(path)

window_1.set_focus(view_1)

def key_press_event_cb(widget, event):
	global previous_paths
	if event.state == 0:
		if event.keyval in [gtk.keysyms.space, gtk.keysyms.F7]:
			previous_paths.append(widget.path)
			path = find_next_path(widget.path)
			widget.go(path)
			return True
		elif event.keyval in [gtk.keysyms.x, gtk.keysyms.Return]:
			widget.fit_mode = TFitMode.Scale_Area
			return True
		elif event.keyval in [gtk.keysyms.BackSpace, gtk.keysyms.F8]:
			if len(previous_paths) > 0:
				path = previous_paths[-1]
				previous_paths = previous_paths[:-1]
				widget.go(path)
				return True
		elif event.keyval in [gtk.keysyms.Right, gtk.keysyms.Escape]:
			widget.rotation = (widget.rotation + 90) % 360
			return True
		elif event.keyval == gtk.keysyms.Left:
			widget.rotation = (360 + widget.rotation - 90) % 360
			return True
		else:
			print "key", event.keyval

	return False


remainder_count = get_remainder_count(path)
view_1.connect("key-press-event", key_press_event_cb)
view_1.description = "(%d remaining)" % (remainder_count - 1)
view_1.path = path

#while path != None:
#	print "path", path
#	path = find_next_path(path)

gtk.main()
