def RenderVTKVolume(image, volprops):
    volmap = vtk.vtkVolumeRayCastMapper()
    volmap.SetVolumeRayCastFunction(vtk.vtkVolumeRayCastCompositeFunction())
    volmap.SetInputConnection(image.GetOutputPort())

    vol = vtk.vtkVolume()
    vol.SetMapper(volmap)
    vol.SetProperty(volprops)

    #Standard VTK stuff
    ren = vtk.vtkRenderer()
    ren.AddVolume(vol)
    ren.SetBackground((1, 1, 1))

    renwin = vtk.vtkRenderWindow()
    renwin.AddRenderer(ren)

    istyle = vtk.vtkInteractorStyleSwitch()
    istyle.SetCurrentStyleToTrackballCamera()

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renwin)
    iren.SetInteractorStyle(istyle)

    renwin.Render()
    iren.Start()

 
#File: App.py Project: kvkenyon/projects
def volumeRender(reader,ren,renWin):
	#Create transfer mapping scalar value to opacity
	opacityTransferFunction = vtk.vtkPiecewiseFunction()
	opacityTransferFunction.AddPoint(1, 0.0)
	opacityTransferFunction.AddPoint(100, 0.1)
	opacityTransferFunction.AddPoint(255,1.0)

	colorTransferFunction = vtk.vtkColorTransferFunction()
	colorTransferFunction.AddRGBPoint(0.0,0.0,0.0,0.0)	
	colorTransferFunction.AddRGBPoint(64.0,1.0,0.0,0.0)	
	colorTransferFunction.AddRGBPoint(128.0,0.0,0.0,1.0)	
	colorTransferFunction.AddRGBPoint(192.0,0.0,1.0,0.0)	
	colorTransferFunction.AddRGBPoint(255.0,0.0,0.2,0.0)	

	volumeProperty = vtk.vtkVolumeProperty()
	volumeProperty.SetColor(colorTransferFunction)
	volumeProperty.SetScalarOpacity(opacityTransferFunction)
	volumeProperty.ShadeOn()
	volumeProperty.SetInterpolationTypeToLinear()

	compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
	volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
	volumeMapper.SetInputConnection(reader.GetOutputPort())

	volume = vtk.vtkVolume()
	volume.SetMapper(volumeMapper)
	volume.SetProperty(volumeProperty)

	ren.RemoveAllViewProps()

	ren.AddVolume(volume)
	ren.SetBackground(1,1,1)

	renWin.Render()
 
#File: volume_render.py Project: daniel-perry/visualization
def main(argv):
  if len(argv) < 2:
    print "usage:",argv[0]," data.vtk"
    exit(1)
  data_fn = argv[1]
  reader = vtk.vtkStructuredPointsReader()
  reader.Set#FileName(data_fn)
  reader.Update()
  data = reader.GetOutput()
  updateColorOpacity()
  # composite function (using ray tracing)
  compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
  volumeMapper = vtk.vtkVolumeRayCastMapper()
  volumeMapper.SetVolumeRayCastFunction(compositeFunction)
  volumeMapper.SetInput(data)
  # make the volume
  #volume = vtk.vtkVolume()
  global volume
  volume.SetMapper(volumeMapper)
  volume.SetProperty(volumeProperty)
  # renderer
  renderer = vtk.vtkRenderer()
  renderWin = vtk.vtkRenderWindow()
  renderWin.AddRenderer(renderer)
  renderInteractor = vtk.vtkRenderWindowInteractor()
  renderInteractor.SetRenderWindow(renderWin)
  renderInteractor.AddObserver( vtk.vtkCommand.KeyPressEvent, keyPressed )
  renderer.AddVolume(volume)
  renderer.SetBackground(0,0,0)
  renderWin.SetSize(400, 400)
  renderInteractor.Initialize()
  renderWin.Render()
  renderInteractor.Start()
 
#File: Draw.py Project: EthanGlasserman/Brainbow
def axonComparison(axons, N):
    axonsToRender = []
    for i in range(N):
        axon = axons.pop(random.randrange(len(axons)))
        axonsToRender.append(axon)
    bins = main.BINS
    data_matrix = numpy.zeros([500, 500, 500], dtype=numpy.uint16)
    dataImporter = vtk.vtkImageImport()
    data_string = data_matrix.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetDataExtent(0, 500, 0, 500, 0, 500)
    dataImporter.SetWholeExtent(0, 500, 0, 500, 0, 500)
    volumeProperty = vtk.vtkVolumeProperty()
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)
    renderer.SetBackground(1, 1, 1)
    renderWin.SetSize(400, 400)
    for axon in axonsToRender:
        renderer = Utils.renderSingleAxon(axon, renderer, [random.random(), random.random(), random.random()])
    renderWin.AddObserver("AbortCheckEvent", exitCheck)
    renderInteractor.Initialize()
    renderWin.Render()
    renderInteractor.Start()
 
#File: VolumeRender.py Project: fvpolpeta/devide
 def _setup_for_raycast(self):
     self._volume_raycast_function = vtk.vtkVolumeRayCastCompositeFunction()
     self._volume_mapper = vtk.vtkVolumeRayCastMapper()
     self._volume_mapper.SetVolumeRayCastFunction(self._volume_raycast_function)
     
     module_utils.setup_vtk_object_progress(self, self._volume_mapper, 'Preparing render.')
 
#File: volume_render.py Project: daniel-perry/rt
def main(argv):
  if len(argv) < 2:
    print "usage:",argv[0]," data.nrrd data.cmap"
    exit(1)
  data_fn = argv[1]
  cmap_fn = argv[2]
  reader = vtk.vtkPNrrdReader()
  reader.Set#FileName(data_fn)
  reader.Update()
  data = reader.GetOutput()
  # opacity function
  opacityFunction = vtk.vtkPiecewiseFunction()
  # color function
  colorFunction = vtk.vtkColorTransferFunction()
  cmap = open(cmap_fn, 'r')
  for line in cmap.readlines():
    parts = line.split()
    value = float(parts[0])
    r = float(parts[1])
    g = float(parts[2])
    b = float(parts[3])
    a = float(parts[4])
    opacityFunction.AddPoint(value, a)
    colorFunction.AddRGBPoint(value, r, g, b)
  # volume setup:
  #volumeProperty = vtk.vtkVolumeProperty()
  global volumeProperty
  volumeProperty.SetColor(colorFunction)
  volumeProperty.SetScalarOpacity(opacityFunction)
  # composite function (using ray tracing)
  compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
  volumeMapper = vtk.vtkVolumeRayCastMapper()
  volumeMapper.SetVolumeRayCastFunction(compositeFunction)
  volumeMapper.SetInput(data)
  # make the volume
  #volume = vtk.vtkVolume()
  global volume
  volume.SetMapper(volumeMapper)
  volume.SetProperty(volumeProperty)
  # renderer
  renderer = vtk.vtkRenderer()
  renderWin = vtk.vtkRenderWindow()
  renderWin.AddRenderer(renderer)
  renderInteractor = vtk.vtkRenderWindowInteractor()
  renderInteractor.SetRenderWindow(renderWin)
  renderInteractor.AddObserver( vtk.vtkCommand.KeyPressEvent, keyPressed )
  renderer.AddVolume(volume)
  renderer.SetBackground(0,0,0)
  renderWin.SetSize(400, 400)
  renderInteractor.Initialize()
  renderWin.Render()
  renderInteractor.Start()
 
#File: volume.py Project: 151706061/invesalius
 def SetTypeRaycasting(self):
     if self.volume_mapper.IsA("vtkFixedPointVolumeRayCastMapper"):
         if self.config.get('MIP', False): self.volume_mapper.SetBlendModeToMaximumIntensity()
         else: self.volume_mapper.SetBlendModeToComposite()
     else:
         if self.config.get('MIP', False): raycasting_function = vtk.vtkVolumeRayCastMIPFunction()
         else:
             raycasting_function = vtk.vtkVolumeRayCastCompositeFunction()
             raycasting_function.SetCompositeMethodToInterpolateFirst()
         self.volume_mapper.SetVolumeRayCastFunction(raycasting_function)
 
#File: volumerendering.py Project: 151706061/Medical-Image-Analysis-IPython-Tutorials
def volumeRender(img, tf=[],spacing=[1.0,1.0,1.0]):
    importer = numpy2VTK(img,spacing)

    # Transfer Functions
    opacity_tf = vtk.vtkPiecewiseFunction()
    color_tf = vtk.vtkColorTransferFunction()

    if not len(tf):
        tf.append([img.min(),0,0,0,0])
        tf.append([img.max(),1,1,1,1])

    for p in tf:
        color_tf.AddRGBPoint(p[0], p[1], p[2], p[3])
        opacity_tf.AddPoint(p[0], p[4])

    # working on the GPU
    # volMapper = vtk.vtkGPUVolumeRayCastMapper()
    # volMapper.SetInputConnection(importer.GetOutputPort())

    # # The property describes how the data will look
    # volProperty =  vtk.vtkVolumeProperty()
    # volProperty.SetColor(color_tf)
    # volProperty.SetScalarOpacity(opacity_tf)
    # volProperty.ShadeOn()
    # volProperty.SetInterpolationTypeToLinear()

    # working on the CPU
    volMapper = vtk.vtkVolumeRayCastMapper()
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    compositeFunction.SetCompositeMethodToInterpolateFirst()
    volMapper.SetVolumeRayCastFunction(compositeFunction)
    volMapper.SetInputConnection(importer.GetOutputPort())

    # The property describes how the data will look
    volProperty =  vtk.vtkVolumeProperty()
    volProperty.SetColor(color_tf)
    volProperty.SetScalarOpacity(opacity_tf)
    volProperty.ShadeOn()
    volProperty.SetInterpolationTypeToLinear()
    
    # Do the lines below speed things up?
    # pix_diag = 5.0
    # volMapper.SetSampleDistance(pix_diag / 5.0)    
    # volProperty.SetScalarOpacityUnitDistance(pix_diag) 
    

    vol = vtk.vtkVolume()
    vol.SetMapper(volMapper)
    vol.SetProperty(volProperty)
    
    return [vol]
 
#File: volume_rendering.py Project: arun04ceg/3D-spatial-data
def volumeProperty(reader, opacityTransferFunction, colorTransferFunction):
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.ShadeOn()
    volumeProperty.SetSpecular(0.3)
    volumeProperty.SetInterpolationTypeToLinear()

    rayCastFunction = vtk.vtkVolumeRayCastCompositeFunction()

    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetSampleDistance(1)
    volumeMapper.SetInput(reader.GetOutput())
    volumeMapper.SetVolumeRayCastFunction(rayCastFunction)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    volume.RotateX(-90)
    return volume
 
#File: VesselDisplayWindow.py Project: CoderXv/vas
    def render_volume_data(self, vtk_img_data):
        # Create transfer mapping scalar value to opacity
        opacity_transfer_function = vtk.vtkPiecewiseFunction()
        opacity_transfer_function.AddPoint(0, 0.0)
        opacity_transfer_function.AddPoint(50, 0.0)
        opacity_transfer_function.AddPoint(100, 0.8)
        opacity_transfer_function.AddPoint(1200, 0.8)

        # Create transfer mapping scalar value to color
        color_transfer_function = vtk.vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(0, 0.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(50, 0.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(100, 1.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(1200, 1.0, 0.0, 0.0)

        # The property describes how the data will look
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer_function)
        volume_property.SetScalarOpacity(opacity_transfer_function)
        volume_property.ShadeOff()
        volume_property.SetInterpolationTypeToLinear()

        # The mapper / ray cast function know how to render the data
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        volume_mapper = vtk.vtkVolumeRayCastMapper()
        volume_mapper.SetVolumeRayCastFunction(compositeFunction)
        if vtk.VTK_MAJOR_VERSION <= 5:
            volume_mapper.SetInput(vtk_img_data)
        else:
            volume_mapper.SetInputData(vtk_img_data)
        volume_mapper.SetBlendModeToMaximumIntensity()

        # The volume holds the mapper and the property and
        # can be used to position/orient the volume
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        self.ren.AddVolume(volume)
        self.ren.ResetCamera()
        self.iren.Initialize()
 
#File: oldAssignmnet1.py Project: kvkenyon/projects
	def volumeRender(self):
		#Create transfer mapping scalar value to opacity
		opacityTransferFunction = vtk.vtkPiecewiseFunction()
		opacityTransferFunction.AddPoint(1, 0.0)
		opacityTransferFunction.AddPoint(100, 0.1)
		opacityTransferFunction.AddPoint(255,1.0)

		colorTransferFunction = vtk.vtkColorTransferFunction()
		colorTransferFunction.AddRGBPoint(0.0,0.0,0.0,0.0)	
		colorTransferFunction.AddRGBPoint(64.0,1.0,0.0,0.0)	
		colorTransferFunction.AddRGBPoint(128.0,0.0,0.0,1.0)	
		colorTransferFunction.AddRGBPoint(192.0,0.0,1.0,0.0)	
		colorTransferFunction.AddRGBPoint(255.0,0.0,0.2,0.0)	

		volumeProperty = vtk.vtkVolumeProperty()
		volumeProperty.SetColor(colorTransferFunction)
		volumeProperty.SetScalarOpacity(opacityTransferFunction)
		volumeProperty.ShadeOn()
		volumeProperty.SetInterpolationTypeToLinear()

		compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
		volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
		volumeMapper.SetInputConnection(self.reader.GetOutputPort())

		volume = vtk.vtkVolume()
		volume.SetMapper(volumeMapper)
		volume.SetProperty(volumeProperty)

		ren = vtk.vtkRenderer()
		renWin = vtk.vtkRenderWindow()
		renWin.AddRenderer(ren)
		iren = vtk.vtkRenderWindowInteractor()
		iren.SetRenderWindow(renWin)
		ren.AddVolume(volume)
		ren.SetBackground(1,1,1)
		renWin.SetSize(600,600)
		renWin.Render()

		iren.Initialize()
		renWin.Render()
		iren.Start() 
 
#File: vtkOptics.py Project: HerrMuellerluedenscheid/derec
    def vtkCube(self, data_matrix=None):

        # We begin by creating the data we want to render.
        # For this tutorial, we create a 3D-image containing three overlaping cubes.
        # This data can of course easily be replaced by data from a medical CT-scan or anything else three dimensional.
        # The only limit is that the data must be reduced to unsigned 8 bit or 16 bit integers.
        #data_matrix = zeros([75, 75, 75], dtype=uint8)
        #data_matrix[0:35, 0:35, 0:35] = 50
        #data_matrix[25:55, 25:55, 25:55] = 100
        #data_matrix[45:74, 45:74, 45:74] = 150

        # For VTK to be able to use the data, it must be stored as a VTK-image. This can be done by the vtkImageImport-class which
        # imports raw data and stores it.
        dataImporter = vtk.vtkImageImport()
        # The preaviusly created array is converted to a string of chars and imported.
        data_string = data_matrix.tostring()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        # The type of the newly imported data is set to unsigned char (uint8)
        dataImporter.SetDataScalarTypeToUnsignedChar()
        # Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
        # must be told this is the case.
        dataImporter.SetNumberOfScalarComponents(1)
        # The following two functions describe how the data is stored and the dimensions of the array it is stored in. For this
        # simple case, all axes are of length 75 and begins with the first element. For other data, this is probably not the case.
        # I have to admit however, that I honestly dont know the difference between SetDataExtent() and SetWholeExtent() although
        # VTK complains if not both are used.
        dataImporter.SetDataExtent(0, 9, 0, 9, 0, 9)
        dataImporter.SetWholeExtent(0, 9, 0, 9, 0, 9)
        #dataImporter.SetDataExtent(0, 74, 0, 74, 0, 74)
        #dataImporter.SetWholeExtent(0, 74, 0, 74, 0, 74)

        # The following class is used to store transparencyv-values for later retrival. In our case, we want the value 0 to be
        # completly opaque whereas the three different cubes are given different transperancy-values to show how it works.
        alphaChannelFunc = vtk.vtkPiecewiseFunction()
        alphaChannelFunc.AddPoint(0, 0.6)
        alphaChannelFunc.AddPoint(33, 0.2)
        alphaChannelFunc.AddPoint(66, 0.1)
        alphaChannelFunc.AddPoint(100, 0.01)

        # Gradient opacity
        # other way: misfit 0 is anti opacity
        volumeGradientOpacity = vtk.vtkPiecewiseFunction()
        volumeGradientOpacity.AddPoint(70,   1.0)
        volumeGradientOpacity.AddPoint(50,  0.5)
        volumeGradientOpacity.AddPoint(20, 0.0)

        # This class stores color data and can create color tables from a few color points. For this demo, we want the three cubes
        # to be of the colors red green and blue.
        colorFunc = vtk.vtkColorTransferFunction()
        colorFunc.AddRGBPoint(00, 1.0, 0.0, 0.0)
        colorFunc.AddRGBPoint(30, 0.0, 1.0, 0.0)
        colorFunc.AddRGBPoint(60, 0.0, 0.0, 1.0)

        # The preavius two classes stored properties. Because we want to apply these properties to the volume we want to render,
        # we have to store them in a class that stores volume prpoperties.
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorFunc)
        volumeProperty.SetScalarOpacity(alphaChannelFunc)
        volumeProperty.SetGradientOpacity(volumeGradientOpacity)
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.ShadeOff()
        volumeProperty.SetAmbient(0.1)
        volumeProperty.SetDiffuse(0.6)
        volumeProperty.SetSpecular(0.2)

        # This class describes how the volume is rendered (through ray tracing).
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        # We can finally create our volume. We also have to specify the data for it, as well as how the data will be rendered.
        volumeMapper = vtk.vtkVolumeRayCastMapper()
        volumeMapper.SetVolumeRayCastFunction(compositeFunction)
        volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

        # The class vtkVolume is used to pair the preaviusly declared volume as well as the properties to be used when rendering that volume.
        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)

        # Text am Nullpunkt
        atext = vtk.vtkVectorText()
        atext.SetText("(0,0,0)")
        textMapper = vtk.vtkPolyDataMapper()
        textMapper.SetInputConnection(atext.GetOutputPort())
        textActor = vtk.vtkFollower()
        textActor.SetMapper(textMapper)
        textActor.SetScale(10, 10, 10)
        textActor.AddPosition(0, -0.1, 78)

        # Cube to give some orientation 
        # (from http://www.vtk.org/Wiki/VTK/Examples/Python/Widgets/OrientationMarkerWidget)

        axesActor = vtk.vtkAnnotatedCubeActor();
        axesActor.SetXPlusFaceText('N')
        axesActor.SetXMinusFaceText('S')
        axesActor.SetYMinusFaceText('W')
        axesActor.SetYPlusFaceText('E')
        axesActor.SetZMinusFaceText('D')
        axesActor.SetZPlusFaceText('U')
        axesActor.GetTextEdgesProperty().SetColor(1,1,0)
        axesActor.GetTextEdgesProperty().SetLineWidth(2)
        axesActor.GetCubeProperty().SetColor(0,0,1)

        # With almost everything else ready, its time to initialize the renderer and window, as well as creating a method for exiting the application
        renderer = vtk.vtkRenderer()
        renderWin = vtk.vtkRenderWindow()
        renderWin.AddRenderer(renderer)
        renderInteractor = vtk.vtkRenderWindowInteractor()
        renderInteractor.SetRenderWindow(renderWin)

        axes = vtk.vtkOrientationMarkerWidget()
        axes.SetOrientationMarker(axesActor)
        axes.SetInteractor(renderInteractor)
        axes.EnabledOn()
        axes.InteractiveOn()
        renderer.ResetCamera()

        # We add the volume to the renderer ...
        renderer.AddVolume(volume)
        # ... set background color to white ...
        renderer.SetBackground(0.7,0.7,0.7)
        # ... and set window size.
        renderWin.SetSize(400, 400)

        # Fuege Text am Nullpunkt hinzu:
        renderer.AddActor(textActor)
        
        # A simple function to be called when the user decides to quit the application.
        def exitCheck(obj, event):
            if obj.GetEventPending() != 0:
                obj.SetAbortRender(1)

        # Tell the application to use the function as an exit check.
        renderWin.AddObserver("AbortCheckEvent", exitCheck)

        renderInteractor.Initialize()
        # Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.
        renderWin.Render()
        renderInteractor.Start()
 
#File: boundarypruningvtk.py Project: mattbierbaum/cuda-plasticity
def plot(N, field, prefix="vtkrender", animate=True, write=True):
    field = numpy.tanh(field / field.mean() / 8)
    maxfield = field.max()
    if maxfield > 5:
        maxfield = 5
    field = field * 255 / maxfield
    field = (field > 255) * 255 + (field <= 255) * field
    field = field.astype("uint8")

    timeSeries = True
    mip = False

    minopacity = 0.001
    maxopacity = 0.1

    dataImporter = vtk.vtkImageImport()
    dataImporter.SetDataScalarTypeToUnsignedChar()
    data_string = rho.tostring()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))

    dataImporter.SetDataExtent(0, N - 1, 0, N - 1, 0, N - 1)
    dataImporter.SetWholeExtent(0, N - 1, 0, N - 1, 0, N - 1)

    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, maxopacity)
    alphaChannelFunc.AddPoint(255, minopacity)

    volumeProperty = vtk.vtkVolumeProperty()
    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(255, 1.0, 1.0, 1.0)

    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)

    volumeMapper = vtk.vtkVolumeRayCastMapper()
    if mip:
        mipFunction = vtk.vtkVolumeRayCastMIPFunction()
        mipFunction.SetMaximizeMethodToOpacity()
        volumeMapper.SetVolumeRayCastFunction(mipFunction)
    else:
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)

    renderer.AddVolume(volume)
    renderer.SetBackground(1.0, 1.0, 1.0)
    # renderer.SetBackground(0.6,0.6,0.6)
    renderWin.SetSize(400, 400)

    renderWin.AddObserver("AbortCheckEvent", exitCheck)

    if not animate:
        renderInteractor.Initialize()
    renderWin.Render()

    if not animate:
        renderInteractor.Start()

    if animate:
        writer = vtk.vtkPNGWriter()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(renderWin)
        writer.SetInputConnection(w2i.GetOutputPort())

        renderWin.Render()
        ac = renderer.GetActiveCamera()
        ac.Elevation(20)
        step = 1
        current = 0
        for i in range(0, 360, step):
            ac.Azimuth(step)
            # ac.Elevation(1*((-1)**(int(i/90))))
            renderer.ResetCameraClippingRange()
            renderWin.Render()
            w2i.Modified()

            if write:
                writer.Set#FileName("%s%04d.gif" % (prefix, current))
                writer.Write()
            current += 1
        writer.End()
 
#File: fos.py Project: arokem/Fos
def volume(vol,voxsz=(1.0,1.0,1.0),affine=None,center_origin=1,info=1,maptype=0,trilinear=1,iso=0,iso_thr=100,opacitymap=None,colormap=None):    
    ''' Create a volume and return a volumetric actor using volumetric rendering. 
    This function has many different interesting capabilities. The maptype, opacitymap and colormap are the most crucial parameters here.
    
    Parameters:
    ----------------
    vol : array, shape (N, M, K), dtype uint8
         an array representing the volumetric dataset that we want to visualize using volumetric rendering            
        
    voxsz : sequence of 3 floats
            default (1., 1., 1.)
            
    affine : array, shape (4,4), default None
            as given by volumeimages             
            
    center_origin : int {0,1}, default 1
             it considers that the center of the volume is the 
            point (-vol.shape[0]/2.0+0.5,-vol.shape[1]/2.0+0.5,-vol.shape[2]/2.0+0.5)
            
    info : int {0,1}, default 1
            if 1 it prints out some info about the volume, the method and the dataset.
            
    trilinear: int {0,1}, default 1
            Use trilinear interpolation, default 1, gives smoother rendering. If you want faster interpolation use 0 (Nearest).
            
    maptype : int {0,1}, default 0,        
            The maptype is a very important parameter which affects the raycasting algorithm in use for the rendering. 
            The options are:
            If 0 then vtkVolumeTextureMapper2D is used.
            If 1 then vtkVolumeRayCastFunction is used.
            
    iso : int {0,1} default 0,
            If iso is 1 and maptype is 1 then  we use vtkVolumeRayCastIsosurfaceFunction which generates an isosurface at 
            the predefined iso_thr value. If iso is 0 and maptype is 1 vtkVolumeRayCastCompositeFunction is used.
            
    iso_thr : int, default 100,
            if iso is 1 then then this threshold in the volume defines the value which will be used to create the isosurface.
            
    opacitymap : array, shape (N,2), default None.
            The opacity map assigns a transparency coefficient to every point in the volume.
            The default value uses the histogram of the volume to calculate the opacitymap.
    colormap : array, shape (N,4), default None.
            The color map assigns a color value to every point in the volume.
            When None from the histogram it uses a red-blue colormap.
                
    Returns:
    ----------
    vtkVolume    
    
    Notes:
    --------
    What is the difference between TextureMapper2D and RayCastFunction? 
    Coming soon... See VTK user's guide [book] & The Visualization Toolkit [book] and VTK's online documentation & online docs.
    
    What is the difference between RayCastIsosurfaceFunction and RayCastCompositeFunction?
    Coming soon... See VTK user's guide [book] & The Visualization Toolkit [book] and VTK's online documentation & online docs.
    
    What about trilinear interpolation?
    Coming soon... well when time permits really ... :-)
    
    Examples:
    ------------
    First example random points    
    
    >>> from dipy.viz import fos
    >>> import numpy as np
    >>> vol=100*np.random.rand(100,100,100)
    >>> vol=vol.astype('uint8')
    >>> print vol.min(), vol.max()
    >>> r = fos.ren()
    >>> v = fos.volume(vol)
    >>> fos.add(r,v)
    >>> fos.show(r)
    
    Second example with a more complicated function
        
    >>> from dipy.viz import fos
    >>> import numpy as np
    >>> x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
    >>> s = np.sin(x*y*z)/(x*y*z)
    >>> r = fos.ren()
    >>> v = fos.volume(s)
    >>> fos.add(r,v)
    >>> fos.show(r)
    
    If you find this function too complicated you can always use mayavi. 
    Please do not forget to use the -wthread switch in ipython if you are running mayavi.
    
    >>> from enthought.mayavi import mlab       
    >>> import numpy as np
    >>> x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
    >>> s = np.sin(x*y*z)/(x*y*z)
    >>> mlab.pipeline.volume(mlab.pipeline.scalar_field(s))
    >>> mlab.show()
    
    More mayavi demos are available here:
    
    http://code.enthought.com/projects/mayavi/docs/development/html/mayavi/mlab.html
    
    '''
    if vol.ndim!=3:    
        raise ValueError('3d numpy arrays only please')
    
    if info :
        print('Datatype',vol.dtype,'converted to uint8' )
    
    vol=np.interp(vol,[vol.min(),vol.max()],[0,255])
    vol=vol.astype('uint8')

    if opacitymap==None:
        
        bin,res=np.histogram(vol.ravel())
        res2=np.interp(res,[vol.min(),vol.max()],[0,1])
        opacitymap=np.vstack((res,res2)).T
        opacitymap=opacitymap.astype('float32')
                
        '''
        opacitymap=np.array([[ 0.0, 0.0],
                          [50.0, 0.9]])
        ''' 

    if info:
        print 'opacitymap', opacitymap
        
    if colormap==None:

        bin,res=np.histogram(vol.ravel())
        res2=np.interp(res,[vol.min(),vol.max()],[0,1])
        zer=np.zeros(res2.shape)
        colormap=np.vstack((res,res2,zer,res2[::-1])).T
        colormap=colormap.astype('float32')

        '''
        colormap=np.array([[0.0, 0.5, 0.0, 0.0],
                                        [64.0, 1.0, 0.5, 0.5],
                                        [128.0, 0.9, 0.2, 0.3],
                                        [196.0, 0.81, 0.27, 0.1],
                                        [255.0, 0.5, 0.5, 0.5]])
        '''

    if info:
        print 'colormap', colormap                        
    
    im = vtk.vtkImageData()
    im.SetScalarTypeToUnsignedChar()
    im.SetDimensions(vol.shape[0],vol.shape[1],vol.shape[2])
    #im.SetOrigin(0,0,0)
    #im.SetSpacing(voxsz[2],voxsz[0],voxsz[1])
    im.AllocateScalars()        
    
    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in range(vol.shape[2]):
                
                im.SetScalarComponentFromFloat(i,j,k,0,vol[i,j,k])
    
    if affine != None:

        aff = vtk.vtkMatrix4x4()
        aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],affine[0,3],affine[1,0],affine[1,1],affine[1,2],affine[1,3],affine[2,0],affine[2,1],affine[2,2],affine[2,3],affine[3,0],affine[3,1],affine[3,2],affine[3,3]))
        #aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],0,affine[1,0],affine[1,1],affine[1,2],0,affine[2,0],affine[2,1],affine[2,2],0,affine[3,0],affine[3,1],affine[3,2],1))
        #aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],127.5,affine[1,0],affine[1,1],affine[1,2],-127.5,affine[2,0],affine[2,1],affine[2,2],-127.5,affine[3,0],affine[3,1],affine[3,2],1))
        
        reslice = vtk.vtkImageReslice()
        reslice.SetInput(im)
        #reslice.SetOutputDimensionality(2)
        #reslice.SetOutputOrigin(127,-145,147)    
        
        reslice.SetResliceAxes(aff)
        #reslice.SetOutputOrigin(-127,-127,-127)    
        #reslice.SetOutputExtent(-127,128,-127,128,-127,128)
        #reslice.SetResliceAxesOrigin(0,0,0)
        #print 'Get Reslice Axes Origin ', reslice.GetResliceAxesOrigin()
        #reslice.SetOutputSpacing(1.0,1.0,1.0)
        
        reslice.SetInterpolationModeToLinear()    
        #reslice.UpdateWholeExtent()
        
        #print 'reslice GetOutputOrigin', reslice.GetOutputOrigin()
        #print 'reslice GetOutputExtent',reslice.GetOutputExtent()
        #print 'reslice GetOutputSpacing',reslice.GetOutputSpacing()
    
        changeFilter=vtk.vtkImageChangeInformation() 
        changeFilter.SetInput(reslice.GetOutput())
        #changeFilter.SetInput(im)
        if center_origin:
            changeFilter.SetOutputOrigin(-vol.shape[0]/2.0+0.5,-vol.shape[1]/2.0+0.5,-vol.shape[2]/2.0+0.5)
            print 'ChangeFilter ', changeFilter.GetOutputOrigin()
        
    opacity = vtk.vtkPiecewiseFunction()
    for i in range(opacitymap.shape[0]):
        opacity.AddPoint(opacitymap[i,0],opacitymap[i,1])

    color = vtk.vtkColorTransferFunction()
    for i in range(colormap.shape[0]):
        color.AddRGBPoint(colormap[i,0],colormap[i,1],colormap[i,2],colormap[i,3])
        
    if(maptype==0): 
    
        property = vtk.vtkVolumeProperty()
        property.SetColor(color)
        property.SetScalarOpacity(opacity)
        
        if trilinear:
            property.SetInterpolationTypeToLinear()
        else:
            prop.SetInterpolationTypeToNearest()
            
        if info:
            print('mapper VolumeTextureMapper2D')
        mapper = vtk.vtkVolumeTextureMapper2D()
        if affine == None:
            mapper.SetInput(im)
        else:
            #mapper.SetInput(reslice.GetOutput())
            mapper.SetInput(changeFilter.GetOutput())
        
    
    if (maptype==1):

        property = vtk.vtkVolumeProperty()
        property.SetColor(color)
        property.SetScalarOpacity(opacity)
        property.ShadeOn()
        if trilinear:
            property.SetInterpolationTypeToLinear()
        else:
            prop.SetInterpolationTypeToNearest()

        if iso:
            isofunc=vtk.vtkVolumeRayCastIsosurfaceFunction()
            isofunc.SetIsoValue(iso_thr)
        else:
            compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        
        if info:
            print('mapper VolumeRayCastMapper')
            
        mapper = vtk.vtkVolumeRayCastMapper()
        if iso:
            mapper.SetVolumeRayCastFunction(isofunc)
            if info:
                print('Isosurface')
        else:
            mapper.SetVolumeRayCastFunction(compositeFunction)   
            
            #mapper.SetMinimumImageSampleDistance(0.2)
            if info:
                print('Composite')
             
        if affine == None:
            mapper.SetInput(im)
        else:
            #mapper.SetInput(reslice.GetOutput())    
            mapper.SetInput(changeFilter.GetOutput())
            #Return mid position in world space    
            #im2=reslice.GetOutput()
            #index=im2.FindPoint(vol.shape[0]/2.0,vol.shape[1]/2.0,vol.shape[2]/2.0)
            #print 'Image Getpoint ' , im2.GetPoint(index)
           
        
    volum = vtk.vtkVolume()
    volum.SetMapper(mapper)
    volum.SetProperty(property)

    if info :  
         
        print 'Origin',   volum.GetOrigin()
        print 'Orientation',   volum.GetOrientation()
        print 'OrientationW',    volum.GetOrientationWXYZ()
        print 'Position',    volum.GetPosition()
        print 'Center',    volum.GetCenter()  
        print 'Get XRange', volum.GetXRange()
        print 'Get YRange', volum.GetYRange()
        print 'Get ZRange', volum.GetZRange()  
        print 'Volume data type', vol.dtype
        
    return volum
 
#File: viz.py Project: rbaravalle/Pysys
def viz():
    opaq = 0.01
     
    # We begin by creating the data we want to render.
    # For this tutorial, we create a 3D-image containing three overlaping cubes.
    # This data can of course easily be replaced by data from a medical CT-scan or anything else three dimensional.
    # The only limit is that the data must be reduced to unsigned 8 bit or 16 bit integers.
    img = Image.open('imagen3.png').convert('L')
    img = np.asarray(img)
    print img.shape

    Nx = sqrt(img.shape[0])
    Ny = Nx
    Nz = img.shape[1]

    data_matrix = zeros([Nx, Ny, Nz], dtype=uint8)

    for i in range(0,Nz-1):
         temp = img[Nx*i:Nx*(i+1),:]
         data_matrix[:,:,i] = np.uint8(255)-temp
    

    #for i in range(0,maxcoordZ-1):
    #    for k in range(0,maxcoord-1):
    #        data_matrix[k,:,i] = np.uint8(255)-np.array(occupied[i*maxcoord2+k*maxcoord:i*maxcoord2+(k+1)*maxcoord]).astype(np.uint8)

    #data_matrix = occupied#data_matrix[20:150, 20:150, 20:150] = randint(0,150)

    # For VTK to be able to use the data, it must be stored as a VTK-image. This can be done by the vtkImageImport-class which
    # imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()
    # The preaviusly created array is converted to a string of chars and imported.
    data_string = data_matrix.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    # The type of the newly imported data is set to unsigned char (uint8)
    dataImporter.SetDataScalarTypeToUnsignedChar()
    # Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
    # must be told this is the case.
    dataImporter.SetNumberOfScalarComponents(1)
    # The following two functions describe how the data is stored and the dimensions of the array it is stored in. For this
    # simple case, all axes are of length 75 and begins with the first element. For other data, this is probably not the case.
    # I have to admit however, that I honestly dont know the difference between SetDataExtent() and SetWholeExtent() although
    # VTK complains if not both are used.
    dataImporter.SetDataExtent(0, Nx-1, 0, Ny-1, 0, Nz-1)
    dataImporter.SetWholeExtent(0, Nx-1, 0, Ny-1, 0, Nz-1)
     
    # The following class is used to store transparencyv-values for later retrival. In our case, we want the value 0 to be
    # completly opaque whereas the three different cubes are given different transperancy-values to show how it works.
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0)
    alphaChannelFunc.AddPoint(255, opaq)
     
    # This class stores color data and can create color tables from a few color points. For this demo, we want the three cubes
    # to be of the colors red green and blue.
    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(255,0.8, 0.7, 0.6)
     
    # The preavius two classes stored properties. Because we want to apply these properties to the volume we want to render,
    # we have to store them in a class that stores volume prpoperties.
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)
     
    # This class describes how the volume is rendered (through ray tracing).
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    # We can finally create our volume. We also have to specify the data for it, as well as how the data will be rendered.
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
     
    # The class vtkVolume is used to pair the preaviusly declared volume as well as the properties to be used when rendering that volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
     
    # With almost everything else ready, its time to initialize the renderer and window, as well as creating a method for exiting the application
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)
     
    # We add the volume to the renderer ...
    renderer.AddVolume(volume)
    # ... set background color to white ...
    renderer.SetBackground(0,0,0)
    # ... and set window size.
    renderWin.SetSize(800, 800)
     
    # A simple function to be called when the user decides to quit the application.
    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)
     
    # Tell the application to use the function as an exit check.
    renderWin.AddObserver("AbortCheckEvent", exitCheck)
     
    renderInteractor.Initialize()
    # Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.
    renderWin.Render()
    renderInteractor.Start()
 
#File: show3.py Project: vlukes/lisa
def show3(data_matrix = None): # pragma: no coverage

    import vtk
# We begin by creating the data we want to render.
# For this tutorial, we create a 3D-image containing three overlaping cubes.
# This data can of course easily be replaced by data from a medical CT-scan or anything else three dimensional.
# The only limit is that the data must be reduced to unsigned 8 bit or 16 bit integers.
    import pdb; pdb.set_trace()
    if data_matrix == None:
        data_matrix = zeros([75, 75, 75], dtype=uint8)
        data_matrix[0:35, 0:35, 0:35] = 50
        data_matrix[25:55, 25:55, 25:55] = 100
        data_matrix[45:74, 45:74, 45:74] = 150
    else:
        data_matrix[data_matrix==1] = 50
        data_matrix[data_matrix==2] = 100
    val0 = 0
    val1 = 50
    val2 = 100
    val3 = 150

# For VTK to be able to use the data, it must be stored as a VTK-image. This can be done by the vtkImageImport-class which
# imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()
# The preaviusly created array is converted to a string of chars and imported.
    data_string = data_matrix.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
# The type of the newly imported data is set to unsigned char (uint8)
    dataImporter.SetDataScalarTypeToUnsignedChar()
# Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
# must be told this is the case.
    dataImporter.SetNumberOfScalarComponents(1)
# The following two functions describe how the data is stored and the dimensions of the array it is stored in. For this
# simple case, all axes are of length 75 and begins with the first element. For other data, this is probably not the case.
# I have to admit however, that I honestly dont know the difference between SetDataExtent() and SetWholeExtent() although
# VTK complains if not both are used.
    #dataImporter.SetDataExtent(0, 74, 0, 74, 0, 74)
    #dataImporter.SetWholeExtent(0, 74, 0, 74, 0, 74)
    dataImporter.SetDataExtent(0, data_matrix.shape[0]-1, 0, data_matrix.shape[1]-1, 0,data_matrix.shape[2]-1 )
    dataImporter.SetWholeExtent(0, data_matrix.shape[0]-1, 0, data_matrix.shape[1]-1, 0,data_matrix.shape[2]-1 )
    
# The following class is used to store transparencyv-values for later retrival. In our case, we want the value 0 to be
# completly opaque whereas the three different cubes are given different transperancy-values to show how it works.
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(val0, 0.0)
    alphaChannelFunc.AddPoint(val1, 0.05)
    alphaChannelFunc.AddPoint(val2, 0.1)
    alphaChannelFunc.AddPoint(val3, 0.2)
    
# This class stores color data and can create color tables from a few color points. For this demo, we want the three cubes
# to be of the colors red green and blue.
    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(val1, 1.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(val2, 0.0, 1.0, 0.0)
    colorFunc.AddRGBPoint(val3, 0.0, 0.0, 1.0)
    
# The preavius two classes stored properties. Because we want to apply these properties to the volume we want to render,
# we have to store them in a class that stores volume prpoperties.
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)
    
# This class describes how the volume is rendered (through ray tracing).
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
# We can finally create our volume. We also have to specify the data for it, as well as how the data will be rendered.
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
    
# The class vtkVolume is used to pair the preaviusly declared volume as well as the properties to be used when rendering that volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    
# With almost everything else ready, its time to initialize the renderer and window, as well as creating a method for exiting the application
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)
    
# We add the volume to the renderer ...
    renderer.AddVolume(volume)
# ... set background color to white ...
    renderer.SetBackground(0,0,0)
# ... and set window size.
    renderWin.SetSize(400, 400)
    
# A simple function to be called when the user decides to quit the application.
    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)
    
# Tell the application to use the function as an exit check.
    renderWin.AddObserver("AbortCheckEvent", exitCheck)
    
    renderInteractor.Initialize()
# Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.
    renderWin.Render()
    renderInteractor.Start()
    import pdb; pdb.set_trace()
 
#File: render_3d.py Project: ricleal/PythonCode
    def initialise(self):
        dataImporter = vtk.vtkImageImport()
        data_string = self.data_matrix.tostring()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(1)
        dataImporter.SetDataExtent(0, self.data_matrix.shape[0]-1,
                                   0, self.data_matrix.shape[1]-1,
                                   0, self.data_matrix.shape[2]-1)
        dataImporter.SetWholeExtent(0, self.data_matrix.shape[0]-1,
                                   0, self.data_matrix.shape[1]-1,
                                   0, self.data_matrix.shape[2]-1)
        
        
        alphaChannelFunc = vtk.vtkPiecewiseFunction()
        colorFunc = vtk.vtkColorTransferFunction()
        
        for i in range(int(self.data_min),int(self.data_max)):
            alphaChannelFunc.AddPoint(i, i/self.data_max )
            colorFunc.AddRGBPoint(i,i/self.data_max,i/self.data_max,i/self.data_max)
        # for our test sample, we set the black opacity to 0 (transparent) so as
        #to see the sample  
        alphaChannelFunc.AddPoint(0, 0.0)
        colorFunc.AddRGBPoint(0,0,0,0)


        
        
        
        
        
        
        
        
        
         
#         alphaChannelFunc = vtk.vtkPiecewiseFunction()
#         alphaChannelFunc.AddPoint(self.data_min, 0.0)
#         alphaChannelFunc.AddPoint(self.data_max - self.data_min /2, 0.1)
#         alphaChannelFunc.AddPoint(self.data_max, 0.2)
#          
#         # This class stores color data and can create color tables from a few color points. For this demo, we want the three cubes
#         # to be of the colors red green and blue.
#         colorFunc = vtk.vtkColorTransferFunction()
#         colorFunc.AddRGBPoint(50, 1.0, 0.0, 0.0)
#         colorFunc.AddRGBPoint(100, 0.0, 1.0, 0.0)
#         colorFunc.AddRGBPoint(150, 0.0, 0.0, 1.0)
         
        # The preavius two classes stored properties. Because we want to apply these properties to the volume we want to render,
        # we have to store them in a class that stores volume prpoperties.
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorFunc)
        volumeProperty.SetScalarOpacity(alphaChannelFunc)
         
        # This class describes how the volume is rendered (through ray tracing).
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        # We can finally create our volume. We also have to specify the data for it, as well as how the data will be rendered.
        volumeMapper = vtk.vtkVolumeRayCastMapper()
        volumeMapper.SetVolumeRayCastFunction(compositeFunction)
        volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
         
        # The class vtkVolume is used to pair the preaviusly declared volume as well as the properties to be used when rendering that volume.
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(volumeMapper)
        self.volume.SetProperty(volumeProperty)
 
#File: Assignment4_old.py Project: JaroCamphuijsen/SVVR
def createVolumeDict():
    global colourDict
    # 0 is skin colour
    colourDict = {
        1.5: [1.5, 0.0, 1.0, 0.0],
        1: [1.0, 0.75, 0.0, 0.0],
        2: [2.0, 0.65, 0.65, 0.6],
        3: [1.0, 0.75, 0.0, 0.0],
        4: [4.0, 1.0, 1.0, 0.0],
        5: [1.0, 0.75, 0.0, 0.0],
        6: [1.0, 0.75, 0.0, 0.0],
        7: [7.0, 0.0, 1.0, 0.0],
        8: [1.0, 0.75, 0.0, 0.0],
        9: [1.0, 0.75, 0.0, 0.0],
        10: [10.0, 0.0, 1.0, 1.0],
        11: [1.0, 0.75, 0.0, 0.0],
        12: [1.0, 0.75, 0.0, 0.0],
        13: [13.0, 1.0, 1.0, 1.0],
        14: [1.0, 0.75, 0.0, 0.0],
        15: [1.0, 0.75, 0.0, 0.0],
    }
    # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    global volumeDict
    global lut
    volumeDict = {}

    # opacityTransferFunction = createOpacityTransferFunction([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

    for value, colourArray in colourDict.iteritems():
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(value, colourArray[1], colourArray[2], colourArray[3])

        opacityTransferFunction = createOpacityTransferFunction([value])

        # Skin
        if value == 1.5:
            opacityTransferFunction = vtk.vtkPiecewiseFunction()
            opacityTransferFunction.AddPoint(0, 0)
            opacityTransferFunction.AddPoint(1.5, 0.5)
            opacityTransferFunction.AddPoint(3, 0)

        # for value, colourArray in colourDict.iteritems():
        # The property describes how the data will look
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        # volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()

        # The mapper / ray cast function know how to render the data
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        volumeMapper = vtk.vtkVolumeRayCastMapper()
        volumeMapper.SetVolumeRayCastFunction(compositeFunction)
        if value == 1.5:
            volumeMapper.SetInputConnection(readerSkin.GetOutputPort())
        else:
            volumeMapper.SetInputConnection(reader.GetOutputPort())

        # The volume holds the mapper and the property and
        # can be used to position/orient the volume
        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)

        volumeDict[value] = volume

    print volumeDict
 
#File: imgVolRender_GPU.py Project: Hanbusy/PyVolRender
	def addVol(self, data, header=None):
		pix_diag = 5.0/10.0

		img = vtkImageImportFromArray()
		img.SetArray(data)
		img.ConvertIntToUnsignedShortOn()
		'''
		Origin and Data spacing setting are essential for a normalized volume rendering of
		the DWI image volumes
		------- dawdling for a long time for addressing the problem that the volume is too thin
		and even resorted to pre-resampling of the DWI volumes
		'''
		#img.GetImport().SetDataSpacing(0.9375, 0.9375, 4.5200)
		img.GetImport().SetDataSpacing(header['pixdim'][1:4])
		#img.GetImport().SetDataOrigin(128.0, 128.0, 68.50)
		img.GetImport().SetDataOrigin( 
				header['dim'][0]*header['pixdim'][0],
				header['dim'][1]*header['pixdim'][1],
				header['dim'][2]*header['pixdim'][2])
		print img.GetDataExtent()
 
		volMapper = vtk.vtkGPUVolumeRayCastMapper()
		compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
		compositeFunction.SetCompositeMethodToInterpolateFirst()
		#compositeFunction.SetCompositeMethodToClassifyFirst()
		#volMapper.SetVolumeRayCastFunction(compositeFunction)

		#volMapper.SetSampleDistance(pix_diag / 5.0)
		volMapper.SetImageSampleDistance( .5 )
		volMapper.SetSampleDistance(1.0)
		volMapper.SetInputConnection( img.GetOutputPort() )
		volMapper.SetBlendModeToComposite()

		# The property describes how the data will look
		self.volProperty = volProperty = vtk.vtkVolumeProperty()
		volProperty.SetColor(self.color_tf)
		volProperty.SetScalarOpacity(self.opacity_tf)
		volProperty.SetGradientOpacity(self.opacity_tf)
		if self.parent.lighting:
			volProperty.ShadeOn()
		#volProperty.SetInterpolationTypeToLinear()
		volProperty.SetInterpolationTypeToNearest()
		volProperty.SetScalarOpacityUnitDistance(pix_diag/5.0)

		vol = vtk.vtkVolume()
		vol.SetMapper(volMapper)
		vol.SetProperty(volProperty)

		self.ren.AddVolume(vol)

		boxWidget = vtk.vtkBoxWidget()
		boxWidget.SetInteractor(self.parent.m_ui.renderView)
		boxWidget.SetPlaceFactor(1.0)

		planes = vtk.vtkPlanes()
		def ClipVolumeRender(obj, event):
			obj.GetPlanes(planes)
			volMapper.SetClippingPlanes(planes)
         
		boxWidget.SetInput(img.GetOutput())
		boxWidget.PlaceWidget(img.GetOutput().GetBounds())
		boxWidget.InsideOutOn()
		boxWidget.AddObserver("InteractionEvent", ClipVolumeRender)

		outlineProperty = boxWidget.GetOutlineProperty()
		outlineProperty.SetRepresentationToWireframe()
		outlineProperty.SetAmbient(1.0)
		outlineProperty.SetAmbientColor(1, 1, 1)
		outlineProperty.SetLineWidth(3)

		selectedOutlineProperty = boxWidget.GetSelectedOutlineProperty()
		selectedOutlineProperty.SetRepresentationToWireframe()
		selectedOutlineProperty.SetAmbient(1.0)
		selectedOutlineProperty.SetAmbientColor(1, 0, 0)
		selectedOutlineProperty.SetLineWidth(1)

		outline = vtk.vtkOutlineFilter()
		outline.SetInputConnection(img.GetOutputPort())
		outlineMapper = vtk.vtkPolyDataMapper()
		outlineMapper.SetInputConnection(outline.GetOutputPort())
		outlineActor = vtk.vtkActor()
		outlineActor.SetMapper(outlineMapper)

		self.ren.AddActor(outlineActor)
		self.volnum += 1
 
#File: vtkTest5.py Project: avicramer/Image-Segmentation-Tools
def displayer (img):
    img = Image.open(img)
    rsFactor = 0.5
    img1 = img.resize( ( (int(img.size[0]*rsFactor)),(int(img.size[1]*rsFactor))) ,Image.ANTIALIAS)
    imarray = array(img1)
    img.seek(1)
    img1 = img.resize( ( (int(img.size[0]*rsFactor)),(int(img.size[1]*rsFactor))) ,Image.ANTIALIAS)
    imarray = array([imarray, array(img1)])
    for i in range(2,179):
        img.seek(i)
        img1 = img.resize( ( (int(img.size[0]*rsFactor)),(int(img.size[1]*rsFactor))) ,Image.ANTIALIAS)
        a = array(img1)
        imarray = concatenate((imarray, [a]), axis=0)
 
    data_matrix = (imarray + 18 * ones(imarray.shape)).astype('uint8')

# For VTK to be able to use the data, it must be stored as a VTK-image. This can be done by the vtkImageImport-class which
# imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()
# The preaviusly created array is converted to a string of chars and imported.
    data_string = data_matrix.tobytes(order='F')
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
# The type of the newly imported data is set to unsigned char (uint8)
    dataImporter.SetDataScalarTypeToUnsignedChar()
# Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
# must be told this is the case.
    dataImporter.SetNumberOfScalarComponents(1)
# The following two functions describe how the data is stored and the dimensions of the array it is stored in. For this
# simple case, all axes are of length 75 and begins with the first element. For other data, this is probably not the case.
# I have to admit however, that I honestly dont know the difference between SetDataExtent() and SetWholeExtent() although
# VTK complains if not both are used.
    dataImporter.SetDataExtent(0, 178, 0, 395, 0, 186)
    dataImporter.SetWholeExtent(0, 178, 0, 395, 0, 186)
 
# The following class is used to store transparencyv-values for later retrival. In our case, we want the value 0 to be
# completly opaque whereas the three different cubes are given different transperancy-values to show how it works.
#alphaChannelFunc = vtk.vtkPiecewiseFunction()
#alphaChannelFunc.AddPoint(0, 0.0)
#alphaChannelFunc.AddPoint(50, 0.05)
#alphaChannelFunc.AddPoint(100, 0.1)
#alphaChannelFunc.AddPoint(150, 0.2)
 
# This class stores color data and can create color tables from a few color points. For this demo, we want the three cubes
# to be of the colors red green and blue.
#colorFunc = vtk.vtkColorTransferFunction()
#colorFunc.AddRGBPoint(50, 1.0, 0.0, 0.0)
#colorFunc.AddRGBPoint(100, 0.0, 1.0, 0.0)
#colorFunc.AddRGBPoint(150, 0.0, 0.0, 1.0)
 
# The preavius two classes stored properties. Because we want to apply these properties to the volume we want to render,
# we have to store them in a class that stores volume prpoperties.
    volumeProperty = vtk.vtkVolumeProperty()
#volumeProperty.SetColor(colorFunc)
#volumeProperty.SetScalarOpacity(alphaChannelFunc)
 
# This class describes how the volume is rendered (through ray tracing).
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
# We can finally create our volume. We also have to specify the data for it, as well as how the data will be rendered.
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
 
# The class vtkVolume is used to pair the preaviusly declared volume as well as the properties to be used when rendering that volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
 
# With almost everything else ready, its time to initialize the renderer and window, as well as creating a method for exiting the application
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)
 
# We add the volume to the renderer ...
    renderer.AddVolume(volume)
# ... set background color to white ...
    renderer.SetBackground(0,0,0)
# ... and set window size.
    renderWin.SetSize(400, 400)
 

# Tell the application to use the function as an exit check.
    renderWin.AddObserver("AbortCheckEvent", exitCheck)
 
    renderInteractor.Initialize()
# Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.
    renderWin.Render()
    renderInteractor.Start()
 
#File: VolumePicker.py Project: 151706061/VTK
    def testVolumePicker(self):
        # volume render a medical data set

        # renderer and interactor
        ren = vtk.vtkRenderer()

        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)

        iRen = vtk.vtkRenderWindowInteractor()
        iRen.SetRenderWindow(renWin)

        # read the volume
        v16 = vtk.vtkVolume16Reader()
        v16.SetDataDimensions(64, 64)
        v16.SetImageRange(1, 93)
        v16.SetDataByteOrderToLittleEndian()
        v16.Set#FilePrefix(VTK_DATA_ROOT + "/Data/headsq/quarter")
        v16.SetDataSpacing(3.2, 3.2, 1.5)

        #---------------------------------------------------------
        # set up the volume rendering

        rayCastFunction = vtk.vtkVolumeRayCastCompositeFunction()

        volumeMapper = vtk.vtkVolumeRayCastMapper()
        volumeMapper.SetInputConnection(v16.GetOutputPort())
        volumeMapper.SetVolumeRayCastFunction(rayCastFunction)

        volumeColor = vtk.vtkColorTransferFunction()
        volumeColor.AddRGBPoint(0, 0.0, 0.0, 0.0)
        volumeColor.AddRGBPoint(180, 0.3, 0.1, 0.2)
        volumeColor.AddRGBPoint(1000, 1.0, 0.7, 0.6)
        volumeColor.AddRGBPoint(2000, 1.0, 1.0, 0.9)

        volumeScalarOpacity = vtk.vtkPiecewiseFunction()
        volumeScalarOpacity.AddPoint(0, 0.0)
        volumeScalarOpacity.AddPoint(180, 0.0)
        volumeScalarOpacity.AddPoint(1000, 0.2)
        volumeScalarOpacity.AddPoint(2000, 0.8)

        volumeGradientOpacity = vtk.vtkPiecewiseFunction()
        volumeGradientOpacity.AddPoint(0, 0.0)
        volumeGradientOpacity.AddPoint(90, 0.5)
        volumeGradientOpacity.AddPoint(100, 1.0)

        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(volumeColor)
        volumeProperty.SetScalarOpacity(volumeScalarOpacity)
        volumeProperty.SetGradientOpacity(volumeGradientOpacity)
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.ShadeOn()
        volumeProperty.SetAmbient(0.6)
        volumeProperty.SetDiffuse(0.6)
        volumeProperty.SetSpecular(0.1)

        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)

        #---------------------------------------------------------
        # Do the surface rendering
        boneExtractor = vtk.vtkMarchingCubes()
        boneExtractor.SetInputConnection(v16.GetOutputPort())
        boneExtractor.SetValue(0, 1150)

        boneNormals = vtk.vtkPolyDataNormals()
        boneNormals.SetInputConnection(boneExtractor.GetOutputPort())
        boneNormals.SetFeatureAngle(60.0)

        boneStripper = vtk.vtkStripper()
        boneStripper.SetInputConnection(boneNormals.GetOutputPort())

        boneMapper = vtk.vtkPolyDataMapper()
        boneMapper.SetInputConnection(boneStripper.GetOutputPort())
        boneMapper.ScalarVisibilityOff()

        boneProperty = vtk.vtkProperty()
        boneProperty.SetColor(1.0, 1.0, 0.9)

        bone = vtk.vtkActor()
        bone.SetMapper(boneMapper)
        bone.SetProperty(boneProperty)

        #---------------------------------------------------------
        # Create an image actor

        table = vtk.vtkLookupTable()
        table.SetRange(0, 2000)
        table.SetRampToLinear()
        table.SetValueRange(0, 1)
        table.SetHueRange(0, 0)
        table.SetSaturationRange(0, 0)

        mapToColors = vtk.vtkImageMapToColors()
        mapToColors.SetInputConnection(v16.GetOutputPort())
        mapToColors.SetLookupTable(table)

        imageActor = vtk.vtkImageActor()
        imageActor.GetMapper().SetInputConnection(mapToColors.GetOutputPort())
        imageActor.SetDisplayExtent(32, 32, 0, 63, 0, 92)

        #---------------------------------------------------------
        # make a transform and some clipping planes

        transform = vtk.vtkTransform()
        transform.RotateWXYZ(-20, 0.0, -0.7, 0.7)

        volume.SetUserTransform(transform)
        bone.SetUserTransform(transform)
        imageActor.SetUserTransform(transform)

        c = volume.GetCenter()

        volumeClip = vtk.vtkPlane()
        volumeClip.SetNormal(0, 1, 0)
        volumeClip.SetOrigin(c)

        boneClip = vtk.vtkPlane()
        boneClip.SetNormal(0, 0, 1)
        boneClip.SetOrigin(c)

        volumeMapper.AddClippingPlane(volumeClip)
        boneMapper.AddClippingPlane(boneClip)

        #---------------------------------------------------------
        ren.AddViewProp(volume)
        ren.AddViewProp(bone)
        ren.AddViewProp(imageActor)

        camera = ren.GetActiveCamera()
        camera.SetFocalPoint(c)
        camera.SetPosition(c[0] + 500, c[1] - 100, c[2] - 100)
        camera.SetViewUp(0, 0, -1)

        ren.ResetCameraClippingRange()

        renWin.Render()

        #---------------------------------------------------------
        # the cone should point along the Z axis
        coneSource = vtk.vtkConeSource()
        coneSource.CappingOn()
        coneSource.SetHeight(12)
        coneSource.SetRadius(5)
        coneSource.SetResolution(31)
        coneSource.SetCenter(6, 0, 0)
        coneSource.SetDirection(-1, 0, 0)

        #---------------------------------------------------------
        picker = vtk.vtkVolumePicker()
        picker.SetTolerance(1.0e-6)
        picker.SetVolumeOpacityIsovalue(0.01)
        # This should usually be left alone, but is used here to increase coverage
        picker.UseVolumeGradientOpacityOn()

        # A function to point an actor along a vector
        def PointCone(actor, n):
            if n[0] < 0.0:
                actor.RotateWXYZ(180, 0, 1, 0)
                actor.RotateWXYZ(180, (n[0] - 1.0) * 0.5, n[1] * 0.5, n[2] * 0.5)
            else:
                actor.RotateWXYZ(180, (n[0] + 1.0) * 0.5, n[1] * 0.5, n[2] * 0.5)

        # Pick the actor
        picker.Pick(192, 103, 0, ren)
        #print picker
        p = picker.GetPickPosition()
        n = picker.GetPickNormal()

        coneActor1 = vtk.vtkActor()
        coneActor1.PickableOff()
        coneMapper1 = vtk.vtkDataSetMapper()
        coneMapper1.SetInputConnection(coneSource.GetOutputPort())
        coneActor1.SetMapper(coneMapper1)
        coneActor1.GetProperty().SetColor(1, 0, 0)
        coneActor1.SetPosition(p)
        PointCone(coneActor1, n)
        ren.AddViewProp(coneActor1)

        # Pick the volume
        picker.Pick(90, 180, 0, ren)
        p = picker.GetPickPosition()
        n = picker.GetPickNormal()

        coneActor2 = vtk.vtkActor()
        coneActor2.PickableOff()
        coneMapper2 = vtk.vtkDataSetMapper()
        coneMapper2.SetInputConnection(coneSource.GetOutputPort())
        coneActor2.SetMapper(coneMapper2)
        coneActor2.GetProperty().SetColor(1, 0, 0)
        coneActor2.SetPosition(p)
        PointCone(coneActor2, n)
        ren.AddViewProp(coneActor2)

        # Pick the image
        picker.Pick(200, 200, 0, ren)
        p = picker.GetPickPosition()
        n = picker.GetPickNormal()

        coneActor3 = vtk.vtkActor()
        coneActor3.PickableOff()
        coneMapper3 = vtk.vtkDataSetMapper()
        coneMapper3.SetInputConnection(coneSource.GetOutputPort())
        coneActor3.SetMapper(coneMapper3)
        coneActor3.GetProperty().SetColor(1, 0, 0)
        coneActor3.SetPosition(p)
        PointCone(coneActor3, n)
        ren.AddViewProp(coneActor3)

        # Pick a clipping plane
        picker.PickClippingPlanesOn()
        picker.Pick(145, 160, 0, ren)
        p = picker.GetPickPosition()
        n = picker.GetPickNormal()

        coneActor4 = vtk.vtkActor()
        coneActor4.PickableOff()
        coneMapper4 = vtk.vtkDataSetMapper()
        coneMapper4.SetInputConnection(coneSource.GetOutputPort())
        coneActor4.SetMapper(coneMapper4)
        coneActor4.GetProperty().SetColor(1, 0, 0)
        coneActor4.SetPosition(p)
        PointCone(coneActor4, n)
        ren.AddViewProp(coneActor4)

        ren.ResetCameraClippingRange()

        # render and interact with data

        renWin.Render()

        img_file = "VolumePicker.png"
        vtk.test.Testing.compareImage(iRen.GetRenderWindow(), vtk.test.Testing.getAbsImagePath(img_file), threshold=25)
        vtk.test.Testing.interact()
 
#File: display3D_volren.py Project: nicolasBeucher/mamba-image
 def __init__(self, master):
     global _vtk_lib_present
     if not _vtk_lib_present:
         raise ValueError("no VTK")
 
     # Window creation
     tk.Frame.__init__(self, master)
     self.columnconfigure(0, weight=1)
     self.rowconfigure(0, weight=1)
     
     # Renderer and associated widget
     self.im_ref = None
     self._renWidget = vtkTkRenderWidget(self)
     self._ren = vtk.vtkRenderer()
     self._renWidget.GetRenderWindow().AddRenderer(self._ren)
     self._renWidget.grid(row=0, column=0, sticky=tk.E+tk.W+tk.N+tk.S)
     
     # Transfer functions and volume display options and properties
     self.vtk_im = vtkImageImport()
     self.vtk_im.SetDataScalarType(VTK_UNSIGNED_CHAR)
     self.im_flipy = vtk.vtkImageFlip()
     self.im_flipy.SetFilteredAxis(1)
     self.im_flipy.SetInputConnection(self.vtk_im.GetOutputPort());
     self.im_flipz = vtk.vtkImageFlip()
     self.im_flipz.SetFilteredAxis(2)
     self.im_flipz.SetInputConnection(self.im_flipy.GetOutputPort());
     self.opaTF = vtk.vtkPiecewiseFunction()
     self.colTF = vtk.vtkColorTransferFunction()
     self.volProp = vtk.vtkVolumeProperty()
     self.volProp.SetColor(self.colTF)
     self.volProp.SetScalarOpacity(self.opaTF)
     self.volProp.ShadeOn()
     self.volProp.SetInterpolationTypeToLinear()
     self.compoFun = vtk.vtkVolumeRayCastCompositeFunction()
     self.isosfFun = vtk.vtkVolumeRayCastIsosurfaceFunction()
     self.isosfFun.SetIsoValue(0)
     self.mipFun = vtk.vtkVolumeRayCastMIPFunction()
     self.volMap = vtk.vtkVolumeRayCastMapper()
     self.volMap.SetVolumeRayCastFunction(self.compoFun)
     self.volMap.SetInputConnection(self.im_flipz.GetOutputPort())
     self.volume = vtk.vtkVolume()
     self.volume.SetMapper(self.volMap)
     self.volume.SetProperty(self.volProp)
     self.outlineData = vtk.vtkOutlineFilter()
     self.outlineData.SetInputConnection(self.im_flipz.GetOutputPort())
     self.mapOutline = vtk.vtkPolyDataMapper()
     self.mapOutline.SetInputConnection(self.outlineData.GetOutputPort())
     self.outline = vtk.vtkActor()
     self.outline.SetMapper(self.mapOutline)
     self.outline.GetProperty().SetColor(1, 1, 1)
     self._ren.AddVolume(self.volume)
     self._ren.AddActor(self.outline)
     self._ren.SetBackground(116/255.0,214/255.0,220/255.0)
     
     # Control widget
     self.controlbar = ttk.Frame(self)
     self.controlbar.grid(row=0, column=1,
                          sticky=tk.E+tk.W+tk.N+tk.S)
     self.drawControlBar()
     self.controlbar.grid_remove()
     self.controlbar.state = "hidden"
     self.master = master
     
     # Creates the info status bar.
     statusbar = ttk.Frame(self)
     statusbar.columnconfigure(0, weight=1)
     statusbar.grid(row=1, column=0, columnspan=2, sticky=tk.E+tk.W)
     self.infos = []
     for i in range(3):
         v = tk.StringVar(self)
         ttk.Label(statusbar, anchor=tk.W, textvariable=v).grid(row=0, column=i, sticky=tk.E+tk.W)
         self.infos.append(v)
     self.infos[2].set("Hit Tab for control <-")
         
     # Events bindings
     master.bind("<KeyPress-Tab>", self.displayControlEvent)
 
#File: viz.viewCube.py Project: jiahao/openqube
def RenderCubeInVTK(filename = 'test.cube', mindatum = 0.0, maxdatum = 0.0):
    global renWin

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    #######################
    # Read in Gaussian cube
    #######################

    CubeData = vtk.vtkGaussianCubeReader()
    CubeData.Set#FileName(filename)

    CubeData.Update()

    #Get intrinsic scale from data
    scale = sum([x**2 for x in CubeData.GetTransform().GetScale()])

    CubeData.SetHBScale(scale) #scaling factor to compute bonds with hydrogens
    CubeData.SetBScale(scale)  #scaling factor for other bonds

    CubeData.Update()

    ###################
    #Calculate scalings

    #VTK only knows how to render integer data in the interval [0,255] or [0,65535]
    #Here, we calculate scaling factors to map the cube data to the interval.

    if mindatum == maxdatum == 0.0:
        if DEBUG:
            print "Autodetecting range"
            mindatum, maxdatum = CubeData.GetGridOutput().GetPointData().GetScalars().GetRange()

    # Find the remapped value that corresponds to zero
    zeropoint = int(2**ColorDepth*(-mindatum)/(maxdatum-mindatum))
    absmaxdatum = max(-mindatum, maxdatum)

    maxnegativeintensity = min(1.0, 1.0 - (absmaxdatum - abs(mindatum))/absmaxdatum)
    minnegativeintensity = 0.0
    if zeropoint < 0:
        minpositiveintensity = - zeropoint/(2**ColorDepth*absmaxdatum)
    else:
        minpositiveintensity = 0.0
        maxpositiveintensity = min(1.0, 1.0 - (absmaxdatum - abs(maxdatum))/absmaxdatum)
    if DEBUG:
        print "Range plotted = [%f,%f]" % (mindatum, maxdatum)
        print "Negative colors = [0,%d)" % max(0,zeropoint)
        print "Negative intensities = [%f,%f]" % (maxnegativeintensity,minnegativeintensity)
        print "Positive colors = (%d,%d)" % (max(0,zeropoint), 2**ColorDepth)
        print "Positive intensities = [%f,%f]" % (minpositiveintensity,maxpositiveintensity)
        print "On this scale, zero = %d" % zeropoint

    ################################
    # Calculate opacity transfer map

    #The code here differentiates between two cases:
    #1. the scalar data are all positive, so it's just a simple linear ramp
    #2. the scalar data are signed, so do two linear ramps

    opacityTransferFunction = vtk.vtkPiecewiseFunction()

    if zeropoint < 0:
        opacityTransferFunction.AddPoint(        0, minpositiveintensity)
    else:
        opacityTransferFunction.AddPoint(        0, maxnegativeintensity)
        opacityTransferFunction.AddPoint(zeropoint, 0.0)

    opacityTransferFunction.AddPoint(2**ColorDepth-1, maxpositiveintensity)
    opacityTransferFunction.ClampingOn()

    ###########################
    # Create color transfer map

    colorTransferFunction = vtk.vtkColorTransferFunction()

    r1, g1, b1 = NegativeColor
    r2, g2, b2 = PositiveColor
    r0, g0, b0 = BackgroundColor

    if zeropoint < 0:
        colorTransferFunction.AddRGBPoint(          0, r1, g1, b1)
    else:
        colorTransferFunction.AddRGBPoint(          0, r1, g1, b1)
        colorTransferFunction.AddRGBPoint(zeropoint-1, r1, g1, b1)
        colorTransferFunction.AddRGBPoint(zeropoint  , r0, g0, b0)
        colorTransferFunction.AddRGBPoint(zeropoint+1, r2, g2, b2)
    colorTransferFunction.AddRGBPoint(2**ColorDepth-1, r2, g2, b2)

    ########################
    # Now apply the scalings

    ScaledData = vtk.vtkImageShiftScale()
    ScaledData.SetInput(CubeData.GetGridOutput())
    ScaledData.SetShift(-mindatum)
    ScaledData.SetScale((2**ColorDepth-1)/(maxdatum-mindatum))

    if ColorDepth == 16:
        ScaledData.SetOutputScalarTypeToUnsignedShort()
    elif ColorDepth == 8:
        ScaledData.SetOutputScalarTypeToUnsignedChar()
    else:
        print
        print "Error! Unsupported color depth given"
        print
        print "valid values are 8 or 16"
        print
        raise ValueError

    ###############################
    # Form combined coloring scheme

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()

    # The mapper / ray cast function know how to render the data
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()

    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInput(ScaledData.GetOutput())

    #Create a coarse representation
    #Actually a fake - won't display anything

    compositeFunction2 = vtk.vtkVolumeRayCastIsosurfaceFunction()
    compositeFunction2.SetIsoValue(2**ColorDepth-1)

    volumeMapperCoarse = vtk.vtkVolumeRayCastMapper()
    volumeMapperCoarse.SetVolumeRayCastFunction(compositeFunction2)
    volumeMapperCoarse.SetInput(ScaledData.GetOutput())

    # Create volumetric object to be rendered
    # Use level of detail prop so that it won't take forever to look around

    volume = vtk.vtkLODProp3D()
    id1 = volume.AddLOD(volumeMapper, volumeProperty, 0.)
    volume.SetLODProperty(id1, volumeProperty)
    id2 = volume.AddLOD(volumeMapperCoarse, volumeProperty, 0.)
    volume.SetLODProperty(id2, volumeProperty)

    # At this point, we can position and orient the volume

    #################################
    # End of volumetric data pipeline
    #################################

    #########
    #Contours
    #########

    contour = vtk.vtkContourFilter()
    contour.SetInput(CubeData.GetGridOutput())
    contour.SetNumberOfContours(1)
    contour.SetValue(0, 0.0)

    contourMapper = vtk.vtkPolyDataMapper()
    contourMapper.SetInput(contour.GetOutput())
    contourMapper.SetScalarRange(0,0)
    contourMapper.GetLookupTable().SetNumberOfTableValues(1)
    r0, g0, b0 = NodeColor
    contourMapper.GetLookupTable().SetTableValue(0, r0, g0, b0, NodeAlpha)

    contourActor = vtk.vtkLODActor()
    contourActor.SetMapper(contourMapper)
    contourActor.GetProperty().SetOpacity(NodeAlpha)

    ##########################################
    # Create a wireframe outline of the volume
    ##########################################

    frame = vtk.vtkOutlineFilter()
    frame.SetInput(CubeData.GetGridOutput())

    frameMapper = vtk.vtkPolyDataMapper()
    frameMapper.SetInput(frame.GetOutput())

    frameActor = vtk.vtkLODActor()
    frameActor.SetMapper(frameMapper)
    frameActor.GetProperty().SetColor(FrameColor)
    frameActor.GetProperty().SetOpacity(FrameAlpha)

    ######################
    # Draw balls for atoms
    ######################

    Sphere = vtk.vtkSphereSource()
    Sphere.SetThetaResolution(16)
    Sphere.SetPhiResolution(16)
    Sphere.SetRadius(0.4)

    Glyph = vtk.vtkGlyph3D()
    Glyph.SetInput(CubeData.GetOutput())
    Glyph.SetColorMode(1)
    Glyph.SetColorModeToColorByScalar()
    Glyph.SetScaleModeToScaleByVectorComponents()
    Glyph.SetSource(Sphere.GetOutput())

    AtomsMapper = vtk.vtkPolyDataMapper()
    AtomsMapper.SetInput(Glyph.GetOutput())
    AtomsMapper.SetImmediateModeRendering(1)
    AtomsMapper.UseLookupTableScalarRangeOff()
    AtomsMapper.SetScalarVisibility(1)
    AtomsMapper.SetScalarModeToDefault()

    Atoms = vtk.vtkLODActor()
    Atoms.SetMapper(AtomsMapper)

    ############
    # Draw bonds
    ############

    Tube = vtk.vtkTubeFilter()
    Tube.SetInput(CubeData.GetOutput())

    BondsMapper = vtk.vtkPolyDataMapper()
    BondsMapper.SetInput(Tube.GetOutput())

    Bonds = vtk.vtkLODActor()
    Bonds.SetMapper(BondsMapper)

    #######################
    # Now compose the image
    #######################
    if DrawVolume:
        ren.AddVolume(volume)

    if DrawNodes:
        ren.AddActor(contourActor)

    if DrawFrame:
        ren.AddActor(frameActor)

    if DrawAtoms:
        ren.AddActor(Atoms)

    if DrawBonds:
        ren.AddActor(Bonds)

    ren.SetBackground(BackgroundColor)
    renWin.SetSize(OutputHeight, OutputWidth)


    ######################################
    # Let VTK do its magic and render away
    ######################################

    renWin.Render()

    ###################################
    # Now allow user to play with image
    ###################################

    def Keypress(obj, event):
        #This function handles keyboard interaction

        key = obj.GetKeySym()

        if key == 'd' or key == 'F13':
            WriteToPNG()
        elif key == 'h' or key == 'question' or key =='?':
            PrintHelp()
        elif key == 'c':
            camera = ren.GetActiveCamera()
            print "Camera info:"
            print "------------"
            print "Position is: ", camera.GetPosition()
            print "Focal point is:", camera.GetFocalPoint()
            print "Orientation is:", ren.GetActiveCamera().GetOrientation()
            print "WXYZ", ren.GetActiveCamera().GetOrientationWXYZ()
            print "View up direction is:", camera.GetViewUp()
            print "Direction of projection is:", camera.GetDirectionOfProjection()

        else:
            if DEBUG:
                print 'User pressed key:', key 

    if Interactive:
        iren.SetDesiredUpdateRate(25.0) #25 fps when camera is moving around
        iren.SetStillUpdateRate(0.0) #0 fps when camera is not moving

        iren.Initialize()

        #The default interaction style is joystick, which seems unnatural
        style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)

        iren.AddObserver("KeyPressEvent", Keypress)
        iren.Start()
    else:
        WriteToPNG()
 
#File: Demo-2010-02-04.py Project: Atamai/tactics
    def __init__(self):

        #---------------------------------------------------------
        # prep the volume for rendering at 128x128x128

        self.ShiftScale = vtk.vtkImageShiftScale()
        self.ShiftScale.SetOutputScalarTypeToUnsignedShort()

        self.Reslice = vtk.vtkImageReslice()
        self.Reslice.SetInput(self.ShiftScale.GetOutput())
        self.Reslice.SetOutputExtent(0, 127, 0, 127, 0, 127)
        self.Reslice.SetInterpolationModeToCubic()

        #---------------------------------------------------------
        # set up the volume rendering

        self.Mapper = vtk.vtkVolumeRayCastMapper()
        self.Mapper.SetInput(self.Reslice.GetOutput())
        volumeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        self.Mapper.SetVolumeRayCastFunction(volumeFunction)

        self.Mapper3D = vtk.vtkVolumeTextureMapper3D()
        self.Mapper3D.SetInput(self.Reslice.GetOutput())

        self.Mapper2D = vtk.vtkVolumeTextureMapper2D()
        self.Mapper2D.SetInput(self.Reslice.GetOutput())

        self.Color = vtk.vtkColorTransferFunction()
        self.Color.AddRGBPoint(0,0.0,0.0,0.0)
        self.Color.AddRGBPoint(180,0.3,0.1,0.2)
        self.Color.AddRGBPoint(1200,1.0,0.7,0.6)
        self.Color.AddRGBPoint(2500,1.0,1.0,0.9)

        self.ScalarOpacity = vtk.vtkPiecewiseFunction()
        self.ScalarOpacity.AddPoint(0,0.0)
        self.ScalarOpacity.AddPoint(180,0.0)
        self.ScalarOpacity.AddPoint(1200,0.2)
        self.ScalarOpacity.AddPoint(2500,0.8)

        self.GradientOpacity = vtk.vtkPiecewiseFunction()
        self.GradientOpacity.AddPoint(0,0.0)
        self.GradientOpacity.AddPoint(90,0.5)
        self.GradientOpacity.AddPoint(100,1.0)

        self.Property = vtk.vtkVolumeProperty()
        self.Property.SetColor(self.Color)
        self.Property.SetScalarOpacity(self.ScalarOpacity)
        #self.Property.SetGradientOpacity(self.GradientOpacity)
        self.Property.SetInterpolationTypeToLinear()
        self.Property.ShadeOff()
        self.Property.SetAmbient(0.6)
        self.Property.SetDiffuse(0.6)
        self.Property.SetSpecular(0.1)

        self.lod2D = self.AddLOD(self.Mapper2D, self.Property, 0.01)
        self.lod3D = self.AddLOD(self.Mapper3D, self.Property, 0.1)
        self.lodRC = self.AddLOD(self.Mapper, self.Property, 1.0)
        self.SetLODLevel(self.lod2D, 2.0)
        self.SetLODLevel(self.lod3D, 1.0)
        self.SetLODLevel(self.lodRC, 0.0)

        # disable ray casting
        #self.DisableLOD(self.lod3D)
        #self.DisableLOD(self.lod2D)
        self.DisableLOD(self.lodRC)
 
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0.0)
    alphaChannelFunc.AddPoint(5, 0.1)
    alphaChannelFunc.AddPoint(64, 0.001)
    alphaChannelFunc.AddPoint(80, 0.0)
    alphaChannelFunc.AddPoint(191, 0.006)
    alphaChannelFunc.AddPoint(250, 0.1)
    alphaChannelFunc.AddPoint(255, 0.5)


volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorFunc)
volumeProperty.SetScalarOpacity(alphaChannelFunc)
 
volumeMapper = vtk.vtkVolumeRayCastMapper()
volumeMapper.SetVolumeRayCastFunction(vtk.vtkVolumeRayCastCompositeFunction())
volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
 
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)
 
renderWin = vtk.vtkRenderWindow()
renderWin.SetSize(800, 600)
if offscreen:
    print "Will render to PNG files"
    renderWin.SetOffScreenRendering(1)

mainRenderer = vtk.vtkRenderer()
renderWin.AddRenderer(mainRenderer)
mainRenderer.SetViewport(0.3,0,1,1)
 
#File: VolumeRendering.py Project: teracamo/CloudVisualizeServer
def VolumeRenderingDICOMLoader(dicomreader):
    """
    (Not used)

    :param dicomreader:
    :return:
    """

    imcast = vtk.vtkImageCast()
    imcast.SetInputConnection(dicomreader.GetOutputPort())
    imcast.SetOutputScalarTypeToUnsignedShort()
    imcast.ClampOverflowOn()

    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(-2048, 0, 0.5, 0)
    opacityTransferFunction.AddPoint(142.677, 0, 0.5, 0)
    opacityTransferFunction.AddPoint(145.016, 0.116071, 0.5, 0.26)
    opacityTransferFunction.AddPoint(192.174, 0.5625, 0.469638, 0.39)
    opacityTransferFunction.AddPoint(217.24, 0.776786, 0.666667, 0.41)
    opacityTransferFunction.AddPoint(384.347, 0.830357, 0.5, 0)
    opacityTransferFunction.AddPoint(3661, 0.830357, 0.5, 0)

    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(-2048, 0, 0, 0, 0.5, 0)
    colorTransferFunction.AddRGBPoint(142.667, 0, 0, 0, 0.5, 0)
    colorTransferFunction.AddRGBPoint(145.016, 0.615686, 0, 0.156863, 0.5, 0.26)
    colorTransferFunction.AddRGBPoint(192.174, 0.909804, 0.454902, 0, 0.469638, 0.39)
    colorTransferFunction.AddRGBPoint(217.24, 0.972549, 0.807843, 0.611765, 0.666667, 0.41)
    colorTransferFunction.AddRGBPoint(384.347, 0.909804, 0.909804, 1, 0.5, 0)
    colorTransferFunction.AddRGBPoint(3661, 1, 1, 1, 0.5, 0)
    colorTransferFunction.ClampingOn()
    colorTransferFunction.SetColorSpace(1)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetAmbient(0.2)
    volumeProperty.SetDiffuse(1)
    volumeProperty.SetSpecular(0)
    volumeProperty.SetSpecularPower(1)
    volumeProperty.DisableGradientOpacityOn()
    volumeProperty.SetComponentWeight(1, 1)
    volumeProperty.SetScalarOpacityUnitDistance(0.48117)
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.ShadeOn()
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.SetInterpolationTypeToLinear()

    raycast = vtk.vtkVolumeRayCastCompositeFunction()
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(raycast)
    volumeMapper.SetInputConnection(imcast.GetOutputPort())
    # volumeMapper.SetBlendModeToComposite()
    volumeMapper.SetSampleDistance(0.1)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    # === DEBUG TEST ===
    # renderer = vtk.vtkRenderer()
    # renderer.AddVolume(volume)
    # vdisplay = xvfbwrapper.Xvfb()
    # vdisplay.start()
    #
    # print "writing"
    # ImageWriter(renderer, out#FileName="tmp1")
    # print "write 1..."
    # camera = renderer.GetActiveCamera()
    # camera.Zoom(1.3)
    # camera.Azimuth(40)
    # ImageWriter(renderer, camera=camera, out#FileName="tmp2")
    # print "write 2..."
    # renderer.ResetCameraClippingRange()
    # vdisplay.stop()
    # === DEBUG TEST ===
    return volume
 
#File: vtkRender.py Project: caja-matematica/pyTopTools
def render( data, height ):
    """
    Assume 3D data array with integer-valued input.
    """
    # For VTK to be able to use the data, it must be stored as a
    # VTK-image. This can be done by the vtkImageImport-class which
    # imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()
    # The previously created array is converted to a string of chars and imported.
    data_string = data.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    # The type of the newly imported data is set to unsigned char (uint8)
    dataImporter.SetDataScalarTypeToUnsignedChar()
    # Because the data that is imported only contains an intensity
    # value (it isnt RGB-coded or someting similar), the importer must
    # be told this is the case.
    dataImporter.SetNumberOfScalarComponents(1)
    # The following two functions describe how the data is stored and
    # the dimensions of the array it is stored in. For this simple
    # case, all axes are of length 75 and begins with the first
    # element. For other data, this is probably not the case.  I have
    # to admit however, that I honestly dont know the difference
    # between SetDataExtent() and SetWholeExtent() although VTK
    # complains if not both are used.
    nz, ny, nx = data.shape
    dataImporter.SetDataExtent(0, nx-1, 0, ny-1, 0, nz-1)
    dataImporter.SetWholeExtent(0, nx-1, 0, ny-1, 0, nz-1)

    # The following class is used to store transparencyv-values for
    # later retrival. In our case, we want the value 0 to be completly
    # opaque whereas the three different cubes are given different
    # transperancy-values to show how it works.
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0.0)
    alphaChannelFunc.AddPoint(255, 0.0)
    # alphaChannelFunc.AddPoint(dataAvg, 0.1)
    # alphaChannelFunc.AddPoint(dataMax, 0.2)

    # This class stores color data and can create color tables from a
    # the intensity values.
    lut = vtk.vtkLookupTable()
    lut.Build()
    lutNum = data.max()
    lut.SetNumberOfTableValues(lutNum)
    ctf = vtk.vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()
    ctf.AddRGBPoint(0.0, 0, 0, 1.0)
    ctf.AddRGBPoint(data.max(), 1.0, 0, 0 )
    # Conversion to RGB tuples based on intensity values -- coarsen
    # the number of height values to only 256
    for ii,ss in enumerate([float(xx)/float(lutNum) for xx in range(lutNum)]):
        cc = ctf.GetColor( ss )
        lut.SetTableValue(ii,cc[0],cc[1],cc[2],1.0)
   
    # The preavius two classes stored properties. Because we want to
    # apply these properties to the volume we want to render, we have
    # to store them in a class that stores volume properties.
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor( ctf ) #colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)

    # This class describes how the volume is rendered (through ray tracing).
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    # We can finally create our volume. We also have to specify the
    # data for it, as well as how the data will be rendered.
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    # The class vtkVolume is used to pair the preaviusly declared
    # volume as well as the properties to be used when rendering that
    # volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    #create a plane to cut,here it cuts in the XZ direction (xz normal=(1,0,0);XY =(0,0,1),YZ =(0,1,0)
    # plane=vtk.vtkPlane()
    # plane.SetOrigin(0,100,100)
    # plane.SetNormal(1,0,0)
 
    # #create cutter
    # cutter=vtk.vtkCutter()
    # cutter.SetCutFunction(plane)
    # cutter.SetInputConnection(dataImporter.GetOutputPort())
    # cutter.Update()
    # cutterMapper=vtk.vtkPolyDataMapper()
    # cutterMapper.SetInputConnection( cutter.GetOutputPort())

    # #create plane actor
    # planeActor=vtk.vtkActor()
    # planeActor.GetProperty().SetColor(1.0,1,0)
    # planeActor.GetProperty().SetLineWidth(2)
    # planeActor.SetMapper(cutterMapper)

    #create cube actor
    # cubeActor=vtk.vtkActor()
    # cubeActor.GetProperty().SetColor(0.5,1,0.5)
    # cubeActor.GetProperty().SetOpacity(0.5)
    # cubeActor.SetMapper(cubeMapper)

    # With almost everything else ready, its time to initialize the
    # renderer and window, as well as creating a method for exiting
    # the application
    renderer = vtk.vtkRenderer()
    #renderer.AddActor( planeActor )
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
 
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)

    # We add the volume to the renderer ...
    renderer.AddVolume(volume)
    # ... set background color to white ...
    renderer.SetBackground(0,0,0)
    # ... and set window size.
    renderWin.SetSize(600, 600)

    # add a scalar color bar
    sb = vtk.vtkScalarBarActor() 
    sb.SetTitle("Elevation") 
    # If the orientation is vertical there is a problem. 
    sb.SetOrientationToHorizontal() 
    # Vertical is OK. 
    # sb.SetOrientationToVertical() 
    sb.SetWidth(0.6) 
    sb.SetHeight(0.17) 
    sb.SetPosition(0.1, 0.05) 
    sb.SetLookupTable(ctf) 

    sbw = vtk.vtkScalarBarWidget() 
    sbw.SetInteractor(renderInteractor) 
    sbw.SetScalarBarActor(sb) 
    sbw.On() 

    # A simple function to be called when the user decides to quit the application.
    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)

    # Tell the application to use the function as an exit check.
    renderWin.AddObserver("AbortCheckEvent", exitCheck)

    renderInteractor.Initialize()
    # Because nothing will be rendered without any input, we order the
    # first render manually before control is handed over to the
    # main-loop.
    renderWin.Render()
    renderInteractor.Start()

    # write a PNG image to disk
    writer = vtk.vtkPNGWriter()
    writer.Set#FileName("rbc_stackVTK_height_"+str(height)+".png")
    writer.SetInput(dataImporter.GetOutput())
    writer.Write()
 
#File: VolumeRendering.py Project: teracamo/CloudVisualizeServer
def VolumeRenderingRayCast(inVolume, scale=[1, 1, 1], lowerThreshold=0, upperThreshold=None):
    """
    Recieve a numpy volume and render it with RayCast method. This method employs CPU raycast
    and will subject to upgrades of using GPUVolumeMapper in the future. The method returns
    a vtkVolume actor which can be added to a vtkRenderer

    :param inVolume:        numpy volume
    :param scale:           scale [x, y, z] of the slice/voxel spacing to real spacing
    :param lowerThreshold:  lower Threshold for raycast. Default = 0
    :param upperThreshold:  upper Threshold for raycast. Default = inVolume.max()
    :return: vtk.vtkVolume
    """
    inVolume = np.ushort(inVolume)
    inVolumeString = inVolume.tostring()

    # Color map related
    if upperThreshold == None:
        upperThreshold = inVolume.max()

    if upperThreshold <= lowerThreshold:
        raise ValueError("Upper Threshold must be larger than lower Threshold.")

    centerThreshold = (upperThreshold - lowerThreshold) / 2.0 + lowerThreshold
    lowerQuardThreshold = (centerThreshold - lowerThreshold) / 2.0 + lowerThreshold
    upperQuardThreshold = (upperThreshold - centerThreshold) / 2.0 + centerThreshold

    dataImporter = vtk.vtkImageImport()
    dataImporter.CopyImportVoidPointer(inVolumeString, len(inVolumeString))
    dataImporter.SetDataScalarTypeToUnsignedShort()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetDataExtent(0, inVolume.shape[2] - 1, 0, inVolume.shape[1] - 1, 0, inVolume.shape[0] - 1)
    dataImporter.SetWholeExtent(0, inVolume.shape[2] - 1, 0, inVolume.shape[1] - 1, 0, inVolume.shape[0] - 1)

    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(lowerThreshold, 0)
    alphaChannelFunc.AddPoint(lowerQuardThreshold, 0.05)
    alphaChannelFunc.AddPoint(centerThreshold, 0.4)
    alphaChannelFunc.AddPoint(upperQuardThreshold, 0.05)
    alphaChannelFunc.AddPoint(upperThreshold, 0)

    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(lowerThreshold, 0, 0, 0)
    colorFunc.AddRGBPoint(centerThreshold, 0.5, 0.5, 0.5)
    colorFunc.AddRGBPoint(upperThreshold, 0.8, 0.8, 0.8)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)

    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    volume.SetScale(scale)

    # Volume is returned for further rendering
    return volume
 
#File: volRCClipPlanes.py Project: timkrentz/SunTracker
colorTransferFunction.AddRGBSegment(50,1,1,1,150,0,0,0)
colorTransferFunction.AddRGBSegment(60,1,1,1,90,0,0,0)
colorTransferFunction.AddHSVSegment(90,1,1,1,105,0,0,0)
colorTransferFunction.AddHSVSegment(40,1,1,1,155,0,0,0)
colorTransferFunction.AddHSVSegment(30,1,1,1,95,0,0,0)
colorTransferFunction.RemoveAllPoints()
colorTransferFunction.AddHSVPoint(0.0,0.01,1.0,1.0)
colorTransferFunction.AddHSVPoint(127.5,0.50,1.0,1.0)
colorTransferFunction.AddHSVPoint(255.0,0.99,1.0,1.0)
colorTransferFunction.SetColorSpaceToHSV()
# Create properties, mappers, volume actors, and ray cast function
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorTransferFunction)
volumeProperty.SetScalarOpacity(opacityTransferFunction)
volumeProperty.SetInterpolationTypeToLinear()
compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
volumeMapper = vtk.vtkVolumeRayCastMapper()
volumeMapper.SetInputConnection(reader.GetOutputPort())
volumeMapper.SetVolumeRayCastFunction(compositeFunction)
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)
# Create geometric sphere
sphereSource = vtk.vtkSphereSource()
sphereSource.SetCenter(25,25,25)
sphereSource.SetRadius(30)
sphereSource.SetThetaResolution(15)
sphereSource.SetPhiResolution(15)
sphereMapper = vtk.vtkPolyDataMapper()
sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
sphereActor = vtk.vtkActor()
 
#File: frog.py Project: HighExecutor/vtkFrogAtlas
frogReader.Set#FilePrefix("./WholeFrog/frog.")
frogReader.Set#FilePattern("%s%03d.raw")
frogReader.Update()

# Frog density
frogDens = vtk.vtkPiecewiseFunction()
frogDens.AddPoint(0.0, 0.0)
frogDens.AddPoint(230.0, 0.002)

#Frog color
frogClr = vtk.vtkColorTransferFunction()
frogClr.AddRGBPoint(0.0, 0.1, 0.9, 0.1)
frogClr.AddRGBPoint(250.0, 0.96, 0.98, 0.98)

# Frog mapper
frogFunc = vtk.vtkVolumeRayCastCompositeFunction()
frogMapper = vtk.vtkVolumeRayCastMapper()
frogMapper.SetInputConnection(frogReader.GetOutputPort())
frogMapper.SetVolumeRayCastFunction(frogFunc)

# Frog property
frogVolProp = vtk.vtkVolumeProperty()
frogVolProp.SetColor(frogClr)
frogVolProp.SetScalarOpacity(frogDens)
frogVolProp.SetInterpolationTypeToLinear()
frogVolProp.ShadeOn()

#Frog actor
frog = vtk.vtkVolume()
frog.SetMapper(frogMapper)
frog.SetProperty(frogVolProp)
