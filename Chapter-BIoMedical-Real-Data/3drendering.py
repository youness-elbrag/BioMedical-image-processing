import os
from vedo import dataurl, Volume, Text2D
from ipyvtk_simple.viewer import ViewInteractiveWidget
from vedo.applications import Slicer3DPlotter
from vedo import Volume, dataurl
from vedo.applications import RayCastPlotter


"""Use sliders to slice volume
Click button to change colormap"""
#vedo.start_xvfb()
path_save = '03-Data-Formats/'

filename = path_save + 'SE1mesh3d.tif'
# filename = dataurl+'embryo.tif'
# filename = dataurl+'vase.vti'

vol = Volume(filename)#.print()

plt = Slicer3DPlotter(
    vol,
    bg="white",
    bg2="lightblue",
    cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
    use_slider3d=True,
)

# plt = RayCastPlotter(vol, bg='black', bg2='blackboard', axes=7)  # Plotter instance

# plt.show(viewup="z").close()

# Can now add any other object to the Plotter scene:
# plt += Text2D('some message')
ViewInteractiveWidget(plt.show())