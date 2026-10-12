"""Register PlateAOVs in the Medratools toolbar."""

import nuke

import plateaovs_nuke

plateaovs_nuke.add_layers()

_toolbar = nuke.menu("Nodes")
_tools = _toolbar.findItem("Medratools") or _toolbar.addMenu("Medratools", icon="medratools.png")
_tools.addCommand("PlateAOVs", "import plateaovs_nuke; plateaovs_nuke.create()", icon="medratools.png")
