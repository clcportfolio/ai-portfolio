# Stereogram Converter — What It Does

## What This Tool Does

This tool turns a special kind of image — called a depth map — into a Magic Eye picture.
A depth map is a greyscale image where bright areas represent things that are close to you
and dark areas represent things that are far away. The tool reads that depth information
and uses it to create a repeating pattern that, when you relax your focus, reveals a
three-dimensional shape floating in space.

## Why It Matters

Magic Eye images (stereograms) are a classic example of how our brains process depth from
patterns — the same underlying principle used in 3D movies, VR headsets, and medical
imaging. This tool makes it easy to create them from any depth image, without needing
special software or artistic skill.

## What You See When You Run It

You open the app in a web browser. On the left side, you upload your depth map image —
this could be a render of a 3D model, a depth photo from a phone camera, or any greyscale
image where brightness represents depth. Optionally, you can upload a texture image that
will be used as the repeating pattern; if you skip this, the tool picks one automatically.

On the right side, two sliders let you control how the stereogram looks: one adjusts how
far apart the "virtual eyes" are (which affects how easy the image is to view), and the
other controls how strong the 3D effect appears.

Click **Generate Stereogram** and within a few seconds the result appears below. You can
view it directly in the browser and download it as a PNG file with one click.

## Who Built This and How

This tool was built by Cody Culver as part of an AI engineering portfolio. It uses only
standard image-processing libraries — no AI is involved in this version. The same
stereogram engine is used as a building block in a more advanced version of the tool
that accepts a text description and generates the 3D shape automatically using AI.
