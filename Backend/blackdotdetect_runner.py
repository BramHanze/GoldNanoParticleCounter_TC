from blackdotdetector import BlackDotDetector
import glob

"""
Runs blackdotdetector on a given image, or all images in given folder.
"""

directory = 'data/Complemented/'
directory = 'data/Complemented/2024-08i compl OADChi E2_24.tif'

if directory.endswith('/'):
    directory += '*'
files = glob.glob(directory, recursive=False)

for file in files:
    BlackDotDetector(file).run()
