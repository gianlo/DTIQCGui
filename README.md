DTIQCGui
========

A pyQT4 Gui for DTI Quality Control for FSL format data.

Installation:

required packages:
numpy
PyQt4

and the following libraries hosted here:
https://github.com/gianlo/pynii

Make sure the required packages are installed in your python distribution.
To install the required packages either use easy_install or pip.
Then install the libraries by getting them from the repositories listed and following the installation instructions.

No further installation is required.

To excute the Gui open a terminal, change directory to the src folder and then run the following command:

python dti_qc_gui.pyw


This gui is used to visually inspect DTI data and eventually remove artifacted images. It opens FSL formatted dti data (i.e. 4d nifti file + bvals and bvecs text files).

