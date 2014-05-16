'''
Created on 10 Apr 2013

@author: gfagiolo
'''

from string import Template
import os

import numpy as np

import pynii
# import nibabel as nb

from pyDTI.pyDTI import FSLDTI

def save_FSL_script(niiname, bvalsname, bvecsname):
    script_t = Template("""#!/bin/sh

#check if FSL is properly installed

if [ -z "$$FSLDIR" ]
then
   echo FSL is not properly installed/configured. Please set the FSLDIR environment variable
    exit 1
fi

echo processing file: $infile

#do brain extraction

bet $infile $brainfile -m -R
echo brain extraction completed: $brainfile, $brainmaskfile

#do eddy current correction

eddy_correct $infile $eddycorrectfile 0
echo eddy current correction completed: $eddycorrectfile

#do tensor model fit

dtifit --data=$eddycorrectfile --out=$outname --mask=$brainmaskfile --bvecs=$bvecs --bvals=$bvals
echo dti fit completed: ${outname}_\(FA, MD, Ln, Vn\)

#view results with fslview

fslview ${outname}_FA ${outname}_V1
""")
    script_name = niiname + '.fsl_dtifit.sh'
    fname = os.path.basename(niiname)
    bname = fname.replace('.nii','.brain.nii')
    subs = {'infile':fname, 'brainfile':bname,
            'brainmaskfile':bname.replace('.nii','_mask.nii'),
            'eddycorrectfile':fname.replace('.nii','.ec.nii'),
            'outname':fname + '.dtifit',
            'bvals':os.path.basename(bvalsname),
            'bvecs':os.path.basename(bvecsname),            
            }
    ofile = open(script_name, 'wb')
    ofile.write(script_t.substitute(subs))
    ofile.close()
    return script_name

class ProcessDTI(FSLDTI):
    '''
    classdocs
    '''
    
    EXCLUDED_DIRS = None
    
    def __init__(self, filename, bvecs=None, bvals=None, nboots=None, maskname=None, changenonpositivevalues=False, verbose=None):
        super(ProcessDTI, self).__init__(filename, bvecs, bvals, nboots, maskname, changenonpositivevalues, verbose)
        self.EXCLUDED_DIRS = set()

    def addExcluded(self, dirn):
        self.EXCLUDED_DIRS.add(dirn)
    
    def removeExcluded(self, dirn):
        self.EXCLUDED_DIRS.remove(dirn)
        
    def clearExcluded(self):
        self.EXCLUDED_DIRS = set()

    def saveData(self, suffix='.exc'):
        log_template = Template("""orig_nifti = $niiin
new_nifti = $niiout
orig_bvals = $bvalsin
new_bvals = $bvalsout
orig_bvecs = $bvecsin
new_bvecs = $bvecsout
fsl_processing_script = $fslscript
imageno_excluded = $excluded
""")
        if self.EXCLUDED_DIRS:
            #save only not EXCLUDED_DIRS data
            save_only_dirs = sorted(list(set(range(self.getSignalLength())).difference(self.EXCLUDED_DIRS)))
#             nii = nb.Nifti1Image(self.getImageData()[:, :, :, save_only_dirs], self.getNiftiAffine(), self.getNiftiHeader())
            nii = pynii.Nifti1Data()
            nii.setData(self.getImageData()[:, :, :, save_only_dirs])
            nii.setAffine(self.getNiftiAffine())
            outname = self.getNiftiFilename().replace('.nii', suffix + '.nii')
#             nb.save(nii, outname)
            nii.write(outname)
            bvalname = self.getBValuesFilename() + suffix
            bvecsname = self.getBVectorsFilename() +  suffix
            bvals = self.getBValues()[save_only_dirs]
            bvecs = self.getBVectors()[:, save_only_dirs]
            np.savetxt(bvalname, np.array(bvals).reshape(len(bvals), 1).T, fmt='%g')
            np.savetxt(bvecsname, np.array(bvecs), fmt='%g')
            script_name = save_FSL_script(outname, bvalname, bvecsname)
            logfilename = self.getNiftiFilename() + '.exc.log'
            with open(logfilename, 'wb') as of:
                of.write(log_template.substitute(niiin=self.getNiftiFilename(), 
                                              niiout=outname,
                                              bvalsin=self.getBValuesFilename(),
                                              bvalsout=bvalname,
                                              bvecsin=self.getBVectorsFilename(),
                                              bvecsout=bvecsname,
                                              fslscript=script_name,
                                              excluded=str(sorted(self.EXCLUDED_DIRS))))
            return (outname, bvalname, bvecsname, logfilename)
        else:
            #save data as is
            print 'Nothing to do!'
        return tuple()

def test(fname):
    dti = ProcessDTI(fname, fname+'.bvecs', fname+'.bvals')
    dti.addExcluded(10)
    dti.addExcluded(15)
    dti.saveData()


