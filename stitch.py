import numpy as np 
import sys
import json
import h5py
from shutil import copyfile


# read meta data
with open(sys.argv[1]) as f:
    meta = json.loads(f.read())

pixel_size = meta['pixel_size'] 
output_name = meta['output_name'] 
del meta['pixel_size']
del meta['output_name']

# take subregion ids
idsy = [int(k[0]) for k in meta.keys()]
idsx = [int(k[2]) for k in meta.keys()]
ny = max(idsy)+1
nx = max(idsx)+1

# read subregion size
with h5py.File(list(meta.values())[0],'r') as f:
    [ntheta,sizey,sizex] = f['/exchange/data'].shape
    ndark = f['/exchange/data_dark'].shape[0]
    nwhite = f['/exchange/data_white'].shape[0]
    dtype = f['/exchange/data'].dtype

# take region positions
posy = np.zeros([len(idsy)])
posx = np.zeros([len(idsx)])
for (k,name) in enumerate(meta.values()):
    with h5py.File(name,'r') as f:
        posy[k] = f['/measurement/instrument/sample/setup/sample_y'][0]
        posx[k] = f['/measurement/instrument/sample/setup/sample_x'][0]
mposy = min(posy)
mposx = min(posx)

# copy the first file structure
copyfile(list(meta.values())[0], output_name)

# update data in the output hdf5 file
with h5py.File(output_name,'a') as fout:
    del fout['/exchange/data']
    del fout['/exchange/data_white']
    del fout['/exchange/data_dark']
    datanew = fout.create_dataset('/exchange/data', (ntheta,ny*sizey, nx*sizex),chunks=(1,ny*sizey, nx*sizex))
    whitenew = fout.create_dataset('/exchange/data_white', (nwhite,ny*sizey, nx*sizex),chunks=(1,ny*sizey, nx*sizex))
    darknew = fout.create_dataset('/exchange/data_dark', (ndark,ny*sizey, nx*sizex),chunks=(1,ny*sizey, nx*sizex))
    for (k,name) in enumerate(meta.values()):
        print(f'processing {name}, {posy[k]=}, {posx[k]=}')
        with h5py.File(name,'r') as f:   
            sty = int((posy[k]-mposy)/pixel_size*1e3+0.5)
            stx = int((posx[k]-mposx)/pixel_size*1e3+0.5)
            datanew[:,sty:sty+sizey,stx:stx+sizex] = f['/exchange/data'][:]
            whitenew[:,sty:sty+sizey,stx:stx+sizex] = f['/exchange/data_white'][:]
            darknew[:,sty:sty+sizey,stx:stx+sizex] = f['/exchange/data_dark'][:]
