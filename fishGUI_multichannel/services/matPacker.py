import pathlib
import numpy as np
import scipy.io

def read(mat: dict, img_number: int, target: str, cell: int, title: str):
    if target == "filename": return mat["Tracked"][0, img_number][target][0,0]
    else: return mat["Tracked"][0, img_number][target][0,0][0, cell][title][0,0]

def create(name: list[str], xy: list[list[(float,float),],], masks: list[list[np.ndarray,],], saveFile: pathlib.Path):
    tracked_dtype = np.dtype([
        ("filename", "O"),
        ("cells", "O")
    ])

    cell_dtype = np.dtype([
        ('mask', 'O'),
        ('pos', 'O'),
        ('size', 'O'),
        ('progenitor', 'O'),
        ('descendants', 'O'),
        ('Fmask', 'O'),
        ('area', 'O'),
        ('Acom', 'O'),
        ('Fpixels', 'O'),
        ('Ftotal', 'O'),
        ('Fmax', 'O'),
        ('Fmean', 'O')
    ])

    imgCount = len(name)
    tracked = np.empty((1, imgCount), dtype="O")

    for eachImg in range(imgCount):
        cellCount = len(xy[eachImg])
        cells = np.empty((1, cellCount), dtype="O")
        for eachCell in range(cellCount):
            cell_data = np.zeros((1, 1), dtype=cell_dtype)
            cell_data[0, 0]["mask"] = masks[eachImg][eachCell].astype(np.double)
            cell_data[0, 0]["pos"] = np.array(xy[eachImg][eachCell]).T # flipped to match matlab's (x,y) order
            cell_data[0, 0]["size"] = np.array([masks[eachImg][eachCell].shape[0], masks[eachImg][eachCell].shape[1]])
            cell_data[0, 0]["area"] = np.array([np.sum(masks[eachImg][eachCell])])

            # Those are all meant to be 'None' in python, but matlab needs them to be empty arrays
            cell_data[0, 0]["progenitor"] = np.array([], dtype='O')  # Empty array for objects
            cell_data[0, 0]["descendants"] = np.array([], dtype='O')
            cell_data[0, 0]["Fmask"] = np.zeros((0, 0))  # Empty mask
            cell_data[0, 0]["Acom"] = np.array([np.nan])  # NaN for missing numerical values
            cell_data[0, 0]["Fpixels"] = np.array([], dtype='O')
            cell_data[0, 0]["Ftotal"] = np.array([0])
            cell_data[0, 0]["Fmax"] = np.array([0])
            cell_data[0, 0]["Fmean"] = np.array([0])
            
            cells[0, eachCell] = cell_data

        tracked[0, eachImg] = np.zeros((1, 1), dtype=tracked_dtype)
        tracked[0, eachImg][0, 0]["filename"] = np.array([[name[eachImg]]], dtype="O")
        tracked[0, eachImg][0, 0]["cells"] = cells
    
    scipy.io.savemat(saveFile, {"Tracked": tracked})