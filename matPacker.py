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
            cell_data[0, 0]["mask"] = masks[eachImg][eachCell]
            cell_data[0, 0]["pos"] = np.array(xy[eachImg][eachCell])
            cell_data[0, 0]["size"] = np.array([np.prod(masks[eachImg][eachCell].shape)])
            cell_data[0, 0]["area"] = np.array([np.sum(masks[eachImg][eachCell])])

            cell_data[0, 0]["progenitor"] = None
            cell_data[0, 0]["descendants"] = None
            cell_data[0, 0]["Fmask"] = None
            cell_data[0, 0]["Acom"] = None
            cell_data[0, 0]["Fpixels"] = None
            cell_data[0, 0]["Ftotal"] = None
            cell_data[0, 0]["Fmax"] = None
            cell_data[0, 0]["Fmean"] = None
            
            cells[0, eachCell] = cell_data

        tracked[0, eachImg] = np.zeros((1, 1), dtype=tracked_dtype)
        tracked[0, eachImg][0, 0]["filename"] = np.array([name[eachImg]])
        tracked[0, eachImg][0, 0]["cells"] = cells
    
    scipy.io.savemat(saveFile, {"Tracked": tracked})