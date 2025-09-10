from pycocotools import mask as maskUtils
import numpy as np
import pathlib
import base64

class bundle():
    def __init__(self, nucleus_path: pathlib.Path, cyto_paths: list[pathlib.Path], bbox: list[list], segment: list[np.ndarray]) -> None:
        self.nucleus_path = nucleus_path
        self.cyto_paths = cyto_paths  
        self.bbox = np.array(bbox, dtype=np.uint16)
        self.rleSeg: list[dict] = []
        for i in range(len(segment)):
            d = maskUtils.encode(np.asfortranarray(segment[i]))
            d["counts"] = base64.b64encode(d['counts']).decode('utf-8')
            self.rleSeg.append(d)

    def L(self) -> tuple[pathlib.Path, list[pathlib.Path], list[list], list[np.ndarray]]:
        segment_r = []
        for i in range(len(self.rleSeg)):
            self.rleSeg[i]['counts'] = base64.b64decode(self.rleSeg[i]['counts'].encode('utf-8'))
            segment_r.append(maskUtils.decode(self.rleSeg[i]))
        return self.nucleus_path, self.cyto_paths, self.bbox.tolist(), segment_r
