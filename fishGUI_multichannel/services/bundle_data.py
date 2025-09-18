from pycocotools import mask as maskUtils
import numpy as np
import pathlib
import base64

class bundle():
    """
    Pack generated bbox and segmentation masks for cytoplasms so that it can be saved
    and loaded later. It is a data container representing data for
    single frame.
    """
    def __init__(self, sample_id, nucleus_path: pathlib.Path, cyto_paths: list[pathlib.Path], bbox: list[list], segment: list[np.ndarray]) -> None:
        self.sample_id = sample_id 
        self.nucleus_path = nucleus_path
        self.cyto_paths = cyto_paths  
        self.bbox = np.array(bbox, dtype=np.uint16)
        self.rleSeg: dict[str, list[dict]] = {}
        for ch in segment:
            self.rleSeg[ch] = []
            for mask in segment[ch]:
                d = maskUtils.encode(np.asfortranarray(mask))
                d["counts"] = base64.b64encode(d['counts']).decode('utf-8')
                self.rleSeg[ch].append(d)

    def extract_data_from_bundles(self) -> tuple[pathlib.Path, list[pathlib.Path], list[list], list[np.ndarray]]:
        segment_r = {}
        for ch in self.rleSeg:
            segment_r[ch] = []
            for d in self.rleSeg[ch]:
                d = d.copy()
                d['counts'] = base64.b64decode(d['counts'].encode('utf-8'))
                segment_r[ch].append(maskUtils.decode(d))
        return self.sample_id, self.nucleus_path, self.cyto_paths, self.bbox.tolist(), segment_r