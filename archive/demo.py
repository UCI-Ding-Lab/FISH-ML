import numpy as np

def compute_area(rect):
    x1, y1, x2, y2 = rect
    return (x2 - x1) * (y2 - y1)

def compute_intersection_area(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    
    if xi1 < xi2 and yi1 < yi2:
        return (xi2 - xi1) * (yi2 - yi1)
    else:
        return 0

def filter_rects(rects):
    N = len(rects)
    to_delete = set()
    
    areas = np.array([compute_area(rect) for rect in rects])
    
    for i in range(N):
        for j in range(i + 1, N):
            if j in to_delete:
                continue
                
            intersection_area = compute_intersection_area(rects[i], rects[j])
            
            if intersection_area >= 0.9 * min(areas[i], areas[j]):
                if areas[i] > areas[j]:
                    to_delete.add(i)
                else:
                    to_delete.add(j)
    
    filtered_rects = [rect for k, rect in enumerate(rects) if k not in to_delete]
    return np.array(filtered_rects)
