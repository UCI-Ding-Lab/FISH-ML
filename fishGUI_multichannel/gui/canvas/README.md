# gui/canvas Directory Overview

This folder contains the core classes for interactive drawing and annotation on the image canvas in the FISH-ML GUI.  
Each file defines a specific component used for bounding boxes, segmentation masks, and anchor points.

**Summary:**

- `stove.py`: The main canvas and event handler.
- `box.py`: Bounding box logic and drawing.
- `segment.py`: Segmentation mask logic and drawing.
- `anchor.py`: Draggable points for resizing boxes.

These classes work together to provide interactive annotation tools in the FISH-ML

---

## **stove.py**

**Overview:**  
Manages the main drawing canvas where images are displayed and annotated.

**Responsibilities:**

- Integrates matplotlib with Tkinter for interactive image display.
- Handles user interactions (mouse clicks, drags, releases) for drawing and editing.
- Coordinates the display and updates of boxes, segments, and anchors.
- Provides methods for adjusting contrast/brightness and clearing the canvas.

**How it connects:**  
Acts as the central hub for all drawing actions. Other classes (box, segment, anchor) are drawn on this canvas.

---

## **box.py**

**Overview:**  
Defines the bounding box class for marking regions of interest on the image.

**Responsibilities:**

- Draws and manages a rectangular bounding box.
- Handles selection, resizing, and moving of the box.
- Manages anchor points for resizing.
- Updates its appearance based on user actions.

**How it connects:**  
Each box is drawn on the stove canvas and uses anchor objects for resizing.  
Communicates with the GUI to update its state and appearance.

---

## **segment.py**

**Overview:**  
Defines the segmentation mask class for marking detailed regions on the image.

**Responsibilities:**

- Draws and manages a segmentation mask (free-form region).
- Handles selection, editing (brush/eraser), and deletion of masks.
- Updates its visual patch based on user input.

**How it connects:**  
Each segment is drawn on the stove canvas.  
Works with the GUI’s tool panel for brush and eraser actions.

---

## **anchor.py**

**Overview:**  
Defines anchor points used for resizing and moving bounding boxes.

**Responsibilities:**

- Draws draggable points at the corners and edges of a box.
- Handles selection and color changes when active.
- Supports moving and resizing the bounding box via user interaction.

**How it connects:**  
Anchors are created and managed by each box.  
They are drawn on the stove canvas and respond to mouse events for resizing boxes.
