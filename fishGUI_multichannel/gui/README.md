# fishGUI_multichannel/gui Directory Overview

This directory contains all **graphical user interface components** for the FISH-ML application for multichannel segmentation.

---

## abstract.py

**Core data model for an image/frame**  
Loads and preprocesses image data. Manages bounding boxes, segmentation masks, and thumbnail states. Provides methods for selection, segmentation, and exporting frame data.

---

## frames.py

**Frame layout management**  
Defines subframes for organizing buttons, thumbnails, and canvas. Provides methods to pack/unpack frames and access them.

---

## buttons.py

**Button controls and logic**  
Handles import, selection, bounding box, segmentation, mask application, and export actions. Manages button states and triggers GUI updates.

---

## toolbar.py

**Custom matplotlib toolbar.**  
Extends NavigationToolbar2Tk, adds methods to reset tool states and handle toolbar events.

---

## thumbnails.py

**Image gallery management.**  
Handles scrolling, thumbnail display, and grouping by sample/channel. Adds images to the gallery and updates their visual state.

---

## tools_panel.py

**Tool panel for segmentation and image adjustments.**  
Provides brush, eraser, and bounding box tools. Includes controls for marker size, contrast, brightness, and channel selection. Handles save/load progress and mask application.

---

## canvas/

### stove.py

**Main drawing canvas and image display.**  
Integrates matplotlib with Tkinter. Handles image rendering, patch management, and mouse events for interaction.

### box.py

**Bounding box annotation class.**  
Handles drawing, selection, resizing, and anchor management for bounding boxes.

### segment.py

**Segmentation mask class.**  
Handles mask drawing, selection, updating, and deletion. Manages mask patches and buffer for current selection.

### anchor.py

**Draggable anchor points for bounding boxes.**  
Handles anchor drawing, selection, and color changes. Supports resizing and moving bounding boxes via anchors.

---

## utils.py

**Shared utility functions for the GUI.**  
Provides helper functions used across multiple modules to keep code DRY and organized.

---

## How to Use

- Start the GUI via `main.py` or the designated entry point.
- Use the toolbar, buttons, and tool panel to interact with images and annotations.
- Refer to each module for specific logic and customization.
