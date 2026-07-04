# TouchDesigner Integration Guide

This directory contains instructions and helper scripts to integrate the Multi-Layer Finger-Controlled Reality Filters with Derivative TouchDesigner.

The python tracker script runs MediaPipe, calculates the panels and gestures, and broadcasts coordinates over OSC (UDP protocol) to TouchDesigner. TouchDesigner receives these coordinates and handles rendering on the GPU.

---

## 1. OSC Address Protocol Reference

The Python script broadcasts to `127.0.0.1` on port `9000` (UDP) by default. The following OSC messages are sent:

| OSC Address | Data Type | Description |
| :--- | :--- | :--- |
| `/hand/left/present` | `int` | `1` if the left hand is tracked, `0` otherwise |
| `/hand/right/present` | `int` | `1` if the right hand is tracked, `0` otherwise |
| `/hand/left/gesture` | `string` | Current gesture detected: `"None"`, `"Pinch"`, `"Fist"`, `"OpenPalm"` |
| `/hand/right/gesture` | `string` | Current gesture detected: `"None"`, `"Pinch"`, `"Fist"`, `"OpenPalm"` |
| `/hand/left/fingertips`| `float[15]` | Array of 15 floats (x, y, z for Thumb, Index, Middle, Ring, Pinky tips) |
| `/hand/right/fingertips`| `float[15]` | Array of 15 floats (x, y, z for Thumb, Index, Middle, Ring, Pinky tips) |
| `/panels/count` | `int` | The number of active filter panels (accordion bands + spawned panels) |
| `/panels/<id>/corners` | `float[8]` | Flat array of 8 floats representing `[x0, y0, x1, y1, x2, y2, x3, y3]` |
| `/panels/<id>/effect` | `string` | Filter shader name: `"ThermalVision"`, `"VHSGlitch"`, `"HalftoneDot"`, `"GlassRefraction"` |
| `/panels/<id>/alpha` | `float` | Panel fade multiplier `[0.0, 1.0]` (used for occlusion transitions) |
| `/system/freeze` | `int` | `1` if a fist is active (freeze visual frame), `0` otherwise |

*Note: `<id>` is `band_1` through `band_4` for the fingers-accordion panels, and `spawned_1`, `spawned_2`, etc., for the user-spawned floating panels.*

---

## 2. TouchDesigner Network Setup

Follow these steps to build the TouchDesigner project:

### Step A: Capture the Camera & Setup Freeze Frame
1. Add a **Video Device In TOP** to capture the live webcam stream.
2. Add a **Cache TOP** connected to the camera feed.
3. Add a **Switch TOP** with two inputs:
   - Input 0: The live **Video Device In TOP**.
   - Input 1: The **Cache TOP** output.
4. Drive the Switch TOP index parameter using the `/system/freeze` OSC channel (see below). When `/system/freeze == 1`, switch to the cache to freeze the image inside the filters!

### Step B: Receive OSC Tracking Data
1. Add an **OSC In CHOP**. Set the `Network Port` to `9000`.
2. This will generate channels like:
   - `hand_left_present`
   - `hand_right_present`
   - `system_freeze`
   - Individual coordinate channels.

### Step C: Handle Dynamic Panels using Python
To handle the dynamic corners and rendering of panels (both the 4 finger-bands and user-spawned floating panels), you can use an **OSC In DAT** to parse the panel addresses and map them.

1. Add an **OSC In DAT** and set the port to `9000`.
2. Set the callback script of the OSC In DAT to run [osc_handler.py](file:///e:/Projects101/FilterTrack/src/touchdesigner/osc_handler.py).
3. The script will write incoming panel corners, alpha, and effect categories to a set of **Table DATs** or **Constant CHOPs**.

### Step D: Render the Panels with GLSL and Corner Pin
For each panel (e.g., Band 1 through Band 4, and any spawned quads):

1. **Effects Shader**:
   - Create a **GLSL TOP**.
   - Connect the Switch TOP (from Step A) as Input 0.
   - Load the relevant fragment shader (e.g., `ThermalVision.frag`, `VHSGlitch.frag`, `HalftoneDot.frag`) into the GLSL TOP's Pixel Shader page.
   - Set the GLSL TOP parameters:
     - GLSL Version: `3.30` or higher.
     - Sample Uniform `uTexture` mapped to input 0.
     - Uniforms for parameters like `uTime` (linked to `absTime.seconds`).

2. **Corner Pinning**:
   - Connect the output of the GLSL TOP to a **Corner Pin TOP**.
   - Link the Corner Pin TOP corner parameters (`Bottom Left`, `Bottom Right`, `Top Right`, `Top Left`) to the corresponding OSC channels for that panel ID.
   - Set the alignment coordinates to map from normalized coordinates `[0, 1]` to the TouchDesigner width and height.

3. **Composite Background**:
   - Add a **Composite TOP** (operation: `Over`).
   - Add the live background webcam feed as the bottom layer.
   - Layer all active Corner Pin TOP outputs on top of it.
   - Add an **Out TOP** to output the final render display!
