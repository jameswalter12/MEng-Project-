# Notes

- [ ] January 14th 2026 - So there is a line of code in Shehan's main fail that directs the script to PRUSA slicer files that are downloaded within the src folder, however the code agent in VSCODE has clocked that these should be updated to be MacOS files, ensure this is changed

- [ ] So I need to look into what the serial ports are, there is functionality in the code that should let you connect to the 3D printer by the serial? Not sure how this works, if I have to do anything with the printer/code, if anything has to be initialised or sorted out - can I just put my laptop next to the printer?? Does anything have to actually be done here??
- [ ] With respect to the above item, it seems like I just need a usb connection with my laptop to the printer (so usb to usb-c) which I have? Well, I need usb to usb from printer to my dock, and then the dock to usb-c for my laptop...
- [ ] Currently, I think that all of the self.monitor_x_var and self.compare_mesh_var is currently not doing anything and is just defined and bound to the checkbox, self.compare_layer_img_var does however do something.

    * Now the above is an interesting one, the compare layer, is a function that allowes the comparison of  of a printed layer against the expected layer fromt the sliced model. So it takes a photo, saves it, then loads the model's layer passess both images to a layer comparison and computes a correction G-Code. Now if this works that is brilliant, however, surely this is where my Photogrammetry should come in, I really doubt this is done here... 
   * Okay so this is fairly big, see next section...
     
## Summary: What My Photogrammetry Scripts Do vs What Shehan’s CompareLayer Does

### My Photogrammetry / Computer Vision Scripts (Localisation + ROI Extraction + (optional) Matching)
My scripts focus on making messy, real-world camera images usable by reliably finding the printed part in-frame.

They currently do:
- **Localisation in cluttered scenes**: detect the target object within a busy full-frame image (e.g. printer bed + surroundings).
- **Colour / structure-based masking**: use HSV colour thresholding (and/or structure masks like Otsu/adaptive/edges) to isolate candidate regions.
- **Candidate ranking / filtering**: score contours by square-likeness, solidity, extent, area, and optionally closeness to expected location/size.
- **ROI extraction**: output a **cropped region of interest (ROI)** containing the printed object (and optionally processed versions like grayscale/CLAHE).
- **Optional feature detection / matching**: corner detection and ORB keypoints for potential alignment or tracking (not yet used as a final comparison stage).

**Main output:** a reliable ROI (and associated metadata) that can be passed to a downstream comparison algorithm.

---

### Shehan’s CompareLayer (Alignment + Segmentation + Differencing + Optional Correction G-code)
Shehan’s CompareLayer is designed to compare “expected vs actual” at the layer level and (optionally) generate a corrective toolpath.

It currently does:
- **Marker-based alignment (ArUco)**: detect ArUco markers (IDs 0–3), estimate rotation, rotate/crop the image into a consistent frame.
- **Segmentation of printed material**: convert image to LAB and threshold the A-channel to isolate **red** printed regions, then clean with morphology.
- **Binary mask generation**: convert segmented image to a binary representation of printed vs not printed.
- **Layer comparison / differencing**: compare the processed actual layer mask with the expected model layer image using bitwise operations to find **missing material**.
- **Defect skeletonization**: skeletonize defect regions to thin them into line-like features.
- **Defect grouping**: cluster adjacent defect pixels into connected groups.
- **Optional corrective G-code generation**: convert clustered defect geometry into simple G-code moves with extrusion (retraction → move → unretract → extrude).

**Main output:** defect map (implicit) and a list of **correction G-code commands**.

---

## How They Fit Together (Pipeline Architecture)

My scripts solve the upstream robustness problem:
- “In a busy image, find the correct object and produce a clean ROI.”

Shehan’s CompareLayer solves the downstream comparison problem:
- “Given an aligned view, compare actual vs expected and identify defects (and optionally generate correction).”

The clean integration is:
- Use **my localisation + ROI extraction** to ensure the correct target is selected,
- Feed that ROI into **CompareLayer’s alignment + comparison**, or replace ArUco with ORB alignment if markers are not used.

---

## Architecture Diagram (Markdown)

```text
                   ┌────────────────────────────────────────────┐
                   │              Inputs / Sources               │
                   │  - Full-frame camera image (during print)   │
                   │  - Expected layer image (from slicer/model) │
                   │  - Optional priors (colour, size, position) │
                   └────────────────────────────────────────────┘
                                      │
                                      ▼
          ┌──────────────────────────────────────────────────────────┐
          │  Module 1: Localisation (My Photogrammetry / CV Scripts)  │
          │  - HSV / structure masks                                  │
          │  - contour scoring + filters                               │
          │  - reject clutter / labels / wrong candidates              │
          │  - output ROI + bbox + confidence                          │
          └──────────────────────────────────────────────────────────┘
                                      │
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │   Module 2: Registration / Alignment (Hybrid approach)      │
        │   Option A: ArUco markers (CompareLayer)                    │
        │   Option B: ORB + homography (My scripts / hybrid)          │
        └────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
          ┌──────────────────────────────────────────────────────────┐
          │     Module 3: Comparison (Shehan’s CompareLayer Core)     │
          │  - segmentation → binary mask                              │
          │  - model vs actual differencing                            │
          │  - defect map + skeletonization                            │
          │  - cluster defects                                         │
          └──────────────────────────────────────────────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
┌──────────────────────────────────┐        ┌──────────────────────────────────┐
│ Outputs for Dissertation / Eval   │        │ Optional Closed-loop Extension   │
│ - defect map (visual)             │        │ - correction G-code generation   │
│ - defect score / metrics          │        │ - corrective print pass          │
│ - performance vs defects dataset  │        │ (treat as exploratory/future)    │
└──────────────────────────────────┘        └──────────────────────────────────┘

## Quick conclusion of the above

SO this is a primising architecture of what I might be able to do, combining my photogrammetry with this part of Shehan's script that is actually integrated with the main application and the printer controls/setup. However a lot of investigation needs to be undertaken into the actual function and eficacy of the compare_layer scripts; is it doable? how much does it depend on angle/photos taken, is Aruco markers the best way forward?

## Further Important Note

Now all of the above is promising and interesting, however is should CERTAINLY be noted and considered that it has been expressed in weekly meetings, that the focus of my photogrammetry and sensing should not necessarily be focused on an arial (from above view) and we should mroeso be focusing on the side view, how will this fit in? can similar logic be used? how will it be used? 
