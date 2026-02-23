diff --git a//Users/jameswalter/Desktop/Photogammetry/PIPELINE/COLMAP_SPARSE_CREATION_FLOWCHART.mmd b//Users/jameswalter/Desktop/Photogammetry/PIPELINE/COLMAP_SPARSE_CREATION_FLOWCHART.mmd
new file mode 100644
--- /dev/null
+++ b//Users/jameswalter/Desktop/Photogammetry/PIPELINE/COLMAP_SPARSE_CREATION_FLOWCHART.mmd
@@ -0,0 +1,68 @@
+flowchart TD
+    A([Start]) --> B[Parse CLI args]
+    B --> C{preset != "none"?}
+    C -- Yes --> C1[Load preset values and merge into args]
+    C -- No --> D
+    C1 --> D{camera_model missing?}
+    D -- Yes --> D1[Set camera_model = PINHOLE]
+    D -- No --> E
+    D1 --> E{camera_model not PINHOLE or SIMPLE_PINHOLE?}
+    E -- Yes --> E1[Print compatibility warning]
+    E -- No --> F
+    E1 --> F{threads provided?}
+    F -- Yes --> F1[Append num_threads to feature, match, mapper opts]
+    F -- No --> G
+    F1 --> G[Validate project dir and images subdir]
+    G --> H{no_single_camera?}
+    H -- Yes --> H1[Set single_camera = false]
+    H -- No --> I{single_camera and mixed image sizes?}
+    H1 --> J
+    I -- Yes --> I1[Warn and disable single_camera]
+    I -- No --> J[Write .colmap_image_list.txt]
+    I1 --> J
+    J --> K[Find COLMAP binary]
+    K --> L[Read feature/matcher help and resolve GPU option keys]
+    L --> M[Build database/sparse paths]
+    M --> N{force?}
+    N -- Yes --> N1[Delete database, sparse_raw, sparse, sparse.ply]
+    N -- No --> O
+    N1 --> O[Create sparse_raw directory]
+    O --> P{project_ini provided?}
+    P -- Yes --> P1[Resolve and validate ini path]
+    P -- No --> Q
+    P1 --> Q[Parse and dedupe feature/match/mapper options]
+    Q --> R[Ensure mapper defaults: multiple_models=0, max_num_models=1]
+    R --> S[Run COLMAP feature_extractor]
+    S --> T{matcher == exhaustive?}
+    T -- Yes --> T1[Build exhaustive_matcher command]
+    T -- No --> T2[Build sequential_matcher command]
+    T1 --> U[Run matcher command]
+    T2 --> U
+    U --> V[Run COLMAP mapper -> sparse_raw]
+    V --> W{dry_run?}
+    W -- Yes --> W1[Print DRY RUN COMPLETE]
+    W1 --> Z([Stop])
+    W -- No --> X[flatten_sparse_model(sparse_raw, sparse, colmap)]
+
+    subgraph FS[flatten_sparse_model()]
+        X --> FS1[Find numeric model folders under sparse_raw]
+        FS1 --> FS2{No model dirs?}
+        FS2 -- Yes --> FS_ERR1[Raise FileNotFoundError]
+        FS2 -- No --> FS3[For each model: count images and 3D points]
+        FS3 --> FS4{count_model_stats failed?}
+        FS4 -- Yes --> FS4A[Use fallback stats 0,0]
+        FS4 -- No --> FS5
+        FS4A --> FS5[Add model to ranked list]
+        FS5 --> FS6[Sort by images desc, points desc, folder index desc]
+        FS6 --> FS7[Select top-ranked model and print candidates]
+        FS7 --> FS8{Missing cameras/images/points3D?}
+        FS8 -- Yes --> FS_ERR2[Raise FileNotFoundError]
+        FS8 -- No --> FS9[Ensure sparse dir and copy model bins]
+    end
+
+    FS9 --> Y[Run model_converter: sparse -> sparse.ply]
+    Y --> Y1[Print DONE and next step MESH_FROM_SPARSE.py]
+    Y1 --> Z
+
+    FS_ERR1 --> Z
+    FS_ERR2 --> Z
