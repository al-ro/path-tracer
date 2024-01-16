# Path Tracer

This work is in progress.

Multithreaded C++ path tracer with bounding volume hierarchy (BVH) based on [the tutorial series by jbikker](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)

[CUDA version for Nvidia GPUs](https://github.com/al-ro/path-tracer)

<table width="100%">
  <thead>
    <tr>
      <th width="50%">Scene 0: 3 instances</th>
      <th width="50%">BVH heat map (max 267) 0.08 s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="50%"><img src="images/three_stl.png"/></td>
      <td width="50%"><img src="images/three_stl_bvh.png"/></td>
    </tr>
  </tbody>
</table>

<table width="100%">
  <thead>
    <tr>
      <th width="50%">Scene 1: 10,000 instances</th>
      <th width="50%">BVH heat map (max 600) 0.26 s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="50%"><img src="images/scatter_stl.png"/></td>
      <td width="50%"><img src="images/scatter_stl_bvh.png"/></td>
    </tr>
  </tbody>
</table>

<table width="100%">
  <thead>
    <tr>
      <th width="50%">Scene 2: Model with vertex attributes</th>
      <th width="50%">BVH heat map (max 123) 0.07 s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="50%"><img src="images/obj.png"/></td>
      <td width="50%"><img src="images/obj_bvh.png"/></td>
    </tr>
  </tbody>
</table>

- Resolution: 1500 x 800
- Samples: 320
- Threads: 10
- Max bounces: 6

Scene | Render | BVH build | TLAS nodes | Triangles (per model) | BLAS nodes (per model)
:---:|:---:|:---:|:---:|:---:|:---:|
0 | 36 s | 0.88 s | 5 | 505,848 | 792,591 
1 | 269 s | 0.88 s | 19,819 | 505,848 | 792,591 
2 | 107 s | 0.005 s |  1 | 3,828 | 4899 


["Bust of Menelaus"](https://www.myminifactory.com/object/3d-print-bust-of-menelaus-32197) by Scan The World

["Viking room"](https://sketchfab.com/3d-models/viking-room-a49f1b8e4f5c4ecf9e1fe7d81915ad38) by nigelgoh (edited)

Environment map from [HDR Haven](https://hdri-haven.com/)

## Features

- BVH (BLAS + TLAS) construction and traversal
- Cook-Torrance
- Trowbridge-Reitz (GGX) specular
- Lambertian diffuse
- BRDF importance sampling

## Use

Running outputs parameters and progress to the console

Resulting image is output as *output.bmp*

- make clean (to remove old executable)
- make build
- make run (with default settings)
- make all (clean, build, run)
- ./PathTracer \[options\]
    - *-w* width of rendered image (e.g. -w 512)
    - *-h* height of rendered image
    - *-s* samples per pixel
    - *-b* maximum bounces per ray per sample
    - *-t* number of threads
    - *-p* preset scene \[0, 2\]
    - *-a* output BVH heatmap (ignores -s and -b)


## Dependencies

- [GLM](https://github.com/g-truc/glm) for maths functions and data structures
- [stb](https://github.com/nothings/stb) for reading and writing images (included in /lib)
- [stl_reader](https://github.com/sreiter/stl_reader) for reading STL files (included in /lib)
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) for loading OBJ files (included in /lib)
- C++17
- make (optional)
