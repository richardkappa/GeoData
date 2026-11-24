"""Section 3 - new data creation: per-postcode features.

Everything is computed for the postcodes of the section 2 lookup and keyed on
``postcode_ns``. Modules:

* :mod:`.grid`       densities / sums / coverage within a radius (raster grid
                     with row prefix sums: one vectorised gather per row of the
                     disc, so cost does not grow with how many points are inside)
* :mod:`.nearest`    distance to the nearest point / line / polygon (KD-tree)
* :mod:`.network`    walking and driving distances over OS Open Roads
                     (sparse graph + multi-source Dijkstra)
* :mod:`.adjacency`  LSOA neighbourhoods via a sparse contiguity matrix
* :mod:`.terrain`    Terrain 50 elevation, slope, aspect, relative height
* :mod:`.trends`     house-price and crime trends
"""
