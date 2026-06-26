#!/usr/bin/env python3
"""dtm_merge_reproject.py - mosaic PST LiDAR DTM tiles and reproject to a metric CRS.

The "Download Banca Dati PST" portal delivers the terrain model as several GeoTIFF tiles
in geographic coordinates (WGS84 / EPSG:4326). This merges them into one continuous DTM
and reprojects to a projected CRS (default UTM 33N / EPSG:32633, correct for Calabria),
so the terrain shares the coordinate system of a GPS-georeferenced COLMAP/3DGS model
(see geo_align.py) and is ready for ICP-based refinement (e.g. CloudCompare).

Requires: rasterio, numpy   (pip install rasterio numpy)

Usage:
  python3 dtm_merge_reproject.py --in ./dtm_tiles --out aspromonte_dtm_utm33n.tif
  python3 dtm_merge_reproject.py --in ./dtm_tiles --out dem.tif --epsg 32632 --resampling cubic
"""
import argparse
import glob
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="Mosaic + reproject DTM GeoTIFF tiles.")
    ap.add_argument("--in", dest="indir", required=True, help="Folder with the *_DTM.tif(f) tiles.")
    ap.add_argument("--out", default="dtm_utm33n.tif")
    ap.add_argument("--epsg", type=int, default=32633, help="Target CRS (default UTM 33N).")
    ap.add_argument("--resampling", default="bilinear", choices=["nearest", "bilinear", "cubic"],
                    help="Resampling for reprojection (bilinear suits continuous elevation).")
    args = ap.parse_args()

    import numpy as np
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    tiles = sorted(set(glob.glob(os.path.join(args.indir, "*.tif")) +
                       glob.glob(os.path.join(args.indir, "*.tiff"))))
    if not tiles:
        sys.exit(f"No .tif/.tiff tiles found in {args.indir}")
    print(f"Found {len(tiles)} tiles.")

    # 1) Mosaic the tiles into one array (they share a CRS and resolution).
    srcs = [rasterio.open(t) for t in tiles]
    src_crs, nodata, dtype = srcs[0].crs, srcs[0].nodata, srcs[0].dtypes[0]
    mosaic, m_transform = merge(srcs, nodata=nodata)
    for s in srcs:
        s.close()
    bands, h, w = mosaic.shape
    left, top = m_transform.c, m_transform.f
    right, bottom = left + w * m_transform.a, top + h * m_transform.e
    print(f"Mosaic: {w}x{h}, CRS {src_crs}, dtype {dtype}, nodata {nodata}")

    # 2) Reproject the mosaic to the target metric CRS.
    dst_crs = f"EPSG:{args.epsg}"
    dst_transform, dw, dh = calculate_default_transform(
        src_crs, dst_crs, w, h, left=left, bottom=bottom, right=right, top=top)
    fill = nodata if nodata is not None else 0
    dst = np.full((bands, dh, dw), fill, dtype=dtype)
    reproject(source=mosaic, destination=dst,
              src_transform=m_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              src_nodata=nodata, dst_nodata=nodata,
              resampling=getattr(Resampling, args.resampling))

    profile = {
        "driver": "GTiff", "dtype": dtype, "count": bands,
        "height": dh, "width": dw, "crs": dst_crs, "transform": dst_transform,
        "nodata": nodata, "compress": "deflate", "tiled": True,
        "predictor": 3 if np.issubdtype(np.dtype(dtype), np.floating) else 2,
    }
    with rasterio.open(args.out, "w", **profile) as f:
        f.write(dst)
    print(f"Wrote {args.out}  ({dw}x{dh}, EPSG:{args.epsg}, {args.resampling} resampling).")


if __name__ == "__main__":
    main()