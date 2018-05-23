from . import loaders
import plantcv as pcv
import os


def do_parallel_auto_segmentation(file, coords, outdir, img_height, img_width):
    def auto_segment(in_file, out_file, coords):
        x_adj = coords[0]
        y_adj = coords[1]
        w_adj = coords[2]
        h_adj = coords[3]

        device = 0
        debug = 'None'

        img, _, _ = pcv.readimage(in_file)
        device, sat = pcv.rgb2gray_hsv(img, 's', device, debug)

        device, img_binary = pcv.otsu_auto_threshold(sat, 255, 'light', device, debug)

        device, id_objects, obj_hierarchy = pcv.find_objects(img, img_binary, device, debug)

        device, roi, roi_hierarchy = pcv.define_roi(img, 'rectangle', device, None, 'default', debug, True,
                                                    x_adj, y_adj, w_adj, h_adj)

        device, roi_objects, roi_obj_hierarchy, _, _ = pcv.roi_objects(img, 'partial', roi, roi_hierarchy, id_objects,
                                                                       obj_hierarchy, device, debug)

        device, obj, mask = pcv.object_composition(img, roi_objects, roi_obj_hierarchy, device, debug)

        device, masked_image = pcv.apply_mask(img, mask, 'white', device, debug)

        pcv.print_image(masked_image, out_file)

    filename = os.path.basename(file)
    outpath = os.path.join(outdir, filename)

    fixed_coords = loaders.pascal_voc_coordinates_to_pcv_coordinates(img_height, img_width, coords)

    if not os.path.exists(outpath):
        auto_segment(file, outpath, fixed_coords)

    return outpath
