import loaders
import numpy as np
import plantcv as pcv
import os


def doParallelAutoSegmentation(file, coords, outdir, img_height, img_width):
    def autoSegment(in_file, out_file, coords):
        x_adj = coords[0]
        y_adj = coords[1]
        w_adj = coords[2]
        h_adj = coords[3]

        device = 0
        debug = 'None'

        img, path, filename = pcv.readimage(in_file)
        device, img_gray_sat = pcv.rgb2gray_hsv(img, 's', device, debug)

        device, img_binary = pcv.binary_threshold(img_gray_sat, 50, 255, 'light', device, debug)
        mask = np.copy(img_binary)

        device, fill_image = pcv.fill(img_binary, mask, 300, device, debug)

        device, dilated = pcv.dilate(fill_image, 1, 1, device, debug)

        device, id_objects, obj_hierarchy = pcv.find_objects(img, dilated, device, debug)

        device, roi, roi_hierarchy = pcv.define_roi(img, 'rectangle', device, None, 'default', debug, True,
                                                    x_adj, y_adj, w_adj, h_adj)

        device, roi_objects, roi_obj_hierarchy, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi,
                                                                                      roi_hierarchy,
                                                                                      id_objects, obj_hierarchy, device,
                                                                                      debug)

        device, obj, mask = pcv.object_composition(img, roi_objects, roi_obj_hierarchy, device, debug)

        device, masked_image = pcv.apply_mask(img, mask, 'white', device, debug)

        pcv.print_image(masked_image, out_file)

    filename = os.path.basename(file)
    outpath = os.path.join(outdir, filename)

    fixed_coords = loaders.pascalVOCCoordinatesToPCVCoordinates(img_height, img_width, coords)

    if not os.path.exists(outpath):
        autoSegment(file, outpath, fixed_coords)

    return outpath
