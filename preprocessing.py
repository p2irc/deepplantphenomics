import numpy as np
import plantcv as pcv
import os


def doParallelAutoSegmentation(file, outdir):
    filename = os.path.basename(file)
    outpath = os.path.join(outdir, filename)

    if not os.path.exists(outpath):
        auto_segment(file, outpath)

    return outpath


def auto_segment(in_file, out_file):
    device = 0
    debug = 'None'

    img, path, filename = pcv.readimage(in_file)
    device, img_gray_sat = pcv.rgb2gray_hsv(img, 's', device, debug)

    device, img_binary = pcv.binary_threshold(img_gray_sat, 50, 255, 'light', device, debug)
    mask = np.copy(img_binary)

    device, fill_image = pcv.fill(img_binary, mask, 300, device, debug)

    device, dilated = pcv.dilate(fill_image, 1, 1, device, debug)

    device, id_objects, obj_hierarchy = pcv.find_objects(img, dilated, device, debug)

    device, roi, roi_hierarchy = pcv.define_roi(img, 'rectangle', device, None, 'default', debug, True, 600, 100, -600, -300)

    device, roi_objects, roi_obj_hierarchy, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi, roi_hierarchy,
                                                                                  id_objects, obj_hierarchy, device,
                                                                                  debug)

    device, obj, mask = pcv.object_composition(img, roi_objects, roi_obj_hierarchy, device, debug)

    device, masked_image = pcv.apply_mask(img, mask, 'white', device, debug)

    pcv.print_image(masked_image, out_file)
