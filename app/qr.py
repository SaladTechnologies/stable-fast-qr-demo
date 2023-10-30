import os
import qrcode
from qrcode.image.styledpil import StyledPilImage
import qrcode.image.styles.moduledrawers.pil as module_drawers
from qrcode.image.styles import colormasks
from qreader import QReader
import numpy as np


image_size = int(os.getenv("IMAGE_SIZE", "512"))

qr_decoder = QReader(reencode_to=None)

color_mask_defaults = {
    "SolidFill": {"front_color": (0, 0, 0), "back_color": (128, 128, 128)},
    "RadialGradiant": {
        "center_color": (0, 0, 0),
        "back_color": (128, 128, 128),
        "edge_color": (0, 0, 255),
    },
    "SquareGradiant": {
        "center_color": (0, 0, 0),
        "back_color": (128, 128, 128),
        "edge_color": (0, 0, 255),
    },
    "HorizontalGradiant": {
        "left_color": (0, 0, 0),
        "back_color": (128, 128, 128),
        "right_color": (0, 0, 255),
    },
    "VerticalGradiant": {
        "top_color": (0, 0, 0),
        "back_color": (128, 128, 128),
        "bottom_color": (0, 0, 255),
    },
}


def get_qr_control_image(
    url,
    size=image_size,
    error_correction="M",
    drawer="RoundedModule",
    color_mask="SolidFill",
    color_mask_params=None,
):
    error_corrector = getattr(qrcode.constants, f"ERROR_CORRECT_{error_correction}")
    module_drawer = getattr(module_drawers, f"{drawer}Drawer")
    if color_mask is not None:
        mask = getattr(colormasks, f"{color_mask}ColorMask")
        if color_mask_params is None:
            color_mask_params = color_mask_defaults[color_mask]
        else:
            color_mask_params = {**color_mask_defaults[color_mask], **color_mask_params}
        mask = mask(**color_mask_params)
    else:
        mask = None

    make_image_params = {
        "image_factory": StyledPilImage,
        "module_drawer": module_drawer(),
    }
    if mask is not None:
        make_image_params["color_mask"] = mask

    qr = qrcode.QRCode(error_correction=error_corrector)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(
        **make_image_params,
    )
    return img.resize((size, size))


def detect_qr_code(img):
    value = qr_decoder.detect_and_decode(image=np.array(img))
    if value:
        return value[0]
    else:
        return None
