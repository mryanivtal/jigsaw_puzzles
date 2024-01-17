
# -------------------- inference flow - display two patches and image in flow-------------------
from src.utils.image_utils import display_image

pair_idx = 22
plain_image, _ = super(DogsVsCatsJigsawDataset, dataset).get_item(image_idx, for_display=False)
display_image(plain_image)
display_image(pair_patches[pair_idx][:3])
display_image(pair_patches[pair_idx][3:])
# ---------------------------------------

