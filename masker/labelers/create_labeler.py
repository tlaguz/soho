from masker.labelers.warped_labeler import WarpedLabeler


def create_labeler():
    return WarpedLabeler(height = 1024, width = 1024, sigma_r = 9, sigma_phi = 0.08)
