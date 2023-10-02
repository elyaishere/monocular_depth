import numpy as np

def getStep(counts, width=255, kind='adaptive'):
    if kind == 'simple':
        return [max(1, int(width / len(counts))) for _ in counts]
    return [max(1, int((l * width) / sum(counts))) for l in counts]


def simple_correction(mask, corrected_depth, current_min, step):
    current_min = max(current_min, np.min(corrected_depth[mask]))
    current_max = min(current_min + step, 255)
    assert current_min < current_max, f"min = {current_min}, max = {current_max}"
    corrected_depth[mask] = np.clip(corrected_depth[mask], current_min, current_max)
    return corrected_depth, current_min, current_max



def shift_correction(mask, corrected_depth, current_min, step):
    current_min = max(current_min, np.min(corrected_depth[mask]))
    current_max = min(current_min + step, 255)
    assert current_min < current_max, f"min = {current_min}, max = {current_max}"      
    shift = np.min(corrected_depth[mask]) - current_min
    corrected_depth[mask] = corrected_depth[mask] - shift
    corrected_depth[mask] = np.clip(corrected_depth[mask], current_min, current_max)
    return corrected_depth, current_min, current_max


def affine_correction(mask, corrected_depth, current_min, step):
    current_min = max(current_min, np.min(corrected_depth[mask]))
    current_max = min(current_min + step, 255)
    assert current_min < current_max, f"min = {current_min}, max = {current_max}"
    depth_min = np.min(corrected_depth[mask])
    depth_max = np.max(corrected_depth[mask])
    if depth_max != depth_min:
        scale = (current_max - current_min) / (depth_max - depth_min)
        shift = current_min - scale * depth_min
    else:
        scale = 0
        shift = current_min - depth_min
    corrected_depth[mask] = (scale * corrected_depth[mask] + shift).round()
    corrected_depth[mask] = np.where(corrected_depth[mask] < current_min, current_min, corrected_depth[mask])
    corrected_depth[mask] = np.where(corrected_depth[mask] > current_max, current_max, corrected_depth[mask])
    return corrected_depth, current_min, current_max


def correct_depth(predicted_depth, gt_relative_depth, method=simple_correction, pad_token=209, step_kind='adaptive'):
    corrected_depth = np.copy(predicted_depth) - np.min(predicted_depth) # shift to 0
    depths = np.unique(gt_relative_depth)[:-1]
    interdepths = np.unique([d % 10 for d in depths])
    lens = [len(np.where((gt_relative_depth % 10 == d) * (gt_relative_depth != pad_token))[0]) for d in interdepths]
    steps = getStep(lens, kind=step_kind)
    current_min = 0
    for i, interdepth in enumerate(interdepths):
        mask = (gt_relative_depth % 10 == interdepth) * (gt_relative_depth != pad_token)
        # correction within one interdepth
        corrected_depth, current_min, current_max = method(mask, corrected_depth, current_min, steps[i])

        intradepths = sorted(depths[depths % 10 == interdepth])
        sublens = [len(np.where(gt_relative_depth == d)[0]) for d in intradepths]
        substeps = getStep(sublens, width=steps[i], kind=step_kind)
        submin = current_min
        for j, intradepth in enumerate(intradepths):
            submask = gt_relative_depth == intradepth
            # correction within one intradepth
            corrected_depth, submin, _ = method(submask, corrected_depth, submin, substeps[j])
            submin = min(np.max(corrected_depth[submask]) + 1, current_max)

        current_min = min(np.max(corrected_depth[mask]) + 1, 255)
    
    return corrected_depth


def sanity_check(gt_relative_depth, corrected_depth, pad_token=209):
    depths = [i for i in np.unique(gt_relative_depth) if i != pad_token] # without unannotated parts
    interdepths = np.unique([d % 10 for d in depths])
    for i, interdepth in enumerate(interdepths):
        if i > 0:
            prev_max = np.max(corrected_depth[
              (gt_relative_depth % 10 == interdepths[i-1]) *
              (gt_relative_depth != pad_token)
            ])
            cur_min = np.min(corrected_depth[
              (gt_relative_depth % 10 == interdepth) *
              (gt_relative_depth != pad_token)
            ])
            assert prev_max < cur_min, f"prev_max = {prev_max}, cur_min = {cur_min}"
        intradepths = sorted(depths[depths % 10 == interdepth])
        if len(intradepths) > 1:
            for i, intradepth in enumerate(intradepths):
                if i > 0:
                    prev_max = np.max(corrected_depth[gt_relative_depth == intradepths[i-1]])
                    cur_min = np.min(corrected_depth[gt_relative_depth == intradepth])
                    assert prev_max < cur_min, f"prev_max = {prev_max}, cur_min = {cur_min}"
