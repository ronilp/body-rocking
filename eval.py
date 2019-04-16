#inputs are "maps" which are just a list of lists.
#Each index in the list represents an event or detection interval.
#map[i][0] is the start index of the ith event/detection.
#Similarly map[i][1] is the end index of the ith event/detection
def get_metric_values(gt_map,pred_map):
    gt_idx = 0
    pred_idx = 0
    while gt_idx < len(gt_map) and pred_idx < len(pred_map):
        if gt_map[gt_idx][0] <= pred_map[pred_idx][0]:
            if gt_map[gt_idx][1] < pred_map[pred_idx][0]:
                FN += 1
                gt_idx += 1
            elif gt_map[gt_idx][1] >= pred_map[pred_idx][1]:
                TP += 1
                gt_idx, pred_idx = increment_idcs_exit_event(gt_map, pred_map, gt_idx, pred_idx)
            elif get_overlap(gt_map[gt_idx],pred_map[pred_idx]) < 0.5:
                FN += 1
                gt_idx += 1
            else:
                TP += 1
                gt_idx += 1
                pred_idx += 1
        else:
            if gt_map[gt_idx][0] > pred_map[pred_idx][1]:
                FP += 1
                pred_idx += 1
            elif get_overlap(gt_map[gt_idx],pred_map[pred_idx]) < 0.5:
                FP += 1
                pred_idx += 1
            else:
                TP += 1
                gt_idx, pred_idx = increment_idcs_exit_event(gt_map, pred_map, gt_idx, pred_idx)
    FN += (len(gt_map) - gt_idx)
    FP += (len(pred_map) - pred_idx)
    return FP, FN, TP
    
    
def increment_idcs_exit_event(gt_map, pred_map, gt_idx, pred_idx):
    while gt_map[gt_idx][1] > pred_map[pred_idx][0]:
        if get_overlap(gt_map[gt_idx],pred_map[pred_idx]) < 0.5:
            gt_idx += 1
            return gt_idx, pred_idx
        pred_idx += 1
        if gt_idx >= len(gt_map) or pred_idx >= len(pred_map):
            return gt_idx, pred_idx
    gt_idx += 1
    return gt_idx, pred_idx