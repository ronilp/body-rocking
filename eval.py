# inputs are "maps" which are just a list of lists. Please ignore the poor naming convention.
# Each index in the list represents an event or detection interval.
# map[i][0] is the start index of the ith event/detection.
# Similarly map[i][1] is the end index of the ith event/detection
def get_metric_values(gt_map, pred_map):
    gt_idx = 0
    pred_idx = 0
    FN = 0
    TP = 0
    FP = 0
    while gt_idx < len(gt_map) and pred_idx < len(pred_map):
        if gt_map[gt_idx][0] <= pred_map[pred_idx][0]:
            if gt_map[gt_idx][1] < pred_map[pred_idx][0]:
                FN += 1
                gt_idx += 1
            elif gt_map[gt_idx][1] >= pred_map[pred_idx][1]:
                TP += 1
                gt_idx, pred_idx = increment_idcs_exit_event(gt_map, pred_map, gt_idx, pred_idx)
            elif get_overlap(gt_map[gt_idx], pred_map[pred_idx]) < 0.5:
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
            elif get_overlap(gt_map[gt_idx], pred_map[pred_idx]) < 0.5:
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
        if get_overlap(gt_map[gt_idx], pred_map[pred_idx]) < 0.5:
            gt_idx += 1
            return gt_idx, pred_idx
        pred_idx += 1
        if gt_idx >= len(gt_map) or pred_idx >= len(pred_map):
            return gt_idx, pred_idx
    gt_idx += 1
    return gt_idx, pred_idx


def get_overlap(gt_tup, pred_tup):
    if pred_tup[1] > gt_tup[1]:
        num = (gt_tup[1] - pred_tup[0]) + 1
    else:
        num = (pred_tup[1] - gt_tup[0]) + 1
    return num / ((pred_tup[1] - pred_tup[0]) + 1)


prediction_files = ["Session02.txt", "Session03.txt"]
ground_truth_files = ["data/TrainingData/test/Session02/detection.txt", "data/TrainingData/test/Session03/detection.txt"]


def get_precision(TP, FP):
    return TP / (TP + FP)


def get_recall(TP, FN):
    return TP / (TP + FN)


def get_f1(TP, FP, FN):
    p = get_precision(TP, FP)
    r = get_recall(TP, FN)
    return (2 * p * r) / (p + r + 0.0000000001)

def get_map(values):
    gt_map = []
    j = -1
    for i in range(len(values)):
        val = values[i]
        if val == 0:
            continue

        if i > 0 and values[i - 1] == 1:
            gt_map[j][1] = i
        else:
            j += 1
            gt_map.append([0, 0])
            gt_map[j][0] = i
            gt_map[j][1] = i

    return gt_map

FP_running = 0
FN_running = 0
TP_running = 0

if __name__ == "__main__":

    for i in range(len(ground_truth_files)):
        print("Reading ", prediction_files[i])
        ground_truth = []
        with open(ground_truth_files[i]) as f:
            for line in f:
                ground_truth.append(float(line.strip()))

        predictions = []
        with open(prediction_files[i]) as f:
            for line in f:
                predictions.append(float(line.strip()))

        gt_map = get_map(ground_truth)
        pred_map = get_map(predictions)

        # print("Ground truth")
        # for event in gt_map:
        #     print (event[1] - event[0])
        #
        # print("Predictions")
        # for event in pred_map:
        #     print (event[1] - event[0])

        FP, FN, TP = get_metric_values(gt_map, pred_map)
        FP_running += FP
        FN_running += FN
        TP_running += TP

        print ("Results")
        print(TP, FN, FP)
        print("Precision :", get_precision(TP, FP))
        print("Recall :", get_recall(TP, FN))
        print("F1 :", get_f1(TP, FP, FN))

    print("Averages")
    TP_avg = TP_running/len(prediction_files)
    FN_avg = FN_running/len(prediction_files)
    FP_avg = FP_running/len(prediction_files)
    print(TP_avg, FN_avg, FP_avg)

    print ("Precision :", get_precision(TP_avg, FP_avg))
    print ("Recall :", get_recall(TP_avg, FN_avg))
    print ("F1 :", get_f1(TP_avg, FP_avg, FN_avg))