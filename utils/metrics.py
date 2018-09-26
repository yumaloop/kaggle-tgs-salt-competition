import numpy as np
import tensorflow as tf

def get_iou_vector(v_true, v_pred):
    v_true = np.squeeze(v_true) 
    v_pred = np.squeeze(v_pred) 
    
    batch_size = v_true.shape[0]
    metric = []    
    
    for batch in range(batch_size):
        
        t,  p = v_true[batch] > 0,  v_pred[batch] > 0
        
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue
        
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0)  )/ (np.sum(union > 0) )
        thresholds = np.arange(0.5, 1, 0.05)
        
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)


def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.0], tf.float64)