import numpy as np


def normalize(x):
    """ Normalize """
    x_min = np.min(x)
    x_max = np.max(x)
    denom = x_max - x_min
    if denom == 0:
        return np.zeros_like(x)
    x = (x - x_min) / denom
    return x


def normalize_per_feature(x, eps=1e-12):
    """Min-max normalize each feature column independently."""
    x = np.asarray(x)
    x_min = np.min(x, axis=0, keepdims=True)
    x_max = np.max(x, axis=0, keepdims=True)
    denom = np.maximum(x_max - x_min, eps)
    return (x - x_min) / denom


def standardize_per_feature(x, eps=1e-12, clip_value=None):
    """Z-score standardize each feature column independently."""
    x = np.asarray(x)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.maximum(std, eps)
    z = (x - mean) / std
    if clip_value is not None:
        z = np.clip(z, -float(clip_value), float(clip_value))
    return z

def cal_std(logger, *arg):
    """ print clustering results """
    if len(arg) == 4:
        logger.info(arg[0])
        logger.info(arg[1])
        logger.info(arg[2])
        logger.info(arg[3])
        output = """ 
                     ACC {:.2f} std {:.2f}
                     NMI {:.2f} std {:.2f} 
                     ARI {:.2f} std {:.2f}
                     F-score {:.2f} std {:.2f}""".format(
            np.mean(arg[0]) * 100, np.std(arg[0]) * 100,
            np.mean(arg[1]) * 100, np.std(arg[1]) * 100,
            np.mean(arg[2]) * 100, np.std(arg[2]) * 100,
            np.mean(arg[3]) * 100, np.std(arg[3]) * 100)
        logger.info(output)
        output2 = str(round(np.mean(arg[0]) * 100, 2)) + ',' + str(round(np.std(arg[0]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[1]) * 100, 2)) + ',' + str(round(np.std(arg[1]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[2]) * 100, 2)) + ',' + str(round(np.std(arg[2]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[3]) * 100, 2)) + ',' + str(round(np.std(arg[3]) * 100, 2)) + ';'
        logger.info(output2)
        return round(np.mean(arg[0]) * 100, 2), round(np.mean(arg[1]) * 100, 2), \
               round(np.mean(arg[2]) * 100, 2), round(np.mean(arg[3]) * 100, 2)

    elif len(arg) == 3:
        logger.info(arg[0])
        logger.info(arg[1])
        logger.info(arg[2])
        output = """ 
                     ACC {:.2f} std {:.2f}
                     NMI {:.2f} std {:.2f} 
                     ARI {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100, np.mean(arg[1]) * 100,
                                                     np.std(arg[1]) * 100, np.mean(arg[2]) * 100, np.std(arg[2]) * 100)
        logger.info(output)
        output2 = str(round(np.mean(arg[0]) * 100, 2)) + ',' + str(round(np.std(arg[0]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[1]) * 100, 2)) + ',' + str(round(np.std(arg[1]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[2]) * 100, 2)) + ',' + str(round(np.std(arg[2]) * 100, 2)) + ';'
        logger.info(output2)
        return round(np.mean(arg[0]) * 100, 2), round(np.mean(arg[1]) * 100, 2), round(np.mean(arg[2]) * 100, 2)

    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
        logger.info(output)


def cal_classify(logger, *arg):
    """ print classification results """
    if len(arg) == 3:
        logger.info(arg[0])
        logger.info(arg[1])
        logger.info(arg[2])
        output = """ 
                     ACC {:.2f} std {:.2f}
                     Precision {:.2f} std {:.2f} 
                     F-measure {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100,
                                                           np.mean(arg[1]) * 100,
                                                           np.std(arg[1]) * 100, np.mean(arg[2]) * 100,
                                                           np.std(arg[2]) * 100)
        logger.info(output)
        output2 = str(round(np.mean(arg[0]) * 100, 2)) + ',' + str(round(np.std(arg[0]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[1]) * 100, 2)) + ',' + str(round(np.std(arg[1]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[2]) * 100, 2)) + ',' + str(round(np.std(arg[2]) * 100, 2)) + ';'
        logger.info(output2)
        return round(np.mean(arg[0]) * 100, 2), round(np.mean(arg[1]) * 100, 2), round(np.mean(arg[2]) * 100, 2)
    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
        logger.info(output)
    return