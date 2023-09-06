import numpy as np
import pandas as pd


class initialize():
    def __init__(self, C=1, fuzzifier=1.5, epsilon=0.01):
        self.C = C
        self.fuzzifier = fuzzifier # m
        self.epsilon = epsilon # η
        
        assert self.fuzzifier >= 1
        assert self.epsilon > 0

    def random_select(self, data, itr, U, seed):
        # Probabilities to choose cluster center
        N = data.shape[0]
        if itr == 1:
            p_trans = np.ones(N) / N
        else:
            p = np.max(U, axis=0)
            p_trans = p.copy()
            alpha_cut = 0.5
            p_trans[p > alpha_cut] = 0
            p_trans[p <= alpha_cut] = 1 - p[p <= alpha_cut]
            p_trans /= np.sum(p_trans)
            
        # Select v with possibilities
        np.random.seed(seed)
        selected_v = np.random.choice(np.arange(N), size=1, p=p_trans)
        return data[selected_v[0]]

    def sp1m(self, data):
        dim = data.shape[1]
        N = data.shape[0]
        
        C = self.C              # 1,  cluster number, infinite
        fzr = self.fuzzifier    # 1.5, fuzzifier
        epsilon = self.epsilon  # 0.01, epsilon threshold
        Uall = []               # membership matrix collection
        Vall = []               # cluster matrix collection
        cov_max = []            # covariance maxtrixs
        point_num = []          # each cluster points number
        label = np.ones(N) * 10 
        U_count = np.zeros(C)
        stop_count = 0
        stop_code = 0
        seed = 2019
        eta = np.ones(C) * 30

        for itr in range(1, C + 1): # C num. clusters
            while True:
                seed += 1
                selected_v = self.random_select(data, itr, Uall, seed)
                v = selected_v.copy()

                while True:  # loop 1 (P1M)
                    # compute d
                    d = np.zeros(N)
                    for m in range(N):
                        d[m] = np.linalg.norm(data[m, :] - v) ** 2

                    # compute u(v,X) typicalities
                    u = np.zeros(N)
                    for m in range(N):
                        u[m] = 1 / (1 + (d[m] / eta[itr - 1]) ** (1 / (fzr - 1)))

                    # compute v(u,X) / for 3 dimensions (cluster centers)
                    dim = data.shape[1]
                    v_up = np.zeros(dim)
                    v_down = 0

                    for m in range(N):
                        v_up += u[m] ** fzr * data[m, :]
                        v_down += u[m] ** fzr

                    v_new = v_up / v_down
                    v_diff = np.linalg.norm(v - v_new) ** 2
                    v = v_new.copy()

                    # if cluster center doesn't move, break the while loop
                    if v_diff < epsilon ** 2:
                        break

                # termination calculation
                stop_count += 1
                for m in range(N):
                    if itr > 1 and max(Uall[:, m]) > 0.5:
                        U_count[itr - 1] += 1

                if stop_count > N * 0.9 - U_count[itr - 1]:
                    stop_code = 1
                    break

                # remove the coincident cluster center
                vw = np.zeros(itr - 1)
                if itr > 1:
                    for j in range(itr - 1):
                        vw[j] = np.linalg.norm(v - Vall[j, :]) ** 2
                    vw_min = np.min(vw)
                    if vw_min > (np.sum(eta.iloc[:itr]) / itr) ** 1.2:
                        break
                else:
                    break

                U_count[itr - 1] = 0

            if stop_code == 1:
                stop_code = 0
                break

            Uall.append(u)  # append Uall with u indexed by itr
            Vall.append(v)

            points = []
            stop_count = 0

            for s in range(N):
                if u[s] > 0.2:
                    points.append(data[s, :])
                    label[s] = itr

            point_num.append(len(points))
            cov_tmp = np.cov(np.array(points).T)
            cov_max.append(cov_tmp)


        mean = np.array(Vall)
        c_num = mean.shape[0]

        anormaly = np.empty((0,dim))
        normaly = np.empty((0,dim))

        cov_max = np.array(cov_max)

        for itr in range(N):
            u_max = Uall[0][itr]

            if u_max < 0.5:
                anormaly = np.append(anormaly,np.array([data[itr,:]]), axis=0)
            else:
                normaly = np.append(normaly,np.array([data[itr,:]]), axis=0)

        model = { 'mean': mean,
                 'cov_max': cov_max,
                 'c_num': c_num,
                 'point_num': point_num,
                 'label': label }

        return model, anormaly
    
    def process(self, data):
        """run SPGMM on data
        
        :param data: numerical feature data set
        :type data: pandas, will document more, come back to this
        """
        
        model, anormaly = self.sp1m(data)
        return model, anormaly
    
class process_stream():
    
    def __init__(self, model, anormaly,dates):
        self.model = model
        self.anormaly = anormaly
        self.dates = dates
        
    def cal_dis(self, point, mean, cov_max):
        c = mean.shape[0]
        eta = 3.8
        distance = np.zeros(c)

        for i in range(c):
            distance[i] = np.linalg.norm(point - mean[i, :]) # euclidean l2 norm is default

        mahal_min = np.min(distance)
        win_index = np.argmin(distance) # get index of minimum values
        typicality = 1 / (1 + (mahal_min / eta) ** 2)
        inPrototype = mahal_min < eta

        return inPrototype, win_index, typicality

    def change_label(self, index, new_label, label, win_index):
        N = len(label)
        for i in range(N):
            if label[i] == 0:
                index -= 1
            if index == 0:
                new_label[i] = win_index
                return new_label
        return new_label

    def early_pred_avg3(self, early_pred, index):
        if index - 2 < 1:
            early_pred_avg3_value = early_pred[index] # Note the -1 as Python is 0-based
            return early_pred_avg3_value

        early_pred_avg3_value = 0
        for i in range(index, index-3, -1): #adusted window from -2 to -3
            early_pred_avg3_value += early_pred[i] # Note the -1 as Python is 0-based

        early_pred_avg3_value = early_pred_avg3_value / 3
        return early_pred_avg3_value

    def early_check(self, stream, early_pred, idx, typicality, win_mean, trend):
        # all of these parameters effect the generating of alerts and need to be validated against ground truth
        # need to configure processing to make them configurable from lib function call
        lowPriority = 4 # adjusted
        mediumPriority = 5 #adjusted 
        highPriority = 7 #adjusted 
        starter = highPriority #adjusted 
        early_pred[idx] = typicality # Python indexing starts at 0
        priority = 0
        window = 5
        epsilon = 0.002 # threshold

        if idx <= starter-1: # adjusted from starter+1 in matlab
            early_change = False
            return early_pred, early_change, priority, trend

        cur_point = stream[idx, :]

        vec1s = np.zeros((window, stream.shape[1]))

        for i in range(window):
            vec1s[i, :] = stream[idx + 1 - (i+1), :] - stream[idx - (i+1), :]

        vec1 = np.sum(vec1s, axis=0) / window 
        vec2 = win_mean - cur_point

        ca_numer = np.dot(vec1, vec2.T) 
        ca_denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_alpha = ca_numer / ca_denom 
        trend[idx] = cos_alpha 

        # max typicality check        
        for i in range(idx, idx - lowPriority, -1): 
            if (self.early_pred_avg3(early_pred, i) - self.early_pred_avg3(early_pred, i - 1)) > epsilon:
                early_change = False
                return early_pred, early_change, priority, trend

        for i in range(idx, idx - mediumPriority, -1): 
            if (self.early_pred_avg3(early_pred, i) - self.early_pred_avg3(early_pred, i - 1)) > epsilon:
                early_change = True
                priority = 1 # weak warning
                return early_pred, early_change, priority, trend
        
        for i in range(idx, idx - highPriority, -1): 
            if (self.early_pred_avg3(early_pred, i) - self.early_pred_avg3(early_pred, i - 1)) > epsilon:
                early_change = True
                priority = 2 # medium warning
                return early_pred, early_change, priority, trend

        early_change = True 
        priority = 3 # strong warning
        return early_pred, early_change, priority, trend
    
    
    def check_newcluster(self, anormaly, point, mean, cov_max, c_num, point_num, label):
        # need to make parameters here in the lib function call
        M = 35 # min num outliers needed to generate new cluster
        newFound = False
        eta = 3 # threshold compare against distance for outliers to be added to new cluster
        dim = mean.shape[1] # n columns

        if anormaly.ndim == 1:
            anormaly = anormaly.reshape(1, -1)
        rows = anormaly.shape[0]

        while True:
            count = 1
            new_sum = np.zeros((1,dim))
            new_cluster = np.empty((0,dim))

            for i in range(anormaly.shape[0]):
                anorm = anormaly[i]
                dist = np.linalg.norm(point - anorm)
                if dist < eta: # if current point is close to anormal add to new cluster
                    new_sum    += anorm
                    new_cluster = np.append(new_cluster,np.array([anorm]), axis=0)
                    count      += 1

            new_mean = new_sum / (count - 1)
            if np.linalg.norm(point - new_mean) < 1 or np.sum(new_sum) == 0: # iter until distance is below 1 or 0
                break
            point = new_mean

        new_anormaly = anormaly

        new_label = label.copy()

        if count > M: #at least 35 anormaly points for new cluster
            newFound   = True
            cov_new    = np.cov(np.array(new_cluster).T)
            cov_new    = cov_new[np.newaxis, :, :]
            mean = np.vstack((mean, new_mean))
            cov_max = np.concatenate((cov_max, cov_new), axis=0)
            c_num     += 1
            point_num  = np.append(point_num, count)
            new_anormaly = np.empty((0,dim))

            for i in range(rows):
                anorm = anormaly[i]

                if np.linalg.norm(new_mean - anorm) > eta * 1.5: # i dont know where 1.5 came from - NM
                    new_anormaly = np.append(new_anormaly,np.array([anorm]), axis=0)

                else:
                    new_label = self.change_label(i, new_label, label, c_num)

        return mean, cov_max, c_num, point_num, newFound, new_anormaly, new_label
    
    def check_anormaly(self, anormaly, mean, cov_max, point_num, label):
        eta = 2 # threshold to compare against the minimum distance to a cluster for the given point to be an outlier
        dim = mean.shape[1]
        new_anormaly = np.empty((0,dim))
        new_label = label.copy()

        for i in range(anormaly.shape[0]):
            dist = np.zeros(mean.shape[0])
            for j in range(mean.shape[0]):
                dist[j] = np.linalg.norm(mean[j, :] - anormaly[i, :])

            win_index = np.argmin(dist)
            min_dist = dist[win_index]

            if min_dist < eta:
                # Update the winning mean and covariance
                point_num[win_index] += 1
                cov_max[win_index,:, :] = (
                    (point_num[win_index] - 1) * cov_max[win_index,:, :] +
                    np.outer(anormaly[i, :] - mean[win_index, :], 
                             anormaly[i, :] - mean[win_index, :])
                    ) / point_num[win_index]

                mean[win_index, :] = (mean[win_index, :] + 
                                      (anormaly[i, :] - mean[win_index, :]) / 
                                      point_num[win_index])

                new_label = self.change_label(i, new_label, label, win_index)
            else:
                new_anormaly = np.append(new_anormaly,np.array([anormaly[i]]), axis=0)
    
        return new_anormaly, mean, cov_max, point_num, new_label
    
    def sp1ms(self, stream):
        INIT_DAYS = 30
        anormaly = self.anormaly
        days = self.dates
        mean = self.model['mean']
        cov_max = self.model['cov_max']
        c_num = self.model['c_num']
        point_num = self.model['point_num']
        label = self.model['label']
        model = {}

        early_pred = np.zeros(stream.shape[0])
        early_pred_avg = np.zeros(stream.shape[0])
        trend = np.zeros(stream.shape[0])
        
        warnings = pd.DataFrame(columns=['date', 'priority'])

        for i in range(stream.shape[0]): #iter through stream
            inPrototype, win_index, typicality = self.cal_dis(stream[i, :], mean, cov_max)

            if inPrototype:
                early_pred, early_change, priority, trend = self.early_check(
                    stream, early_pred, i, typicality, mean[win_index, :], trend)

                early_pred_avg[i] = self.early_pred_avg3(early_pred, i)

                if (early_change == True) & (trend[i] < 0):
                    warning = {'date': days[INIT_DAYS+i], 'priority': priority}
                    warnings = warnings.append(warning, ignore_index=True)
                    if priority ==1:
                        print('weak warning',days[INIT_DAYS+i])
                        warnings
                    elif priority ==2:
                        print('medium warning',days[INIT_DAYS+i])
                    elif priority ==3:
                        print('strong warning',days[INIT_DAYS+i])

                point_num[win_index] += 1

                cov_max[win_index,:,:] = (
                    (point_num[win_index] - 1) * 
                    cov_max[win_index,:,:] + 
                    np.outer(stream[i, :] - mean[win_index, :],
                             stream[i, :] - mean[win_index, :])
                    ) / point_num[win_index]

                mean[win_index, :] = (
                    mean[win_index, :] + 
                    (stream[i, :] - mean[win_index, :]) / 
                    point_num[win_index])

                label = np.append(label, win_index+1)

                anormaly, mean, cov_max, point_num, label = self.check_anormaly(
                    anormaly, mean, cov_max, point_num, label)

            else:

                early_pred[i] = typicality
                early_pred_avg[i] = self.early_pred_avg3(early_pred, i)

                if i < 5: # first 5 points, dont compute trend
                    trend[i] = 0
                else:
                    pre_points = stream[i:i-5:-1, :] 
                    vec1s = stream[i, :] - pre_points 
                    vec1 = np.sum(vec1s, axis=0) / 5 
                    vec2 = mean[win_index, :] - stream[i, :]
                    cos_alpha = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) 
                    trend[i] = cos_alpha

                mean, cov_max, c_num, point_num, newFound, anormaly, label = self.check_newcluster(
                    anormaly, stream[i, :], mean, cov_max,c_num, point_num, label)

                if not newFound:            
                    anormaly = np.append(anormaly,np.array([stream[i, :]]), axis=0) 
                    label = np.append(label, 0)

                else:
                    label = np.append(label, c_num)

        model['mean'] = mean
        model['cov_max'] = cov_max
        model['c_num'] = c_num
        model['point_num'] = point_num
        model['label'] = label

        return model, anormaly, warnings

    
    def process(self, data):
        """run SPGMM on data
        
        :param data: numerical feature data set
        :type data: pandas, will document more, come back to this
        """
        
        model, anormaly, warnings = self.sp1ms(data)
        return model, anormaly, warnings
