from math import log, sqrt

def squaredDistance(vec1, vec2):
    sum = 0 
    dim = len(vec1)  
    for i in range(dim):
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]) 
    
    return sum



def count_occurrence(list):
    d = {}
    for i in list:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d

def cal_entropy(assignment):

    # count the # points in each cluster
    occ = count_occurrence(assignment) 
    n = float(len(assignment)) 
    h = 0 
    for id in occ:
        p = occ[id] / n # the probability of cluster C_id
        if p != 0:
            h += p*log(p)
    return -h

def purity(groundtruthAssignment, algorithmAssignment):
    purity = 0
    ids = sorted(set(algorithmAssignment)) # sorted unique clusterID
    matching = 0
    for id in ids:
        # indices of points that belong to cluster id
        indices = [i for i, j in enumerate(algorithmAssignment) if j == id]
        # the true assignment of these points
        cluster = [groundtruthAssignment[i] for i in indices]
        occ = count_occurrence(cluster)
        matching += max(occ.values())
    purity =  matching / float(len(groundtruthAssignment))
    return purity 


def NMI(groundtruthAssignment, algorithmAssignment):
    NMI = 0
    h_c = cal_entropy(algorithmAssignment) # Entropy of clustering C
    h_t = cal_entropy(groundtruthAssignment) # Entropy of partitioning T

    ## compute Mutual information
    occ_c = count_occurrence(algorithmAssignment) # get occurrence: for the probability of cluster C_id
    n_c = float(sum(occ_c.values())) # total # of cluster C_id
    occ_t = count_occurrence(groundtruthAssignment) # get occurrence: for the probability of cluster T_id
    n_t = float(sum(occ_t.values())) # total # of cluster T_id
    ids_c = sorted(set(algorithmAssignment))
    ids_t = sorted(set(groundtruthAssignment))

    # cartesian product for all possible id combination
    cp = [(i,j) for i in ids_c for j in ids_t]
    p = dict(zip(cp,[0]*len(cp)))

    for (i,j) in zip(algorithmAssignment,groundtruthAssignment):
        p[(i,j)] += 1

    mi = 0 # mutual information
    for c in ids_c:
        for t in ids_t:
            if p[(c,t)] != 0:
                mi += (p[(c,t)]/n_c) * log( (p[(c,t)]/n_c) / ((occ_c[c]/n_c)*(occ_t[t]/n_t)) )
    NMI = mi / sqrt(float(h_c*h_t))
    return NMI
