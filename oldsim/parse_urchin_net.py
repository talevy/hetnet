
import numpy as np
import pickle
import operator

    
def get_net(gvd):
    net = []
    names = {}
    # pass one get names
    with open(gvd) as f:
        inx = 0
        for line in f:
            if 'node ' in line:
                names[line[5:-2]] = inx
                inx += 1
    # build network
    with open(gvd) as f:
        cur_gene = -1
        for line in f:
            if 'node ' in line:
                cur_gene += 1
                net.append({'name': line[5:].strip(), 'clusters':[], 'adj':[]})
            elif 'cluster ' in line:
                net[cur_gene]['clusters'].append(line[8:].strip())
            elif 'edge ' in line:
                net[cur_gene]['adj'].append(names[line[5:].strip()])
    print sorted(names.iteritems(), key=operator.itemgetter(1))
    return net

def to_matrix(net):
    n = len(net)
    mat = np.matrix([[0]*n]*n)
    for i, gene in enumerate(net):
        for j in gene['adj']:
            mat[j, i] = 1
    return mat
    
if __name__ == '__main__':
    net = get_net(gvd = "annotated_sea_urchin_net.gvd")
    mat = to_matrix(net)
    #output = open("urchin.pkl", "wb")
    #pickle.dump(mat, output)
    #output.close()


