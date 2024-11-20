def get_nk(gkk_path):
    with open(gkk_path) as f:
        for i in range(1000):
            a = f.readline(i)
            # if a.split(' ')[0]=="Size":
            line = " ".join(a.split()).split(' ')
            if line[0]=='Size':
                if line[2]=='k':
                    nk=line[7]
    return int(nk)
def get_nq(gkk_path):
    with open(gkk_path) as f:
        for i in range(1000):
            a = f.readline(i)
            # if a.split(' ')[0]=="Size":
            line = " ".join(a.split()).split(' ')
            if line[0]=='Size':
                if line[2]=='q':
                    nq =line[7]
    return int(nq)
def get_ef(gkk_path):
    with open(gkk_path) as f:
        for i in range(1000):
            a = f.readline(i)
            # if a.split(' ')[0]=="Size":
            line = " ".join(a.split()).split(' ')
            if line[0]=='Fermi':
                ef = line[-2]
    return float(ef)
def get_nph(gkk_path):
    counter=0
    with open(gkk_path) as f:
        counter=0
        for i, line in enumerate(f):
            a = " ".join(line.split()).split(' ')
            # print(i,a)
            if a[0]=='ik':
                if counter==1:
                    num =i
                    # print(a)
                counter+=1
            # print(a)
            # print(num)
            if i>1000:
                break
    with open(gkk_path) as f:
        for i, line in enumerate(f):
            if i==num-3:
                a = " ".join(line.split()).split(' ')
                nph = a[2]
            if i>1000:
                break
    return(int(nph))


