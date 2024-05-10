import matplotlib.pyplot as plt


def plot_susceptibility(suscep,sym,labels,show=False,save=False,path=False):
    plt.figure(figsize=(3,4))
    plt.plot(suscep[0],label='Re(χ)')
    plt.plot(suscep[1],label='Im(χ)')
    plt.xticks(ticks=sym, labels=labels, fontsize=15)
    plt.xlim(sym[0], sym[-1])
    for i in sym[1:-1]:
        plt.axvline(i, c="black")
    plt.ylabel("1/eV")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig('susceptibility.pdf')
    elif path:
        plt.savefig(f'{path}/susceptibility.pdf')
        

        