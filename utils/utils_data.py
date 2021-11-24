# model
import torch
import torch_geometric as tg

# data pre-processing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde

# data visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from ase import Atoms
from ase.visualize.plot import plot_atoms

# utilities
from tqdm import tqdm


# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)


# standard formatting for plots
fontsize = 16
textsize = 14
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plt.rcParams['font.family'] = 'lato'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = textsize


# colors for datasets
palette = ['#2876B2', '#F39957', '#67C7C2', '#C86646']
datasets = ['train', 'valid', 'test']
colors = dict(zip(datasets, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])


def load_data(filename):
    # load data from a csv file and derive formula and species columns from structure
    df = pd.read_csv(filename)
    
    try:
        # structure provided as Atoms object
        df['structure'] = df['structure'].apply(eval).progress_map(lambda x: Atoms.fromdict(x))
    
    except:
        # no structure provided
        species = []

    else:
        df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
        df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
        species = sorted(list(set(df['species'].sum())))

    df['phfreq'] = df['phfreq'].apply(eval).apply(np.array)
    df['phdos'] = df['phdos'].apply(eval).apply(np.array)
    df['pdos'] = df['pdos'].apply(eval)

    return df, species


def train_valid_test_split(df, species, valid_size, test_size, seed=12, plot=False):
    # perform an element-balanced train/valid/test split
    print('split train/dev ...')
    dev_size = valid_size + test_size
    stats = get_element_statistics(df, species)
    idx_train, idx_dev = split_data(stats, dev_size, seed)
    
    print('split valid/test ...')
    stats_dev = get_element_statistics(df.iloc[idx_dev], species)
    idx_valid, idx_test = split_data(stats_dev, test_size/dev_size, seed)
    idx_train += df[~df.index.isin(idx_train + idx_valid + idx_test)].index.tolist()

    print('number of training examples:', len(idx_train))
    print('number of validation examples:', len(idx_valid))
    print('number of testing examples:', len(idx_test))
    print('total number of examples:', len(idx_train + idx_valid + idx_test))
    assert len(set.intersection(*map(set, [idx_train, idx_valid, idx_test]))) == 0

    if plot:
        # plot element representation in each dataset
        stats['train'] = stats['data'].map(lambda x: element_representation(x, np.sort(idx_train)))
        stats['valid'] = stats['data'].map(lambda x: element_representation(x, np.sort(idx_valid)))
        stats['test'] = stats['data'].map(lambda x: element_representation(x, np.sort(idx_test)))
        stats = stats.sort_values('symbol')

        fig, ax = plt.subplots(2,1, figsize=(14,7))
        b0, b1 = 0., 0.
        for i, dataset in enumerate(datasets):
            split_subplot(ax[0], stats[:len(stats)//2], species[:len(stats)//2], dataset, bottom=b0, legend=True)
            split_subplot(ax[1], stats[len(stats)//2:], species[len(stats)//2:], dataset, bottom=b1)

            b0 += stats.iloc[:len(stats)//2][dataset].values
            b1 += stats.iloc[len(stats)//2:][dataset].values

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1)

    return idx_train, idx_valid, idx_test


def get_element_statistics(df, species):    
    # create dictionary indexed by element names storing index of samples containing given element
    species_dict = {k: [] for k in species}
    for entry in df.itertuples():
        for specie in entry.species:
            species_dict[specie].append(entry.Index)

    # create dataframe of element statistics
    stats = pd.DataFrame({'symbol': species})
    stats['data'] = stats['symbol'].astype('object')
    for specie in species:
        stats.at[stats.index[stats['symbol'] == specie].values[0], 'data'] = species_dict[specie]
    stats['count'] = stats['data'].apply(len)

    return stats


def split_data(df, test_size, seed):
    # initialize output arrays
    idx_train, idx_test = [], []
    
    # remove empty examples
    df = df[df['data'].str.len()>0]
    
    # sort df in order of fewest to most examples
    df = df.sort_values('count')
    
    for _, entry in tqdm(df.iterrows(), total=len(df), bar_format=bar_format):
        df_specie = entry.to_frame().T.explode('data')

        try:
            idx_train_s, idx_test_s = train_test_split(df_specie['data'].values, test_size=test_size,
                                                       random_state=seed)
        except:
            # too few examples to perform split - these examples will be assigned based on other constituent elements
            # (assuming not elemental examples)
            pass

        else:
            # add new examples that do not exist in previous lists
            idx_train += [k for k in idx_train_s if k not in idx_train + idx_test]
            idx_test += [k for k in idx_test_s if k not in idx_train + idx_test]
    
    return idx_train, idx_test


def element_representation(x, idx):
    # get fraction of samples containing given element in dataset
    return len([k for k in x if k in idx])/len(x)


def split_subplot(ax, df, species, dataset, bottom=0., legend=False):    
    # plot element representation
    width = 0.4
    color = [int(colors[dataset].lstrip('#')[i:i+2], 16)/255. for i in (0,2,4)]
    bx = np.arange(len(species))
        
    ax.bar(bx, df[dataset], width, fc=color+[0.7], ec=color, lw=1.5, bottom=bottom, label=dataset)
        
    ax.set_xticks(bx)
    ax.set_xticklabels(species)
    ax.tick_params(direction='in', length=0, width=1)
    ax.set_ylim(top=1.18)
    if legend: ax.legend(frameon=False, ncol=3, loc='upper left')
        

def plot_example(df, i=12, label_edges=False):
    # plot an example crystal structure and graph
    entry = df.iloc[i]['data']

    # get graph with node and edge attributes
    g = tg.utils.to_networkx(entry, node_attrs=['symbol'], edge_attrs=['edge_len'], to_undirected=True)

    # remove self-loop edges for plotting
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    node_labels = dict(zip([k[0] for k in g.nodes.data()], [k[1]['symbol'] for k in g.nodes.data()]))
    edge_labels = dict(zip([(k[0], k[1]) for k in g.edges.data()], [k[2]['edge_len'] for k in g.edges.data()]))

    # project positions of nodes to 2D for plotting
    pos = dict(zip(list(g.nodes), [np.roll(k,2)[:-1][::-1] for k in entry.pos.numpy()]))

    # plot unit cell
    fig, ax = plt.subplots(1,2, figsize=(14,10), gridspec_kw={'width_ratios': [2,3]})
    atoms = Atoms(symbols=entry.symbol, positions=entry.pos.numpy(), cell=entry.lattice.squeeze().numpy(), pbc=True)
    symbols = np.unique(entry.symbol)
    z = dict(zip(symbols, range(len(symbols))))
    norm = plt.Normalize(vmin=0, vmax=len(symbols)-1)
    color = [mpl.colors.to_hex(k) for k in cmap(norm([z[j] for j in entry.symbol]))]
    plot_atoms(atoms, ax[0], radii=0.25, colors=color, rotation=('0x,90y,0z'))

    # plot graph
    nx.draw_networkx(g, ax=ax[1], labels=node_labels, pos=pos, font_family='lato', node_size=500, node_color=color,
                     edge_color='gray')
    
    if label_edges:
        nx.draw_networkx_edge_labels(g, ax=ax[1], edge_labels=edge_labels, pos=pos, label_pos=0.5, font_family='lato')
    
    # format axes
    ax[0].set_xlabel(r'$x_1\ (\AA)$')
    ax[0].set_ylabel(r'$x_2\ (\AA)$')
    ax[0].set_title('Crystal structure', fontsize=fontsize)
    ax[1].set_aspect('equal')
    ax[1].axis('off')
    ax[1].set_title('Crystal graph', fontsize=fontsize)
    pad = np.array([-0.5, 0.5])
    ax[1].set_xlim(np.array(ax[1].get_xlim()) + pad)
    ax[1].set_ylim(np.array(ax[1].get_ylim()) + pad)
    fig.subplots_adjust(wspace=0.4)

def plot_predictions(df, idx, title=None):    
    # get quartiles
    i_mse = np.argsort(df.iloc[idx]['mse'])
    ds = df.iloc[idx].iloc[i_mse][['formula', 'phdos', 'phdos_pred', 'mse']].reset_index(drop=True)
    quartiles = np.quantile(ds['mse'].values, (0.25, 0.5, 0.75, 1.))
    iq = [0] + [np.argmin(np.abs(ds['mse'].values - k)) for k in quartiles]
    
    n = 7
    s = np.concatenate([np.sort(np.random.randint(iq[k-1], iq[k], size=n)) for k in range(1,5)])
    x = df.iloc[0]['phfreq']

    fig, axs = plt.subplots(4,n+1, figsize=(13,3.5), gridspec_kw={'width_ratios': [0.7] + [1]*n})
    gs = axs[0,0].get_gridspec()
    
    # remove the underlying axes
    for ax in axs[:,0]:
        ax.remove()

    # add long axis
    axl = fig.add_subplot(gs[:,0])

    # plot quartile distribution
    y_min, y_max = ds['mse'].min(), ds['mse'].max()
    y = np.linspace(y_min, y_max, 500)
    kde = gaussian_kde(ds['mse'])
    p = kde.pdf(y)
    axl.plot(p, y, color='black')
    cols = [palette[k] for k in [2,0,1,3]][::-1]
    qs =  list(quartiles)[::-1] + [0]
    for i in range(len(qs)-1):
        axl.fill_between([p.min(), p.max()], y1=[qs[i], qs[i]], y2=[qs[i+1], qs[i+1]], color=cols[i], lw=0, alpha=0.5)
    axl.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axl.invert_yaxis()
    axl.set_xticks([])
    axl.set_ylabel('MSE')

    fontsize = 12
    cols = np.repeat(cols[::-1], n)
    axs = axs[:,1:].ravel()
    for k in range(4*n):
        ax = axs[k]
        i = s[k]
        ax.plot(x, ds.iloc[i]['phdos'], color='black')
        ax.plot(x, ds.iloc[i]['phdos_pred'], color=cols[k])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ds.iloc[i]['formula'].translate(sub), fontsize=fontsize, y=0.95)
        
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    if title: fig.suptitle(title, ha='center', y=1., fontsize=fontsize + 4)


def plot_partials(model, df, idx, device='cpu'):
    # randomly sample r compounds from the dataset
    r = 6
    ids = np.random.choice(df.iloc[idx][df.iloc[idx]['pdos'].str.len()>0].index.tolist(), size=r)
    
    # initialize figure axes
    N = df.iloc[ids]['species'].str.len().max()
    fig, ax = plt.subplots(r, N+1, figsize=(2.4*(N+1),1.2*r), sharex=True, sharey=True)

    # predict output of each site for each sample
    for row, i in enumerate(ids):
        entry = df.iloc[i]
        d = tg.data.Batch.from_data_list([entry.data])

        model.eval()
        with torch.no_grad():
            d.to(device)
            output = model(d).cpu().numpy()

        # average contributions from the same specie over all sites
        n = len(entry.species)
        pdos = dict(zip(entry.species, [np.zeros((output.shape[1])) for k in range(n)]))
        for j in range(output.shape[0]):
            pdos[entry.data.symbol[j]] += output[j,:]

        for j, s in enumerate(entry.species):
            pdos[s] /= entry.data.symbol.count(s)

        # plot total DoS
        ax[row,0].plot(entry.phfreq, entry.phdos, color='black')
        ax[row,0].plot(entry.phfreq, entry.phdos_pred, color=palette[0])
        ax[row,0].set_title(entry.formula.translate(sub), fontsize=fontsize - 2, y=0.99)
        ax[row,0].set_xticks([]); ax[row,0].set_yticks([])

        # plot partial DoS
        for j, s in enumerate(entry.species):
            ax[row,j+1].plot(entry.phfreq, entry.pdos[s], color='black')
            ax[row,j+1].plot(entry.phfreq, pdos[s]/pdos[s].max(), color=palette[1], lw=2)
            ax[row,j+1].set_title(s, fontsize=fontsize - 2, y=0.99)
            ax[row,j+1].set_xticks([]); ax[row,j+1].set_yticks([])

        for j in range(len(entry.species) + 1, N+1):
            ax[row,j].remove()

    try: fig.supylabel('Intensity', fontsize=fontsize, x=0.08)
    except: pass
    else: fig.supxlabel('Frequency', fontsize=fontsize, y=0.06)
    fig.subplots_adjust(hspace=0.8)
    fig.savefig('predictions_partials.svg', bbox_inches='tight')