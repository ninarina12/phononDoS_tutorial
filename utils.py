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
import math
import time
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
    
    print('parsing cif files ...')
    df['structure'] = df['structure'].apply(eval).progress_map(lambda x: Atoms.fromdict(x))
    
    df['phfreq'] = df['phfreq'].apply(eval).apply(np.array)
    df['phdos'] = df['phdos'].apply(eval).apply(np.array)
    df['pdos'] = df['pdos'].apply(eval)
    
    df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
    df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
    species = sorted(list(set(df['species'].sum())))
    return df, species


def train_valid_test_split(df, species, valid_size, test_size, seed=12, plot=False):
    # perform an element-balanced train/valid/teset split
    print('split train/dev ...')
    dev_size = valid_size + test_size
    stats = get_element_statistics(df, species)
    idx_train, idx_dev = split_data(stats, dev_size, seed)
    
    print('split valid/test ...')
    stats_dev = get_element_statistics(df.iloc[idx_dev], species)
    idx_valid, idx_test = split_data(stats_dev, test_size/dev_size, seed)
    
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
    fig, ax = plt.subplots(1,2, figsize=(14,10))
    atoms = Atoms(symbols=entry.symbol, positions=entry.pos.numpy(), cell=entry.lattice.squeeze().numpy(), pbc=True)
    symbols = np.unique(entry.symbol)
    z = dict(zip(symbols, range(len(symbols))))
    norm = plt.Normalize(vmin=0, vmax=len(symbols)-1)
    color = [mpl.colors.to_hex(k) for k in cmap(norm([z[j] for j in entry.symbol]))]
    plot_atoms(atoms, ax[0], radii=0.25, colors=color, rotation=('0x,90y,0z'))

    # plot graph
    nx.draw_networkx(g, ax=ax[1], labels=node_labels, pos=pos, font_family='lato', node_size=500, node_color=color)
    
    if label_edges:
        nx.draw_networkx_edge_labels(g, ax=ax[1], edge_labels=edge_labels, pos=pos, label_pos=0.3, font_family='lato')
    
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


def visualize_layers(model):
    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['gate', 'tp', 'lin2', 'gate']))
    num_layers = len(model.mp.layers)
    num_ops = max([len([k for k in list(model.mp.layers[i].first._modules.keys()) if k not in ['fc', 'alpha']])
                   for i in range(num_layers-1)])

    fig, ax = plt.subplots(num_layers, num_ops, figsize=(14,3.5*num_layers))
    for i in range(num_layers - 1):
        ops = model.mp.layers[i].first._modules.copy()
        ops.pop('fc', None); ops.pop('alpha', None)
        for j, (k, v) in enumerate(ops.items()):
            ax[i,j].set_title(k, fontsize=textsize)
            v.cpu().visualize(ax=ax[i,j])
            ax[i,j].text(0.7,-0.15,'--> to ' + layer_dst[k], fontsize=textsize-2, transform=ax[i,j].transAxes)

    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['gate', 'tp', 'lin2', 'output']))
    ops = model.mp.layers[-1]._modules.copy()
    ops.pop('fc', None); ops.pop('alpha', None)
    for j, (k, v) in enumerate(ops.items()):
        ax[-1,j].set_title(k, fontsize=textsize)
        v.cpu().visualize(ax=ax[-1,j])
        ax[-1,j].text(0.7,-0.15,'--> to ' + layer_dst[k], fontsize=textsize-2, transform=ax[-1,j].transAxes)

    fig.subplots_adjust(wspace=0.3, hspace=0.5)


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))

        
def evaluate(model, dataloader, loss_fn, loss_fn_mae, device):
    model.eval()
    loss_cumulative = 0.
    loss_cumulative_mae = 0.
    start_time = time.time()
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(d)
            loss = loss_fn(output, d.phdos).cpu()
            loss_mae = loss_fn_mae(output, d.phdos).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()
    return loss_cumulative/len(dataloader), loss_cumulative_mae/len(dataloader)


def train(model, optimizer, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, run_name,
          max_iter=101, scheduler=None, device="cpu"):
    model.to(device)

    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()


    try: results = torch.load(run_name + '.torch')
    except:
        history = []
        results = {}
        s0 = 0
    else:
        model.load_state_dict(torch.load(run_name + '.torch')['state'])
        history = results['history']
        s0 = history[-1]['step'] + 1


    for step in range(max_iter):
        model.train()
        loss_cumulative = 0.
        loss_cumulative_mae = 0.
        
        for j, d in tqdm(enumerate(dataloader_train), total=len(dataloader_train), bar_format=bar_format):
            d.to(device)
            output = model(d)
            loss = loss_fn(output, d.phdos).cpu()
            loss_mae = loss_fn_mae(output, d.phdos).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        wall = end_time - start_time

        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            valid_avg_loss = evaluate(model, dataloader_valid, loss_fn, loss_fn_mae, device)
            train_avg_loss = evaluate(model, dataloader_train, loss_fn, loss_fn_mae, device)

            history.append({
                'step': s0 + step,
                'wall': wall,
                'batch': {
                    'loss': loss.item(),
                    'mean_abs': loss_mae.item(),
                },
                'valid': {
                    'loss': valid_avg_loss[0],
                    'mean_abs': valid_avg_loss[1],
                },
                'train': {
                    'loss': train_avg_loss[0],
                    'mean_abs': train_avg_loss[1],
                },
            })

            results = {
                'history': history,
                'state': model.state_dict()
            }

            print(f"Iteration {step+1:4d}   " +
                  f"train loss = {train_avg_loss[0]:8.4f}   " +
                  f"valid loss = {valid_avg_loss[0]:8.4f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")

            with open(run_name + '.torch', 'wb') as f:
                torch.save(results, f)

        if scheduler is not None:
            scheduler.step()


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


def plot_partial_predictions(model, df, idx, device='cpu'):
    # randomly sample r compounds from the dataset
    r = 6
    ids = np.random.choice(df.iloc[idx][df.iloc[idx]['pdos'].str.len()>0].index.tolist(), size=r)
    
    # initialize figure axes
    N = df.iloc[ids]['species'].str.len().max()
    fig, ax = plt.subplots(r, N+1, figsize=(2.8*(N+1),1.2*r), sharex=True, sharey=True)

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

        # plot partial DoS
        for j, s in enumerate(entry.species):
            ax[row,j+1].plot(entry.phfreq, entry.pdos[s], color='black')
            ax[row,j+1].plot(entry.phfreq, pdos[s], color=palette[1], lw=2)
            ax[row,j+1].set_title(s, fontsize=fontsize - 2, y=0.99)
        
        for j in range(len(entry.species) + 1, N+1):
            ax[row,j].remove()

    fig.supylabel('$I/I_{max}$', fontsize=fontsize, x=0.07)
    fig.supxlabel(r'$\omega\ (cm^{-1})$', fontsize=fontsize, y=0.04)
    fig.subplots_adjust(hspace=0.5)