# model
import torch
import torch_geometric as tg

# crystal structure data
from pymatgen.io.cif import CifParser

# data pre-processing
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split

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
palette = ['#2876B2', '#F39957', '#67C7C2']
datasets = ['train', 'valid', 'test']
colors = dict(zip(datasets, palette))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])


def load_data(filename):
    # load data from a csv file and derive formula and species columns from structure
    df = pd.read_csv(filename)
    
    print('parsing cif files ...')
    df['structure'] = df['structure'].progress_map(lambda x: CifParser.from_string(x).get_structures()[0])
    
    df['phfreq'] = df['phfreq'].apply(eval).apply(np.array)
    df['phdos'] = df['phdos'].apply(eval).apply(np.array)
    
    df['phfreq_orig'] = df['phfreq_orig'].apply(eval).apply(np.array)
    df['phdos_orig'] = df['phdos_orig'].apply(eval).apply(np.array)
    
    df['formula'] = df['structure'].map(lambda x: x.formula.replace(' ', ''))
    df['species'] = df['formula'].map(lambda x: re.split(r'(\d*\.?\d+)', x)[:-1:2])
    species = sorted(list(set(df['species'].sum())))
    return df, species


def train_valid_test_split(df, species, seed=12, plot=False):
    # perform an element-balanced train/valid/teset split
    print('split train/dev ...')
    stats = get_element_statistics(df, species)
    idx_train, idx_dev = split_data(stats, 0.2, seed)
    
    print('split valid/test ...')
    stats_dev = get_element_statistics(df.iloc[idx_dev], species)
    idx_valid, idx_test = split_data(stats_dev, 0.5, seed)
    
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
    fig, ax = plt.subplots(1,2, figsize=(14,14))

    # get graph with node and edge attributes
    g = tg.utils.to_networkx(entry, node_attrs=['symbol'], edge_attrs=['edge_len'], to_undirected=True)
    node_labels = dict(zip([k[0] for k in g.nodes.data()], [k[1]['symbol'] for k in g.nodes.data()]))
    edge_labels = dict(zip([(k[0], k[1]) for k in g.edges.data()], [k[2]['edge_len'] for k in g.edges.data()]))

    # project positions of nodes to 2D for plotting
    pos = dict(zip(list(g.nodes), [np.roll(k,2)[:-1][::-1] for k in entry.pos.numpy()]))

    # plot unit cell
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
    ax[0].set_title('Crystal structure')
    ax[1].set_aspect('equal')
    ax[1].axis('off')
    ax[1].set_title('Crystal graph')
    pad = np.array([-0.5, 0.5])
    ax[1].set_xlim(np.array(ax[1].get_xlim()) + pad)
    ax[1].set_ylim(np.array(ax[1].get_ylim()) + pad)
    fig.subplots_adjust(wspace=0.4)


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


def train(model, optimizer, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, max_iter=101, scheduler=None,
          device="cpu"):
    model.to(device)

    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()
    dynamics = []

    for step in range(max_iter):
        model.train()
        loss_cumulative = 0.
        loss_cumulative_mae = 0.
        for j, d in enumerate(dataloader_train):
            d.to(device)
            output = model(d)
            loss = loss_fn(output, d.phdos).cpu()
            loss_mae = loss_fn_mae(output, d.phdos).cpu()
            print(f"Iteration {step+1:4d}    batch {j+1:5d} / {len(dataloader_train):5d}   " +
                  f"batch loss = {loss.data}", end="\r", flush=True)
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

            dynamics.append({
                'step': step,
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

            yield {
                'dynamics': dynamics,
                'state': model.state_dict()
            }

            print(f"Iteration {step+1:4d}    batch {j+1:5d} / {len(dataloader_train):5d}   " +
                  f"train loss = {train_avg_loss[0]:8.3f}   " +
                  f"valid loss = {valid_avg_loss[0]:8.3f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")

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

    fig = plt.figure(figsize=(14,3))
    fontsize = 12
    for k in range(4*n):
        ax = fig.add_subplot(4,n,k+1)
        i = s[k]
        ax.plot(x, ds.iloc[i]['phdos'], color='black')
        ax.plot(x, ds.iloc[i]['phdos_pred'], color=palette[0])
        ax.set_xticks([]); ax.set_yticks([])
        
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.25)
    if title: fig.suptitle(title, ha='center', y=1.05)