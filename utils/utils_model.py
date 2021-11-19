from typing import Dict

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from e3nn.nn.models.v2106.points_convolution import Convolution
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm

# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'


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


class CustomedCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x


class MessagePassing(torch.nn.Module):
    r"""
    Parameters
    ----------
    irreps_node_sequence : list of `e3nn.o3.Irreps`
        representation of the input/hidden/output features
    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the nodes attributes
    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes
    layers : int
        number of gates (non linearities)
    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer
    """
    def __init__(
        self,
        irreps_node_sequence,
        irreps_node_attr,
        irreps_edge_attr,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors

        irreps_node_sequence = [o3.Irreps(irreps) for irreps in irreps_node_sequence]
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        self.irreps_node_sequence = [irreps_node_sequence[0]]
        irreps_node = irreps_node_sequence[0]

        for irreps_node_hidden in irreps_node_sequence[1:-1]:
            irreps_scalars = o3.Irreps([
                (mul, ir)
                for mul, ir in irreps_node_hidden
                if ir.l == 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
            ]).simplify()
            irreps_gated = o3.Irreps([
                (mul, ir)
                for mul, ir in irreps_node_hidden
                if ir.l > 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
            ])
            if irreps_gated.dim > 0:
                if tp_path_exists(irreps_node, self.irreps_edge_attr, "0e"):
                    ir = "0e"
                elif tp_path_exists(irreps_node, self.irreps_edge_attr, "0o"):
                    ir = "0o"
                else:
                    raise ValueError(f"irreps_node={irreps_node} times irreps_edge_attr={self.irreps_edge_attr} is unable to produce gates needed for irreps_gated={irreps_gated}")
            else:
                ir = None
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
            conv = Convolution(
                irreps_node,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                fc_neurons,
                num_neighbors
            )
            self.layers.append(CustomedCompose(conv, gate))
            irreps_node = gate.irreps_out
            self.irreps_node_sequence.append(irreps_node)

        irreps_node_output = irreps_node_sequence[-1]
        self.layers.append(
            Convolution(
                irreps_node,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                irreps_node_output,
                fc_neurons,
                num_neighbors
            )
        )
        self.irreps_node_sequence.append(irreps_node_output)

        self.irreps_node_input = self.irreps_node_sequence[0]
        self.irreps_node_output = self.irreps_node_sequence[-1]

    def forward(self, node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        for lay in self.layers:
            node_features = lay(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)

        return node_features


class SimpleNetwork(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        max_radius,
        num_neighbors,
        num_nodes,
        mul=50,
        layers=3,
        lmax=2,
        pool_nodes=True,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = 10
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes

        irreps_node_hidden = o3.Irreps([
            (mul, (l, p))
            for l in range(lmax + 1)
            for p in [-1, 1]
        ])

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_in] + layers * [irreps_node_hidden] + [irreps_out],
            irreps_node_attr="0e",
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
        )

        self.irreps_in = self.mp.irreps_node_input
        self.irreps_out = self.mp.irreps_node_output

    def preprocess(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        # Create graph
        edge_index = radius_graph(data['pos'], self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        # Edge attributes
        edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]

        return batch, data['x'], edge_src, edge_dst, edge_vec

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch, node_inputs, edge_src, edge_dst, edge_vec = self.preprocess(data)
        del data

        edge_attr = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization='component')

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis='smooth_finite',  # the smooth_finite basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        # Node attributes are not used here
        node_attr = node_inputs.new_ones(node_inputs.shape[0], 1)

        node_outputs = self.mp(node_inputs, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

        if self.pool_nodes:
            return scatter(node_outputs, batch, int(batch.max()) + 1).div(self.num_nodes**0.5)
        else:
            return node_outputs


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


    try: model.load_state_dict(torch.load(run_name + '.torch')['state'])
    except:
        results = {}
        history = []
        s0 = 0
    else:
        results = torch.load(run_name + '.torch')
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