import numpy as np
import torch
import plotly.graph_objects as go
from e3nn.io import SphericalTensor


def plot_orbitals(traces, traces_species, title_str, show_fig=False):
    traces = [go.Surface(**d) for d in traces]
    fig = go.Figure(data= (traces))
    fig.update_traces(showscale=False)
    if show_fig: fig.show()
    fig.update_layout(
        title=title_str,
        scene=dict(annotations=traces_species)
    )
    fig_html = fig.to_html()
    return fig_html


def plotly_surface(sph_tensor, signals, centers=None, res=10, radius=True, relu=False, species=None, normalization='integral'):
    r"""Create traces for plotly
    Examples
    --------
    >>> import plotly.graph_objects as go
    >>> x = SphericalTensor(4, +1, +1)
    >>> traces = x.plotly_surface(x.randn(-1))
    >>> traces = [go.Surface(**d) for d in traces]
    >>> fig = go.Figure(data=traces)
    """
    signals = signals.reshape(-1, sph_tensor.dim)

    if centers is None:
        centers = [None] * len(signals)
    else:
        centers = centers.reshape(-1, 3)

    traces = []
    traces_species = []
    if species is None:
        species = [f"annotation{i}" for i in range(sph_tensor.dim)]
    else:
        colors = np.linspace(0, 1, len(set(species)))
    for i, (signal, center) in enumerate(zip(signals, centers)):
        r, f = plot_r_surface(sph_tensor, signal, center, res, radius, relu, normalization)
        traces += [dict(
            x=r[:, :, 0].numpy(),
            y=r[:, :, 1].numpy(),
            z=r[:, :, 2].numpy(),
            surfacecolor=f.numpy(),
            text = species[i]
        )]
        traces_species += [dict(
            x=center[0].numpy(),
            y=center[1].numpy(),
            z=center[2].numpy(),
            ax=50,
            ay=-50,
            text=species[i],
            arrowhead=1,
            xanchor="center",
            yanchor="middle"
        )]
    return traces, traces_species


def plot_r_surface(sph_tensor, signal, center=None, res=10, radius=True, relu=False, normalization='integral'):
    r"""Create surface in order to make a plot
    """
    assert signal.dim() == 1

    r, f = sph_tensor.signal_on_grid(signal, res, normalization)
    f = f.relu() if relu else f

    # beta: [0, pi]
    r[0] = r.new_tensor([0.0, 1.0, 0.0])
    r[-1] = r.new_tensor([0.0, -1.0, 0.0])
    f[0] = f[0].mean()
    f[-1] = f[-1].mean()

    # alpha: [0, 2pi]
    r = torch.cat([r, r[:, :1]], dim=1)  # [beta, alpha, 3]
    f = torch.cat([f, f[:, :1]], dim=1)  # [beta, alpha]

    if radius:
        r *= f.abs().unsqueeze(-1)

    if center is not None:
        r += center

    return r, f


def build_sphericaltensors(features, irreps):
    sts = []
    st_feats = []

    for j, ir in enumerate(irreps):
        print(ir, ir[1].dim)
        ir_feat = features[:, irreps.slices()[j]]
        ir_feat_summed = ir_feat.view(-1, ir[0], ir[1].dim).sum(dim=-2)
        st = SphericalTensor(lmax=ir[1].l, p_val=ir[1].p, p_arg=ir[1].p ** (ir[1].l - 1))
        assert st[-1][1] in ir

        st_feat = st.randn(features.shape[0], -1) * 0.0
        st_feat[:, -ir[1].dim:] = ir_feat_summed

        sts.append(st)
        st_feats.append(st_feat)
    
    return sts, st_feats


def get_middle_feats(d, model, layer_idx=0, normalize=False):
    model.to('cpu')
    _ = model(d.cpu())
    try: features = model.mp.layers[layer_idx].first_out
    except:
        features = model.layers[layer_idx].first_out
        irreps = model.layers[layer_idx].second.irreps_in
    else: irreps = model.mp.layers[layer_idx].second.irreps_in

    sts, st_feats = build_sphericaltensors(features, irreps)
    if normalize:
        st_feats = [_ / (torch.norm(_, dim=1, keepdim=True) + 1e-12) for _ in st_feats]
    return sts, st_feats