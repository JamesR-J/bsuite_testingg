import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_sweep_variables(df):
    sweep_vars = []
    for k in df.columns.tolist():
        col = df[k]
        if isinstance(col[0], list):
            col = col.map(tuple)
        else:
            if col.nunique() > 1:
                sweep_vars.append(k)
    return sweep_vars

def _mean(data: pd.DataFrame, span: float, edge_tolerance: float = 0.):
    """Compute rolling mean of data via histogram, smooth endpoints.
    Args:
      data: pandas dataframe including columns ['x', 'y'] sorted by 'x'
      span: float in (0, 1) proportion of data to include.
      edge_tolerance: float of how much forgiveness to give to points that are
        close to the histogram boundary (in proportion of bin width).
    Returns:
      output_data: pandas dataframe with 'x', 'y' and 'stderr'
    """
    num_bins = np.ceil(1. / span).astype(np.int32)
    count, edges = np.histogram(data.x, bins=num_bins)
    # Include points that may be slightly on wrong side of histogram bin.
    tol = edge_tolerance * (edges[1] - edges[0])
    x_list = []
    y_list = []
    stderr_list = []
    for i, num_obs in enumerate(count):
        if num_obs > 0:
            sub_df = data.loc[(data.x > edges[i] - tol)
                              & (data.x < edges[i + 1] + tol)]
            x_list.append(sub_df.x.mean())
            y_list.append(sub_df.y.mean())
            stderr_list.append(sub_df.y.std() / np.sqrt(len(sub_df)))
    return pd.DataFrame(dict(x=x_list, y=y_list, stderr=stderr_list))

def smoothed_lineplot(x, y,
                      num_std=2,
                      span=0.1,
                      edge_tolerance=0.,
                      ax=None,
                      **kwargs):
    data = pd.DataFrame(dict(x=x, y=y))
    ax = ax if ax is not None else plt.gca()
    output_data = _mean(data, span, edge_tolerance=edge_tolerance)
    output_data['ymin'] = output_data.y - num_std * output_data.stderr
    output_data['ymax'] = output_data.y + num_std * output_data.stderr
    ax.plot(output_data.x, output_data.y, **kwargs, alpha=0.9)  # added an alpha to main line
    kwargs.pop('label', None)
    ax.fill_between(output_data.x, output_data.ymin, output_data.ymax, alpha=0.2, **kwargs)
    # ax.set_ylim(0, 1)
    # ax.set_ylim(0, 440)

