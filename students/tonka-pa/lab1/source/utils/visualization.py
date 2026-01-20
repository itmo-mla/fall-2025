from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from IPython.display import display
from tabulate import tabulate


def display_nans(df):
    nans_per_col = [(col, df[col].isna().sum(), df[col].isna().sum() / df.shape[0] * 100) for col in df.columns]
    dtype = [('col_name', 'U20'), ('nans', int), ('nans_perc', float)]
    nans_per_col = np.array(nans_per_col, dtype=dtype)
    nans_per_col = nans_per_col[nans_per_col['nans'] > 0]
    nans_per_col = np.sort(nans_per_col, order='nans')

    if nans_per_col.shape[0] == 0:
        print('No nans in the dataset')
        return

    df_show = pd.DataFrame(nans_per_col[::-1])
    print(tabulate(df_show, headers='keys', tablefmt='github'))
    # display(df_show.style.background_gradient(cmap='Blues'))
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    y_pos = np.arange(len(nans_per_col))
    
    ax.barh(y_pos, nans_per_col['nans_perc'], alpha=0.8, edgecolor='black', linewidth=1) 
    ax.set_yticks(y_pos, labels=nans_per_col['col_name'])
    ax.set_xlabel('Nans, %', fontsize=14)
    ax.set_title('Nans rate for each column', fontsize=16)
    ax.set_xlim(0, min(np.max(df_show['nans_perc']) + 5.0, 100.0))
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(axis='x', linestyle='--', linewidth=0.5)
    
    plt.show()


def viz_margins(margins, eps=1.0, display_plot=False, save_path: str = ''):
    if save_path:
        save_path = Path(save_path + 'margins.png')

    sorted_idx = np.argsort(margins)
    sorted_margins = margins[sorted_idx]
    
    line_kwargs      = {'lw': 2}
    pos_fill_kwargs  = {'alpha': 0.25, 'color': 'tab:green'}
    neg_fill_kwargs  = {'alpha': 0.25, 'color': 'tab:red'}
    zero_fill_kwargs = {'alpha': 0.25, 'color': 'gold'}

    # masks
    if eps > 0.0:
        mask_zero = np.abs(sorted_margins) <= eps
        mask_pos  = sorted_margins >  eps
        mask_neg  = sorted_margins < -eps
    else:
        mask_zero = np.zeros_like(sorted_margins, dtype=bool)
        mask_pos  = sorted_margins > 0
        mask_neg  = sorted_margins < 0

    plt.figure(figsize=(12, 7))
    # line
    plot_idx = np.arange(sorted_margins.shape[0])
    plt.plot(plot_idx, sorted_margins, **line_kwargs)
    plt.axhline(0.0, color='black', lw=1, alpha=0.7)

    if np.any(mask_neg):
        plt.fill_between(plot_idx, sorted_margins, 0.0, where=mask_neg, interpolate=True, **neg_fill_kwargs)
    if np.any(mask_zero):
        plt.fill_between(plot_idx, sorted_margins, 0.0, where=mask_zero, interpolate=True, **zero_fill_kwargs)
    if np.any(mask_pos):
        plt.fill_between(plot_idx, sorted_margins, 0.0, where=mask_pos, interpolate=True, **pos_fill_kwargs)

    plt.xlabel("sample index (sorted)")
    plt.ylabel("margin")
    plt.title("Margin curve with signed areas")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
    if display_plot:
        plt.show()
    plt.close()
    return


if __name__ == '__main__':
    pass