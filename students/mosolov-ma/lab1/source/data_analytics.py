import pandas as pd

def describe_dataset(df):
    C = len(df.columns)
    L = len(df.index)
    CN = df.count()
    NP = ((L - CN) / L) * 100
    MN = df.min()
    Q1 = df.quantile(0.25)
    MA = df.mean()
    ME = df.median()
    Q3 = df.quantile(0.75)
    MX = df.max()
    ST = df.std()
    P = df.nunique()
    IQ = Q3 - Q1
    
    frame = pd.concat([CN, NP, MN, Q1, MA, ME, Q3, MX, ST, P, IQ], axis=1)
    frame = frame.T
    frame.index = ['Количество', 'Процент пропусков', 'Минимум', 'Первый квартиль','Среднее', 'Медиана', 
                   'Третий квартиль', 'Максимум','Стандартное отклонение', 'Мощность', 'Интерквартильный размах']
    return frame


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_workflow import load_and_prepare_data, scale_features, train_test_split_data
    from pca import compute_pca
    from graphics import heatmap_corr, plot_pca_scatter

    df = load_and_prepare_data()

    print(describe_dataset(df))

    heatmap_corr(df.corr())

    X = df.drop(columns='target')

    X = scale_features(X)

    y = df['target']

    pca, principal_components = compute_pca(X, n_components=2)
    
    fig, ax = plot_pca_scatter(principal_components, y)

    ax.legend()

    plt.show()