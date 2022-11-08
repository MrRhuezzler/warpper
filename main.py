import math
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorsys


def dtw(x, y, distance):
    N = len(x)
    M = len(y)
    cost_matrix = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_matrix[i, 0] = np.inf
    for j in range(1, M + 1):
        cost_matrix[0, j] = np.inf

    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_matrix[i, j],
                cost_matrix[i, j + 1],
                cost_matrix[i + 1, j]
            ]
            i_penalty = np.argmin(penalty)
            cost_matrix[i + 1, j +
                        1] = distance(i, j, N, M, x, y) + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        p = traceback_mat[i, j]
        if p == 0:
            i = i - 1
            j = j - 1
        elif p == 1:
            i = i - 1
        elif p == 2:
            j = j - 1
        path.append((i, j))

    return path[::-1], (cost_matrix[1:, 1:][N-1, M-1] / (N+M))


def manhattan_distance(i, j, N, M, x, y):
    return abs(x[i] - y[j])


def feature_based_distance(i, j, N, M, x, y):

    def cal_f_local(i, N, x):
        f_local = [x[i], x[i]]
        if i != 0:
            f_local[0] = f_local[0] - x[i - 1]
        if i != N - 1:
            f_local[1] = f_local[1] - x[i + 1]
        return f_local

    def cal_f_global(i, N, x):
        f_global = [x[i], x[i]]

        before_half = sum(x[:i]) / len(x[:i + 1])
        f_global[0] = f_global[0] - before_half

        after_half = sum(x[i + 1:]) / len(x[i:])
        f_global[1] = f_global[1] - after_half
        return f_global

    f_local_x = cal_f_local(i, N, x)
    f_local_y = cal_f_local(j, M, y)

    f_global_x = cal_f_global(i, N, x)
    f_global_y = cal_f_global(j, M, y)

    dist_local = abs(f_local_x[0] - f_local_y[0]) + \
        abs(f_local_x[1] - f_local_y[1])
    dist_global = abs(f_global_x[0] - f_global_y[0]) + \
        abs(f_global_x[1] - f_global_y[1])

    return dist_local + dist_global

def _get_colors(num_colors):

    def rgb_to_hex(rgb):
        return ('#{:02X}{:02X}{:02X}').format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(rgb_to_hex(colorsys.hls_to_rgb(hue, lightness, saturation)))
    return colors

def standardize_prices(x):
    result = []
    for i, j in zip(x, x[1:]):
        result.append(((i - j) * 100) / j)
    return np.array(result)


def min_max_normalize(x):
    result = []
    mx = max(x)
    mn = min(x)
    for i in x:
        result.append(((i - mn) / (mx - mn)) * 100)
    return np.array(result)


if __name__ == "__main__":

    pattern = min_max_normalize(np.array([2.5, -2.5, 2.5, -2.5, 2.5]))
    pattern_length = len(pattern)
    desired_length = pattern_length + 5

    df = pd.read_csv('data/APPL.csv')
    for i in ["Close/Last", "Open", "High", "Low"]:
        df[i] = df[i].apply(lambda x: float(x.replace('$', '')))

    close = df["Close/Last"]

    # Columns = Date,Close/Last,Volume,Open,High,Low
    # fig = go.Figure()
    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(go.Candlestick(x=df['Date'],
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close/Last'], name="TSLA"), row=1, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)


    fig.add_trace(go.Scatter(
        x=np.arange(pattern_length), y=pattern, name="Pattern", line={"color" : "black"}), row=2, col=1)

    colors = _get_colors(pattern_length)


    patterns = dict()
    for _ in range(pattern_length):
        patterns[_] = [[], []]

    for offset in range(0, (len(close) // desired_length) + 1):

        target = min_max_normalize(
            close[offset * desired_length: (offset + 1) * desired_length])
        
        if not len(target):
            continue

        path, cost = dtw(target, pattern, feature_based_distance)

        match_x = dict()
        match_y = dict()

        for _ in range(pattern_length):
            match_x[_] = []
            match_y[_] = []

        for pr, nt in zip(path, path[1:]):
            if pr[1] == nt[1]:
                match_x[pr[1]].append(
                    df['Date'][pr[0] + (desired_length * offset)])
                match_y[pr[1]].append(
                    df['Close/Last'][pr[0] + (desired_length * offset)])
            elif pr[1] != nt[1]:
                match_x[pr[1]].append(
                    df['Date'][pr[0] + (desired_length * offset)])
                match_y[pr[1]].append(
                    df['Close/Last'][pr[0] + (desired_length * offset)])
                match_x[pr[1]].append(
                    df['Date'][nt[0] + (desired_length * offset)])
                match_y[pr[1]].append(
                    df['Close/Last'][nt[0] + (desired_length * offset)])

        match_x[path[-1][1]].append(df['Date']
                                    [path[-1][0] + (desired_length * offset)])
        match_y[path[-1][1]].append(df['Close/Last']
                                    [path[-1][0] + (desired_length * offset)])

        for _ in range(pattern_length):
            fig.add_trace(go.Scatter(
                x=match_x[_], y=match_y[_], name=f"Pat-{_}", line={"color": colors[_]}), row=1, col=1)

    fig.show()
