import flask
from flask import Flask, request, redirect, url_for
import pickle
from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, Div, Plot, LinearAxis, Grid, Legend
from bokeh.models.widgets import TextInput, Button, Select
from bokeh.models.glyphs import VBar
from bokeh.io import curdoc
from bokeh.embed import components
from bokeh.models.callbacks import CustomJS
import numpy as np
import os
import argparse

app = Flask(__name__)

@app.route('/')
def create_main_page():
    dataset_names = os.listdir('./outputs')
    datasets = []
    for name in dataset_names:
        with open('./outputs/' + name, 'rb') as f:
            tmp = pickle.load(f)
        if type(tmp) is tuple:
            if len(tmp) == 5:
                _, data_size, metrics, _, class_sizes = tmp
                descr = []
            else:
                _, data_size, metrics, _, class_sizes, descr = tmp

            cls1 = ['']
            cls2 = ['train']
            cls3 = ['test']
            for label, count in class_sizes:
                cls1.append(label)
                cls2.append(int(count*0.9))
                cls3.append(int(count*0.1))
            classes=[cls1, cls2, cls3]
        else:
            data_size = 100
            descr = [('Most correct guesses: Layer 8, Head 8')]
            classes = []

        datasets.append((name, data_size, descr, classes))
    html = flask.render_template('main.html', datasets=datasets)
    return html

@app.route('/anaphor<suffix>')
def create_anaphora_page(suffix):
    with open('outputs/anaphor' + suffix, 'rb') as f:
        metrics = pickle.load(f)
    return flask.render_template('anaphora.html', metrics=metrics, set='anaphor' + suffix)

@app.route("/<output>")
def create_graph_page(output):
    sent_400 = False
    if output == 'sentiment-400':
        sent_400=True
    with open('outputs/' + output, 'rb') as f:
        tmp = pickle.load(f)
        if type(tmp) is tuple:
            if len(tmp) == 5:
                fold_count, data_size, metrics, rows, class_sizes = tmp
                descr = []
            else:
                fold_count, data_size, metrics, rows, class_sizes, descr = tmp
        else:
            metrics = tmp
            fold_count = -1
            data_size = -1


    metric_names = ['Logistic regression accuracy', 'Logistic regression min', 'Clustering v-measure', 'Average distance ratio', 'Logistic regression train score']
    xs = []
    ys = []
    plots = []
    sources = []
    glyphs = []
    renderers = []
    sorters = []

    # def sort(idx, metric):
    #     if metric == 'Layer':
    #         metric_idx = -2
    #     else:
    #         metric_idx = metric_names.index(metric)

    #     tmp_metrics = metrics.copy()
    #     reverse = True
    #     if metric == 'Average distance ratio' or metric == 'Layer':
    #         reverse = False
    #     tmp_metrics = sorted(tmp_metrics, key=lambda a: a[1 + metric_idx], reverse=reverse)

    #     xs_tmp = [metric[-1] + 1 for metric in tmp_metrics]
    #     for jdx, i in enumerate(xs_tmp):
    #         xs[idx][i-1] = jdx+1
    #     sources[idx] = ColumnDataSource(dict(x=xs[idx], top=ys[idx]))

    #     plots[idx].renderers.remove(renderers[idx])
    #     plots[idx].xaxis.ticker = [i + 1 for i in range(len(xs[idx]))]

    #     override = {}
    #     for i in range(len(metrics)):
    #         override[xs[idx][i]] = str(i+1)
    #     plots[idx].xaxis.major_label_overrides = override

    #     renderers[idx] = plots[idx].add_glyph(sources[idx], VBar(x='x', top='top', bottom=0, width = 0.2))

    callback_code = """ 
        let metric = sorter.value;
        let metric_idx = metrics[0].length - 2;
        if (metric != "Layer")
            metric_idx = metric_names.findIndex(x => x == metric);

        let tmp_metrics = metrics.slice(0);
        let reverse = true;
        if (metric == 'Average distance ratio' || metric == 'Layer')
            reverse = false;

        if (reverse)
            tmp_metrics.sort(function (a, b){return a[1 + metric_idx] < b[1 + metric_idx]});
        else
            tmp_metrics.sort(function (a, b){return a[1 + metric_idx] > b[1 + metric_idx]});

        let xs_tmp = new Array(tmp_metrics.length);
        for (i = 0; i < xs_tmp.length; i++)
            xs_tmp[i] = tmp_metrics[i][tmp_metrics[i].length - 1] + 1

        for (i = 0; i < xs_tmp.length; i++)
            xs[idx][xs_tmp[i] - 1] = i + 1

        source.attributes.data['x'] = xs[idx]

        let override = {}
        for (i = 0; i < metrics.length; i++)
            override[xs[idx][i]] = i + 1
        xaxis[0].attributes.major_label_overrides = override

        source.change.emit()
    """

    for idx, name in enumerate(metric_names):
        if name == 'Logistic regression accuracy':
            t_name = 'Logistic regression test accuracy'
        elif name == 'Logistic regression train score':
            t_name = 'Logistic regression train accuracy'
        else:
            t_name = name
        plots.append(figure(title=t_name, plot_width=600, plot_height=600, toolbar_location=None))
        xs.append([metric[-1] + 1 for metric in metrics])
        ys.append([metric[1 + idx] for metric in metrics])
        sources.append(ColumnDataSource(dict(x=xs[-1], top=ys[-1])))

        glyphs.append(VBar(x='x', top='top', bottom=0, width = 0.2))
        renderers.append(plots[-1].add_glyph(sources[-1], glyphs[-1]))
        plots[-1].xaxis.ticker = [i for i in xs[-1]]
        sorters.append(Select(title='Sort by', options = ['Layer'] + metric_names))
        sorters[-1].callback = CustomJS(args=dict(metrics=metrics, idx=idx, sorter=sorters[-1], source=sources[-1], xs=xs[::-1], ys=ys, metric_names=metric_names, xaxis=plots[-1].xaxis), code=callback_code)


    c_score_plot = figure(title='Train accuracy vs regularisation parameter', plot_width=800, plot_height=600, toolbar_location=None, x_axis_type='log')
    train_class_distr = ''
    test_class_distr = ''
    for label, count in class_sizes:
        train_class_distr += 'class ' + str(label) + ': ' + str(int(count * 0.9)) + ' '
        test_class_distr += 'class ' + str(label) + ': ' + str(int(count * 0.1)) + ' '
    c_score_plot.xaxis.axis_label = 'Regularisation parameter'
    c_score_plot.yaxis.axis_label = 'Accuracy'
    c_s_p_legend_data = []
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'black', 'olive', 'cyan', 'magenta', 'teal', 'grey', 'pink']
    for idx, metric in enumerate(metrics):
        tmp = list(map(list, zip(*metric[-4])))
        cs = tmp[0]
        cv_scores = tmp[1]
        if idx >= 12:
            c_s_p_legend_data.append(('Layer' + str(idx+1), [c_score_plot.square(cs, cv_scores, size=8, color=colors[idx])]))
        else:
            c_s_p_legend_data.append(('Layer' + str(idx+1), [c_score_plot.circle(cs, cv_scores, size=8, color=colors[idx])]))
        c_score_plot.line(cs, cv_scores, color=colors[idx])
    c_s_p_legend = Legend(items=c_s_p_legend_data)
    c_score_plot.add_layout(c_s_p_legend, 'right')


    c_score_plot_test = figure(title='Test accuracy vs regularisation parameter', plot_width=800, plot_height=600, toolbar_location=None, x_axis_type='log')
    c_score_plot_test.xaxis.axis_label = 'Regularisation parameter'
    c_score_plot_test.yaxis.axis_label = 'Accuracy'
    c_s_p_legend_data_test = []
    for idx, metric in enumerate(metrics):
        tmp = list(map(list, zip(*metric[-5])))
        cs = tmp[0]
        cv_scores = tmp[1]
        if idx >= 12:
            c_s_p_legend_data_test.append(('Layer' + str(idx+1), [c_score_plot_test.square(cs, cv_scores, size=8, color=colors[idx])]))
        else:
            c_s_p_legend_data_test.append(('Layer' + str(idx+1), [c_score_plot_test.circle(cs, cv_scores, size=8, color=colors[idx])]))
        c_score_plot_test.line(cs, cv_scores, color=colors[idx])
    c_s_p_legend_test = Legend(items=c_s_p_legend_data_test)
    c_score_plot_test.add_layout(c_s_p_legend_test, 'right')


    interval_plot = figure(title='Logistic regression CV accuracy', plot_width=600, plot_height=600, toolbar_location=None)
    lows = [metric[-2][0] for metric in metrics]
    highs = [metric[-2][1] for metric in metrics]
    lows1 = [metric[-3][0] for metric in metrics]
    highs1 = [metric[-3][1] for metric in metrics]
    points = []
    p_x_vals = []
    for idx, metric in enumerate(metrics):
        points = np.concatenate((points, metric[-6]))
        p_x_vals += [idx + 1 for _ in range(len(metric[-6]))]
    x_vals = [i + 1 for i in range(len(lows))]
    interval_plot.vbar(x_vals, 0.3, highs1, lows1, line_color='blue')
    interval_plot.segment(x_vals, lows, x_vals, highs, line_color='black', line_width=2)
    interval_plot.rect(x_vals, lows, 0.2, 0.0001, line_color='black')
    interval_plot.rect(x_vals, highs, 0.2, 0.0001, line_color='black')
    interval_plot.circle(p_x_vals, points, color='red', size=10)
    interval_plot.xaxis.ticker = x_vals



    avg_dist_div = Div(text="""Отношение среднего расстояния между элементами одного класса к среднему расстоянию между элементами разных. <br> 
                            Чем эта метрика больше, тем менее структурированы векторы, расположены более хаотично, классы хуже сгруппированы.""")
    cluster_div = Div(text="""V-мера кластеризации, равна (2*H*C)/(H+C), где H-однородность, C-полнота кластеризации.<br>
                            Чем эта метрика больше, тем лучше векторы кластеризуются.""")
    lr_min_div = Div(text="""Минимум точности логистической регрессии, полученный при кросс-валидации. 
                            Чем эта метрика больше, тем лучше работает логистическая регрессия.""")
    avg_lr_div = Div(text="""Точность логистической регрессии(с наилучшим найденным параметром регуляризации) на тестовой выборке.""")

    train_lr_div = Div(text="""Точность логистической регрессии на обучающей выборке""")

    divs = [avg_lr_div, lr_min_div, cluster_div, avg_dist_div, train_lr_div]

    sorter_div = [[sorters[i], divs[i]] for i in range(len(sorters))]

    interval_div = Div(text="""Интервалы, в которые попала точность логистической регрессии(с лучшим параметром) при кросс-валидации.<br>
                            Синий прямоугольник - 80% результатов, черная линия - 100% результатов.""")

    layout_list = [[sorter_div[i], plots[i]] for i in range(len(plots))]
    layout_list = []
    for i in range(len(plots)):
        if i == 0:
            layout_list.append([sorter_div[i], plots[i]])
        elif i == 1:
            layout_list.append([sorter_div[-1], plots[-1]])
        else:
            layout_list.append([sorter_div[i-1], plots[i-1]])

    layout_list[2].append(interval_plot)
    layout_list[2].append(interval_div)
    layout_list[1].append(c_score_plot)
    layout_list[0].append(c_score_plot_test)
    l = layout(layout_list, sizing_mode='fixed')

    script, div = components(l)
    html = flask.render_template('plots.html', layout_script=script, layout_div=div, set=output, size=int(0.9 * data_size), test_size=int(0.1*data_size), folds=fold_count, rows=rows, description=descr, train_class_distr=train_class_distr, test_class_distr=test_class_distr)
    return html


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-h', '--host', default='127.0.0.1')
parser.add_argument('-p', '--port', type=int, default=5000)
parser.add_argument('-d', '--debug', action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=args.debug)