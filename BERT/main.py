import flask
from flask import Flask, request, redirect, url_for
import pickle
from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, Div, Plot, LinearAxis, Grid
from bokeh.models.widgets import TextInput, Button, Select
from bokeh.models.glyphs import VBar
from bokeh.io import curdoc
from bokeh.embed import components
from bokeh.models.callbacks import CustomJS
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def create_main_page():
    datasets = os.listdir('./outputs')
    html = flask.render_template('main.html', datasets=datasets)
    return html

@app.route('/anaphor<suffix>')
def create_anaphora_page(suffix):
    with open('outputs/anaphor' + suffix, 'rb') as f:
        metrics = pickle.load(f)
    return flask.render_template('anaphora.html', metrics=metrics, set='anaphor' + suffix)

@app.route("/<output>")
def create_graph_page(output):
    with open('outputs/' + output, 'rb') as f:
        metrics = pickle.load(f)

    metric_names = ['Logistic regression accuracy', 'Logistic regression min', 'Clustering v-measure', 'Average distance ratio']
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
        plots.append(figure(title=name, plot_width=600, plot_height=600, toolbar_location=None))
        xs.append([metric[-1] + 1 for metric in metrics])
        ys.append([metric[1 + idx] for metric in metrics])
        sources.append(ColumnDataSource(dict(x=xs[-1], top=ys[-1])))

        glyphs.append(VBar(x='x', top='top', bottom=0, width = 0.2))
        renderers.append(plots[-1].add_glyph(sources[-1], glyphs[-1]))
        plots[-1].xaxis.ticker = [i for i in xs[-1]]
        sorters.append(Select(title='Sort by', options = ['Layer'] + metric_names))
        sorters[-1].callback = CustomJS(args=dict(metrics=metrics, idx=idx, sorter=sorters[-1], source=sources[-1], xs=xs[::-1], ys=ys, metric_names=metric_names, xaxis=plots[-1].xaxis), code=callback_code)



    interval_plot = figure(title='Logistic regression cross validation scores', plot_width=600, plot_height=600, toolbar_location=None)
    lows = [metric[-2][0] for metric in metrics]
    highs = [metric[-2][1] for metric in metrics]
    lows1 = [metric[-3][0] for metric in metrics]
    highs1 = [metric[-3][1] for metric in metrics]
    x_vals = [i + 1 for i in range(len(lows))]
    interval_plot.vbar(x_vals, 0.3, highs1, lows1, line_color='blue')
    interval_plot.segment(x_vals, lows, x_vals, highs, line_color='black', line_width=2)
    interval_plot.rect(x_vals, lows, 0.2, 0.0001, line_color='black')
    interval_plot.rect(x_vals, highs, 0.2, 0.0001, line_color='black')
    interval_plot.xaxis.ticker = x_vals


    layout_list = [[sorters[i], plots[i]] for i in range(len(plots))]
    layout_list[0].append(interval_plot)
    l = layout(layout_list, sizing_mode='fixed')

    script, div = components(l)
    html = flask.render_template('plots.html', layout_script=script, layout_div=div, set=output)
    return html

if __name__ == "__main__":
    app.run()