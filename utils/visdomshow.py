import torch
import visdom
import time
import numpy as np


class VisdomViz(object):

    def __init__(self, env_name='main', server='http://localhost',
                 port=8097):
        print('=====>')
        print('Initializing visdom env [{}]'.format(env_name))
        self.viz = visdom.Visdom(
            server=server, port=port, env=env_name, use_incoming_socket=False)
        self.wins = {}
        self.last_update_time = 0
        self.update_interval = 1.0
        self.update_cache = {}
        if self.viz.check_connection():
            print('server: {}, port: {} connect sucess'.format(server, port))
        else:
            print('connect failed')
            quit()
        print('<=====')

    def update(self, winname, iter, eval_dict, ylabel = None):
        for k, v in eval_dict.items():
            if ylabel is None:
                if 'acc' in k:
                    ylabel = 'acc'
                elif 'loss' in k:
                    ylabel = 'loss'
                elif 'err' in k:
                    ylabel = 'error'
                else:
                    ylabel = 'unkonw'
            else:
                ylabel = ylabel
            self.append_element(winname, iter, v, k, ylabel=ylabel)

    def append_element(self, window_name, x, y, line_name, xlabel='iters', ylabel='loss'):
        key = '{}/{}'.format(window_name, line_name)
        if key not in self.update_cache:
            self.update_cache[key] = ([x], [y], xlabel)
        else:
            x_prev, y_prev, _ = self.update_cache[key]
            self.update_cache[key] = (x_prev + [x], y_prev + [y], xlabel)

        if time.perf_counter() - self.last_update_time > self.update_interval:
            for k, v in self.update_cache.items():
                win_name, line_name = k.split('/')
                x, y, xlabel = v
                self._append_element(win_name, x, y, line_name, xlabel, ylabel=ylabel)

            self.last_update_time = time.perf_counter()
            self.update_cache = {}

    def _append_element(self, window_name, x, y, line_name,
                        xlabel='iters', ylabel='loss'):
        r"""
            Appends an element to a line

        Paramters
        ---------
        key: str
            Name of window
        x: float
            x-value
        y: float
            y-value
        line_name: str
            Name of line
        xlabel: str
        """
        if window_name in self.wins:
            self.viz.line(
                X=np.array(x),
                Y=np.array(y),
                win=self.wins[window_name],
                name=line_name,
                update='append')
        else:
            self.wins[window_name] = self.viz.line(
                X=np.array(x),
                Y=np.array(y),
                opts=dict(
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=window_name,
                    marginleft=30,
                    marginright=30,
                    marginbottom=30,
                    margintop=30,
                    legend=[line_name]))

    def pc2_show(self, pc1, pc2, markersize=2, title='init', legend=['fix', 'float']):
        label = torch.cat((torch.ones(pc1.size(0)), torch.ones(pc2.size(0)) * 2), dim=0)
        self.viz.scatter(torch.cat((pc1, pc2), dim=0), label,
                    opts={'markersize': markersize, 'title': title, 'legend': legend})

    def pc4_show(self, pc1, pc2, pc3, pc4, markersize=2, title='init', legend=['fix', 'float', 'fix_kp', 'float_kp']):
        label = torch.cat((torch.ones(pc1.size(0)), torch.ones(pc2.size(0)) * 2,
                           torch.ones(pc3.size(0)) * 3, torch.ones(pc4.size(0)) * 4), dim=0)
        self.viz.scatter(torch.cat((pc1, pc2, pc3, pc4), dim=0), label,
                    opts={'markersize': markersize, 'title': title, 'legend': legend})