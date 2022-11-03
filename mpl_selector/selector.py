import numpy as np
import pandas as pd

import collections
import re

from itertools import groupby as _groupby
from operator import itemgetter

from .hashables import hashable, unwrap


def groupby(artists, k, values_only=False):
    """
    by default return sorted list
    """
    l = [(getattr(a, "get_"+k, NA)(), a) for a in artists]
    _m = _groupby(sorted(l, key=itemgetter(0)), key=itemgetter(0))
    # m = dict((k, list(v1 for _, v1 in v)) for k, v in _m)
    m = list((k, list(v1 for _, v1 in v)) for k, v in _m)

    if values_only:
        return [v for k, v in m]
    else:
        return m


def NA(*kl, **kw):
    return None


def select(ax, color=None, fc=None, ec=None,
           klass=None, label=None, repr_re=None):
    p_repr = re.compile(repr_re) if repr_re is not None else None
    matched = []
    # for col in [ax.lines, ax.patches, ax.collections]:
    for col in [ax.patches, ax.collections, ax.lines, ax.artists]:
        for a in col:
            if ((repr_re is None or p_repr.search(repr(a))) and
                (color is None or np.all(getattr(a, "get_color", NA)() == fc)) and
                (fc is None or np.all(getattr(a, "get_fc", NA)() == fc)) and
                (ec is None or np.all(getattr(a, "get_ec", NA)() == ec)) and
                (klass is None or isinstance(a, klass)) and
                (label is None or a.get_label() == label)):
                matched.append(a)

    return matched


class Selector:
    def __init__(self, ax, prop_names=None):
        self.ax = ax

        if prop_names is None:
            prop_names = ["fc", "ec", "color", "lw", "alpha", "ls"]

        collections = [ax.patches,
                       ax.collections,
                       ax.lines,
                       ax.artists]
        self.artists = [a for coll in collections for a in coll]

        vv = []
        for a in self.artists:
            v = dict((k, getattr(a, "get_"+k, NA)())
                     for k in prop_names)
            v = dict((k1, hashable(v1)) for k1, v1 in v.items())
            v["class"] = type(a).__name__

            vv.append(v)

        # nan makes match difficult.
        self._tbl = pd.DataFrame(vv).fillna(0.)

    def guess_categorical(self, axis, ignore_columns=None,
                          import_legend=True):
        """
        axis: x or y
        """
        if axis == "x":
            ticklabels = self.ax.get_xticklabels()
        elif axis == "y":
            ticklabels = self.ax.get_yticklabels()
        else:
            raise ValueError(f"unknown axis: {axis}")

        categories = dict((t.get_text(), i) for i, t
                               in enumerate(ticklabels))
        ncat = len(categories)

        # exclude columns that has unique values.
        if ignore_columns is not None:
            df = self._tbl.drop(ignore_columns, axis=1)
        else:
            df = self._tbl
        cols = [col for col in df if df[col].nunique(dropna=False) > 1]

        kk = {}
        for i, row in df[cols].iterrows():
            k = tuple(row)
            kk.setdefault(k, []).append(i)

        kk_keys = [k for k, ii in kk.items() if len(ii) % ncat == 0]
        artist_indices = [kk[k] for k in kk_keys]

        du = pd.DataFrame(kk_keys, columns=cols)
        group_keys = []
        group_indices = []
        for klass, g in du.groupby("class"):
            _cols = [cn for cn in g if g[cn].nunique() > 1]
            for i, row in g[_cols].iterrows():
                group_keys.append((klass, dict(row)))
                group_indices.append(artist_indices[row.name])

        r = GroupedSelector(self.ax,
                            self.artists,
                            group_keys, group_indices,
                            categories=categories)
        r.import_legend()

        return r


class GroupedSelector:
    def __init__(self, ax, artists, group_keys, group_indices,
                 categories=None):
        self.ax = ax
        self.artists = artists
        self.group_keys = group_keys
        self.group_indices = group_indices
        self.categories = categories

    def import_legend(self, legend=None, *kl):
        if len(kl) == 0:
            kl = ["fc"]

        uu = {}
        if legend is None:
            hh, ll = self.ax.get_legend_handles_labels()
        else:
            # FIXME check if there is a better way
            ll = [t.get_text() for t in legend.get_texts()]
            hh = legend.legendHandles

        for h, l in zip(hh, ll):
            uu[l] = dict((k, getattr(h, "get_"+k, NA)()) for k in kl)

        assert len(uu) == len(ll)

        for l, u in uu.items():
            for o1 in self.group_keys:
                props = o1[1]
                if "fc" in props and np.all(unwrap(props["fc"]) == u["fc"]):
                    props["label"] = l


    def select_indices(self, klass_re, category=None, slice_mode=None,
                       **kw):
        p = re.compile(klass_re) if isinstance(klass_re, str) else klass_re

        ncat = len(self.categories)
        if category is not None:
            ik = self.categories[category]
            # sl = slice(ik, ik+1)
            if slice_mode is None:
                sl = None  # we will set it later
            elif slice_mode == "step":
                sl = slice(ik, None, ncat)
            else:
                raise ValueError(f"unknown slice_mode: {slice_mode}")
        else:
            sl = slice(None)

        r = []
        for i, (klass, prop) in enumerate(self.group_keys):
            if p is None or p.search(klass):
                # b = [prop.get(k, None) == v for k, v in kw.items()]
                if all(prop.get(k, None) == v for k, v in kw.items()):
                    r1 = self.group_indices[i]
                    if sl is None:
                        nk = len(r1) // ncat
                        sl1 = slice(ik * nk, (ik + 1)*nk)
                        r.extend(r1[sl1])
                    else:
                        r.extend(r1[sl])

        return r

    def select(self, klass_re, category=None, slice_mode=None,
               **kw):
        """

        slice_mode : None or "step"
            FIXME: need better name or approach
            This tells how to select artists if a group has more than ncat.
            If "step", it will do the slice of [i::ncat].
            If None, which is the default, it does slice of [i*s:(i+1)*s] where s = len(r) // ncat.
        """
        indices = self.select_indices(klass_re, category=category,
                                      slice_mode=slice_mode,
                                      **kw)

        return GroupSelectee(self, indices)


class GroupSelectee(collections.abc.Sequence):
    def __init__(self, group_selector, indices=None):
        self._group_selector = group_selector
        self.indices = frozenset() if indices is None else frozenset(indices)
        self._sorted_indices = sorted(self.indices)
        # self._artists = [] if artists is None else list(artists)
        self._artists_orig = self._group_selector.artists

    def __getitem__(self, index):
        return self._artists_orig[self._sorted_indices[index]]

    def __len__(self):
        return len(self._artists)

    def union(self, klass="", *kl, **kwargs):
        _new_indices = self.select_indices(klass, *kl, **kwargs)
        indices = self.indices.union(_new_indices)
        return type(self)(self._group_selector, indices)

    def intersection(self, klass="", *kl, **kwargs):
        _new_indices = self.select_indices(klass, *kl, **kwargs)
        indices = self.indices.intersection(_new_indices)
        return type(self)(self._group_selector, indices)

    def difference(self, klass="", *kl, **kwargs):
        _new_indices = self.select_indices(klass, *kl, **kwargs)
        indices = self.indices.difference(_new_indices)
        return type(self)(self._group_selector, indices)

    def select_indices(self, klass_re, category=None, slice_mode=None,
                       **kw):
        """

        slice_mode : None or "step"
            FIXME: need better name or approach
            This tells how to select artists if a group has more than ncat.
            If "step", it will do the slice of [i::ncat].
            If None, which is the default, it does slice of [i*s:(i+1)*s] where s = len(r) // ncat.
        """
        group_selector = self._group_selector
        indices = group_selector.select_indices(klass_re, category=category,
                                                slice_mode=slice_mode,
                                                **kw)

        return indices

    def set(self, prop, *kl, ignore_error=False, **kwargs):
        for a in self:
            try:
                getattr(a, "set_"+prop)(*kl, **kwargs)
            except AttributeError:
                if ignore_error:
                    continue
                else:
                    raise

        return self

