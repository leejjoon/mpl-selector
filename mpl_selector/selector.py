import numpy as np
import pandas as pd

from hashlib import sha1

from numpy import array, uint8
 

class NumpyHashable(object):
    r'''Hashable wrapper for ndarray objects.
        Instances of ndarray are not hashable, meaning they cannot be added to
        sets, nor used as keys in dictionaries. This is by design - ndarray
        objects are mutable, and therefore cannot reliably implement the
        __hash__() method.
        The hashable class allows a way around this limitation. It implements
        the required methods for hashable objects in terms of an encapsulated
        ndarray object. This can be either a copied instance (which is safer)
        or the original object (which requires the user to be careful enough
        not to modify it).
    '''
    def __init__(self, wrapped, tight=False):
        r'''Creates a new hashable object encapsulating an ndarray.
            wrapped
                The wrapped ndarray.
            tight
                Optional. If True, a copy of the input ndaray is created.
                Defaults to False.
        '''
        self.__tight = tight
        self.__wrapped = array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view(uint8)).hexdigest(), 16)

    def __eq__(self, other):
        return all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        r'''Returns the encapsulated ndarray.
            If the wrapper is "tight", a copy of the encapsulated ndarray is
            returned. Otherwise, the encapsulated ndarray itself is returned.
        '''
        if self.__tight:
            return array(self.__wrapped)

        return self.__wrapped

def hashable(v):
    if isinstance(v, np.ndarray):
        if len(v) == 1:
            return hashable(v[0])
        return NumpyHashable(v)
    elif isinstance(v, list):
        if len(v) == 1:
            return hashable(v[0])
        else:
            return NumpyHashable(v)
    else:
        return v


from itertools import groupby as _groupby
from operator import itemgetter

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

import re
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

    # for col in [ax.patches, ax.collections]:
    #     for a in col:
    #         if ((repr_re is None or p_repr.search(repr(a))) and
    #             (fc is None or np.all(a.get_fc() == fc)) and
    #             (ec is None or np.all(a.get_ec() == ec)) and
    #             (label is None or a.get_label() == label)):
    #             matched.append(a)

    return matched

# from itertools import groupby as _groupby
# from operator import itemgetter

# def groupby(artists, k):
#     l = [(getattr(a, "get_"+k, NA)(), a) for a in artists]
#     _m = _groupby(sorted(l, key=itemgetter(0)), key=itemgetter(0))
#     m = dict((k, list(v1 for _, v1 in v)) for k, v in _m)

#     return m

# def find_xind(ax, label):
#     for i, t in enumerate(ax.get_xticklabels()):
#         if t.get_text() == label:
#             return i, t._x

class Selector:
    def __init__(self, ax, prop_names=None):
        self.ax = ax
        # for col in [ax.lines, ax.patches, ax.collections]:
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
            # if isinstance(v1, np.ndarray))
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
            # print(tuple(row))
            k = tuple(row)
            kk.setdefault(k, []).append(i)

        # delete things whose cound is less than ncat. For seaborn violin plot,
        # this could be rect objects added for legend.

        if False:
            i_to_delete = []
            for k, ii in kk.items():
                if len(ii) < ncat:
                    i_to_delete.extend(ii)

            df = df.drop(axis=1, index=i_to_delete)

        # Now find props that may represents the categorical artists (length ==
        # ncat)

        kk_keys = [k for k, ii in kk.items() if len(ii) % ncat == 0]
        artist_indices = [kk[k] for k in kk_keys]
        # print(self.artists_indices)

        # :
        #     if len(ii) >= ncat:
        #         # print(k)
        #         print([k1.unwrap() if isinstance(k1, NumpyHashable) else k1
        #                for k1 in k])
        #         pass
        #     elif len(ii) > ncat:
        #         print([k1.unwrap() if isinstance(k1, NumpyHashable) else k1
        #                for k1 in k])
        #         # i_to_delete.extend(ii)


        du = pd.DataFrame(kk_keys, columns=cols)
        group_keys = []
        group_indices = []
        for klass, g in du.groupby("class"):
            _cols = [cn for cn in g if g[cn].nunique() > 1]
            # print( g[_cols])
            for i, row in g[_cols].iterrows():
                # print(row.name)
                group_keys.append((klass, dict(row)))
                group_indices.append(artist_indices[row.name])
        # for o in prop_keys:
        #     print(o)

        r = GroupedSelector(self.ax,
                            self.artists,
                            group_keys, group_indices,
                            categories=categories)
        r.import_legend()

        return r

        # self.prop_keys = prop_keys

        # self.du = du


    # def select(self, klass_re, x=None, **kw):
    #     p = re.compile(klass_re) if isinstance(klass_re, str) else klass_re

    #     r = []
    #     for i, (klass, prop) in enumerate(self.prop_keys):
    #         if p is None or p.search(klass):
    #             # b = [prop.get(k, None) == v for k, v in kw.items()]
    #             if all(prop.get(k, None) == v for k, v in kw.items()):
    #                 r.extend(self.artists_indices[i])

    #     return r

    # df[cols]

def unwrap(c):
    if hasattr(c, "unwrap"):
        return c.unwrap()
    else:
        return c

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
                    # del props["fc"]
                    # print("match", props)


    def select_indices(self, klass_re, category=None, slice_mode=None,
                       **kw):
        p = re.compile(klass_re) if isinstance(klass_re, str) else klass_re

        ncat = len(self.categories)
        if category is not None:
            ik = self.categories[category]
            # sl = slice(ik, ik+1)
            if slice_mode is None:
                sl = None # we fix it later
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
                        sl1 = slice(ik*nk, (ik+1)*nk)
                        r.extend(r1[sl1])
                        # if klass_re == "Line":
                        #     # print(self.group_indices[i], sl)
                        #     print(sl, i, r, r1[sl1])
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
        r = [self.artists[i] for i in indices]

        return GroupSelectee(self, r)


import collections

class GroupSelectee(collections.abc.Sequence):
    def __init__(self, group_selector, artists=None):
        self._group_selector = group_selector
        # self._indices = {} if indices is None else set(indices)
        self._artists = [] if artists is None else list(artists)

    def __getitem__(self, index):
        return self._artists[index]

    def __len__(self):
        return len(self._artists)

    def union(self, klass="", *kl, **kwargs):
        _new_artists = self.select(klass, *kl, **kwargs)
        artists = set(self._artists).union(_new_artists)
        return type(self)(self._group_selector, artists)

    def difference(self, klass="", *kl, **kwargs):
        _new_artists = self.select(klass, *kl, **kwargs)
        artists = set(self._artists).difference(_new_artists)
        return type(self)(self._group_selector, artists)

    def select(self, klass_re, category=None, slice_mode=None,
               **kw):
        """

        slice_mode : None or "step"
            FIXME: need better name or approach
            This tells how to select artists if a group has more than ncat.
            If "step", it will do the slice of [i::ncat].
            If None, which is the default, it does slice of [i*s:(i+1)*s] where s = len(r) // ncat.
        """
        group_selector = self._group_selector
        artists = group_selector.select(klass_re, category=category,
                                        slice_mode=slice_mode,
                                        **kw)

        return artists

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


if False:
    p = GroupSelectee(grouped_selector, [])
    p1 = p.union("Line")
    p2 = p1.difference(category="Sat")

