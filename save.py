#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021

@author: aguscarpio99
"""

import numpy as np
import warnings

def save_data(data, label, directory="", slicer=slice(None)):
    with open(f"{directory}{label}.npy", "wb") as f:
        try:
            np.save(f, data[slicer])  #  Save file
        except TypeError:
            np.save(f, data)
            warnings.warn("You are saving non iterable data", Warning, stacklevel=3)

def save_return(label: str, directory="", slicer=slice(None)):
    """ Decorator to save return of functions as .npy with a label

    Parameters
    ----------
    label : string preceding .npy extension of saved file
    slice : slice constructor. Default is not slicing

    """
    def save_return_decorator(func):
        def func_and_save(*args, **kwargs):
            returned_value = func(*args, **kwargs)
            save_data(returned_value, label, directory, slicer)
            return returned_value
        return func_and_save
    return save_return_decorator
