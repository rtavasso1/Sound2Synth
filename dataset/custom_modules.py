import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch import tensor

import torchsynth.util as util
from torchsynth.config import BASE_REPRODUCIBLE_BATCH_SIZE, SynthConfig
from torchsynth.parameter import ModuleParameter, ModuleParameterRange
from torchsynth.signal import Signal
from torchsynth import module as m

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

class SynplantOsc(m.VCO):
    default_parameter_ranges: List[ModuleParameterRange] = m.VCO.default_parameter_ranges + [
        ModuleParameterRange(
            0.0,
            1.0,
            name="a_form",),
        ModuleParameterRange(
            0.0,
            1.0,
            name="a_mod",),
        ModuleParameterRange(
            0.0,
            1.0,
            name="a_noise",),
        ModuleParameterRange(
            0.0,
            1.0,
            name="a_freq",),
    ]

    def oscillator(self, argument: Signal, midi_f0: T, module_tensors=None) -> Signal:
        """
        Generates output square/saw audio given a phase argument.

        Args:
            argument: The phase of the oscillator at each time sample.
            midi_f0: Fundamental frequency in midi.
        """
        partials = self.partials_constant(midi_f0, module_tensors).unsqueeze(1)
        square = torch.tanh(torch.pi * partials * torch.sin(argument) / 2)
        a_form = self.p("a_form", module_tensors=module_tensors)

        # slowww
        output = torch.zeros_like(argument)
        for i,form in enumerate(a_form):
            if 0 < form <= 0.57: # sin and saw
                x = map(form, 0, 0.57, 0, 1)
                cos = torch.cos(argument[i])
                output[i] = (1 - x)*(cos) + x*(0.5*square[i]*(1 + cos))
            elif 0.57 < form <= 0.81: # saw and square
                shape = 1 - map(form, 0.57, 0.81, 0, 1)
                output[i] = (1 - shape/2)*square[i]*(1 + shape*torch.cos(argument[i]))
            elif 0.81 < form <= 1: # square and pulse
                # might not be good
                duty = 1 - map(form, 0.81, 1, 0, 1)
                def saw(argument):
                    return 0.5 * torch.tanh(torch.pi * partials[i] * torch.sin(argument[i]) / 2)*(1 + torch.cos(argument[i]))
                output[i] = torch.tanh(2*torch.pi*(saw(argument[i]) - saw(argument[i] - torch.pi * duty)))

        return output

    def partials_constant(self, midi_f0, module_tensors=None):
        """
        Calculates a value to determine the number of overtones in the resulting
        square / saw wave, in order to keep aliasing at an acceptable level.
        Higher fundamental frequencies require fewer partials for a rich sound;
        lower-frequency sounds can safely have more partials without causing
        audible aliasing.

        Args:
            midi_f0: Fundamental frequency in midi.
        """
        max_pitch = midi_f0 + torch.maximum(
            self.p("mod_depth", module_tensors=module_tensors), tensor(0)
        )
        max_f0 = util.midi_to_hz(max_pitch)
        return 12000 / (max_f0 * torch.log10(max_f0))
