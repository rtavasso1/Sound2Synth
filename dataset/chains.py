from torchsynth.synth import AbstractSynth
from torchsynth.config import SynthConfig
from torchsynth.module import (
    ADSR,
    ControlRateUpsample,
    MonophonicKeyboard,
    SquareSawVCO,
    VCA,
    LFO,
    ControlRateVCA,
    AudioMixer,
)
try:
    from custom_modules import SynplantOsc
except:
    from dataset.custom_modules import SynplantOsc
from typing import Optional
import torch
from torchsynth.synth import Voice

class SimpleSynth(AbstractSynth):
    def __init__(self, synthconfig: Optional[SynthConfig] = None):
        super().__init__(synthconfig=synthconfig)
        self.add_synth_modules(
            [
                ("keyboard", MonophonicKeyboard, {"midi_f0": torch.tensor(60.0).repeat(synthconfig.batch_size), "duration": torch.tensor(5.0).repeat(synthconfig.batch_size)}), # 2 params 
                ("adsr", ADSR), # 5 params 
                ("upsample", ControlRateUpsample),
                ("vco", SquareSawVCO), # 4 param
                ("vca", VCA),
            ]
        )
        self.freeze_parameters([("keyboard", "midi_f0"), ("keyboard", "duration")])

    def output(self) -> torch.Tensor:
        midi_f0, note_on_duration = self.keyboard()

        envelope = self.adsr(note_on_duration)
        envelope = self.upsample(envelope)
        out = self.vco(midi_f0)
        out = self.vca(out, envelope)
        return out
    
class Synplant2(AbstractSynth):
    def __init__(self, synthconfig: Optional[SynthConfig] = None):
        super().__init__(synthconfig=synthconfig)
        self.add_synth_modules([
            ('keyboard', MonophonicKeyboard, {"midi_f0": torch.tensor(60.0).repeat(synthconfig.batch_size), "duration": torch.tensor(5.0).repeat(synthconfig.batch_size)}),
            ('vol_adsr', ADSR),
            ('pitch_adsr', ADSR),
            ('vol_lfo', LFO),
            ('pitch_lfo', LFO),
            ('control_vca', ControlRateVCA),
            ('vca', VCA),
            ('upsample', ControlRateUpsample),
            ('vco_a', SynplantOsc),
            ('vco_b', SynplantOsc),
            ("mixer",
             AudioMixer,
             {
                 "n_input": 2,
                 "names": ["vco_a", "vco_b"],
             },
             ),
        ])
        
        self.freeze_parameters([("keyboard", "midi_f0"), ("keyboard", "duration")])

    def output(self) -> torch.Tensor:
        midi_f0, note_on_duration = self.keyboard()

        vol_lfo = self.vol_lfo()
        pitch_lfo = self.pitch_lfo()

        vol = self.control_vca(vol_lfo, self.vol_adsr(note_on_duration))
        pitch = self.control_vca(pitch_lfo, self.pitch_adsr(note_on_duration))

        osc_a = self.vca(self.vco_a(midi_f0, self.upsample(pitch)), self.upsample(vol))
        osc_b = self.vca(self.vco_b(midi_f0, self.upsample(pitch)), self.upsample(vol))
        mixed = self.mixer(osc_a, osc_b)

        return mixed
