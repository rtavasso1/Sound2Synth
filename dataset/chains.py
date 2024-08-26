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
    ModulationMixer,
    SineVCO,
    Noise
)
try:
    from custom_modules import SynplantOsc
except:
    from dataset.custom_modules import SynplantOsc
from typing import Optional
import torch
from torch import Tensor as T
# from torchsynth.synth import Voice

class SimpleSynth(AbstractSynth):
    def __init__(self, synthconfig: Optional[SynthConfig] = None, *args, **kwargs):
        super().__init__(synthconfig=synthconfig, *args, **kwargs)
        self.add_synth_modules(
            [
                # (
                #     "keyboard",
                #     MonophonicKeyboard,
                #     {
                #         "midi_f0": torch.tensor(60.0).repeat(
                #             synthconfig.batch_size
                #         ),
                #         "duration": torch.tensor(3.0).repeat(
                #             synthconfig.batch_size
                #         ),
                #         "use_parameters": True,
                #     },
                # ),  # 2 params
                ("adsr", ADSR, {"use_parameters": False}),  # 5 params
                ("upsample", ControlRateUpsample, {"use_parameters": False}),
                ("vco", SquareSawVCO, {"use_parameters": False}),  # 4 param
                ("vca", VCA, {"use_parameters": False}),
            ]
        )
        # self.freeze_parameters([("keyboard", "midi_f0"), ("keyboard", "duration")])
        self.midi_f0 = torch.tensor(60.0).repeat(synthconfig.batch_size)
        self.duration = torch.tensor(3.0).repeat(synthconfig.batch_size)

    def output(self, module_tensors=None) -> torch.Tensor:
        midi_f0, note_on_duration = self.midi_f0, self.duration # self.keyboard()
        midi_f0 = midi_f0.to(self.device)
        note_on_duration = note_on_duration.to(self.device)

        envelope = self.adsr(note_on_duration, module_tensors=module_tensors)
        envelope = self.upsample(envelope)
        out = self.vco(midi_f0, module_tensors=module_tensors)
        out = self.vca(out, envelope)
        return out


class Synplant2(AbstractSynth):
    def __init__(self, synthconfig: Optional[SynthConfig] = None, *args, **kwargs):
        super().__init__(synthconfig=synthconfig, *args, **kwargs)
        self.add_synth_modules(
            [
                # ('keyboard', MonophonicKeyboard, {"midi_f0": torch.tensor(60.0).repeat(synthconfig.batch_size), "duration": torch.tensor(5.0).repeat(synthconfig.batch_size)}),
                ("vol_adsr", ADSR, {"use_parameters": False}),
                ("pitch_adsr", ADSR, {"use_parameters": False}),
                ("vol_lfo", LFO, {"use_parameters": False}),
                ("pitch_lfo", LFO, {"use_parameters": False}),
                ("control_vca", ControlRateVCA),
                ("vca", VCA),
                ("upsample", ControlRateUpsample),
                ("vco_a", SynplantOsc, {"use_parameters": False}),
                ("vco_b", SynplantOsc, {"use_parameters": False}),
                (
                    "mixer",
                    AudioMixer,
                    {
                        "n_input": 2,
                        "names": ["vco_a", "vco_b"],
                        "use_parameters": False,
                    },
                ),
            ]
        )

        # self.freeze_parameters([("keyboard", "midi_f0"), ("keyboard", "duration")])
        self.midi_f0 = torch.tensor(60.0).repeat(synthconfig.batch_size)
        self.duration = torch.tensor(3.0).repeat(synthconfig.batch_size)

    def output(self, module_tensors=None) -> torch.Tensor:
        midi_f0, note_on_duration = self.midi_f0, self.duration  # self.keyboard()
        midi_f0 = midi_f0.to(self.device)
        note_on_duration = note_on_duration.to(self.device)

        vol_lfo = self.vol_lfo(module_tensors=module_tensors)
        pitch_lfo = self.pitch_lfo(module_tensors=module_tensors)

        vol = self.control_vca(
            vol_lfo,
            self.vol_adsr(note_on_duration, module_tensors=module_tensors),
        )
        pitch = self.control_vca(
            pitch_lfo,
            self.pitch_adsr(note_on_duration, module_tensors=module_tensors),
        )

        osc_a = self.vca(
            self.vco_a(midi_f0, self.upsample(pitch), module_tensors=module_tensors),
            self.upsample(vol),
        )
        osc_b = self.vca(
            self.vco_b(midi_f0, self.upsample(pitch), module_tensors=module_tensors),
            self.upsample(vol),
        )
        mixed = self.mixer(osc_a, osc_b, module_tensors=module_tensors)

        return mixed


class Voice(AbstractSynth):
    """
    The default configuration in torchsynth is the Voice, which is
    the architecture used in synth1B1. The Voice architecture
    comprises the following modules: a
    :class:`~torchsynth.module.MonophonicKeyboard`, two
    :class:`~torchsynth.module.LFO`, six :class:`~torchsynth.module.ADSR`
    envelopes (each :class:`~torchsynth.module.LFO` module includes
    two dedicated :class:`~torchsynth.module.ADSR`: one for rate
    modulation and another for amplitude modulation), one
    :class:`~torchsynth.module.SineVCO`, one
    :class:`~torchsynth.module.SquareSawVCO`, one
    :class:`~torchsynth.module.Noise` generator,
    :class:`~torchsynth.module.VCA`, a
    :class:`~torchsynth.module.ModulationMixer` and an
    :class:`~torchsynth.module.AudioMixer`. Modulation signals
    generated from control modules (:class:`~torchsynth.module.ADSR`
    and :class:`~torchsynth.module.LFO`) are upsampled to the audio
    sample rate before being passed to audio rate modules.

    You can find a diagram of Voice in `Synth Architectures documentation
    <../modular-design/modular-principles.html#synth-architectures>`_.
    """

    def __init__(
        self,
        synthconfig: Optional[SynthConfig] = None,
        nebula: Optional[str] = "default",
        *args,
        **kwargs,
    ):
        AbstractSynth.__init__(self, synthconfig=synthconfig, *args, **kwargs)

        # Register all modules as children
        self.add_synth_modules(
            [
                # ("keyboard", MonophonicKeyboard, {"midi_f0": torch.tensor(60.0).repeat(synthconfig.batch_size), "duration": torch.tensor(5.0).repeat(synthconfig.batch_size)}),
                ("keyboard", MonophonicKeyboard, {"use_parameters": False}),
                ("adsr_1", ADSR, {"use_parameters": False}),
                ("adsr_2", ADSR, {"use_parameters": False}),
                ("lfo_1", LFO, {"use_parameters": False}),
                ("lfo_2", LFO, {"use_parameters": False}),
                ("lfo_1_amp_adsr", ADSR, {"use_parameters": False}),
                ("lfo_2_amp_adsr", ADSR, {"use_parameters": False}),
                ("lfo_1_rate_adsr", ADSR, {"use_parameters": False}),
                ("lfo_2_rate_adsr", ADSR, {"use_parameters": False}),
                ("control_vca", ControlRateVCA, {"use_parameters": False}),
                ("control_upsample", ControlRateUpsample, {"use_parameters": False}),
                (
                    "mod_matrix",
                    ModulationMixer,
                    {
                        "n_input": 4,
                        "n_output": 5,
                        "input_names": ["adsr_1", "adsr_2", "lfo_1", "lfo_2"],
                        "output_names": [
                            "vco_1_pitch",
                            "vco_1_amp",
                            "vco_2_pitch",
                            "vco_2_amp",
                            "noise_amp",
                        ],
                        "use_parameters": False,
                    },
                ),
                ("vco_1", SineVCO, {"use_parameters": False}),
                ("vco_2", SquareSawVCO, {"use_parameters": False}),
                ("noise", Noise, {"seed": 13}),
                ("vca", VCA),
                (
                    "mixer",
                    AudioMixer,
                    {
                        "n_input": 3,
                        "curves": [1.0, 1.0, 0.025],
                        "names": ["vco_1", "vco_2", "noise"],
                        "use_parameters": False,
                    },
                ),
            ]
        )
        # self.freeze_parameters([("keyboard", "midi_f0"), ("keyboard", "duration")])
        # self.midi_f0 = torch.tensor(60.0).repeat(synthconfig.batch_size)
        # self.duration = torch.tensor(3.0).repeat(synthconfig.batch_size)

        # Load the nebula
        # self.load_hyperparameters(nebula)

    def output(self, module_tensors=None) -> T:
        # The convention for triggering a note event is that it has
        # the same note_on_duration for both ADSRs.

        midi_f0, note_on_duration = self.keyboard(module_tensors=module_tensors)
        # print(torch.min(midi_f0), torch.max(midi_f0), torch.min(note_on_duration), torch.max(note_on_duration))
        # midi_f0, note_on_duration = self.midi_f0, self.duration
        # midi_f0 = midi_f0.to(self.device)
        # note_on_duration = note_on_duration.to(self.device)

        # ADSRs for modulating LFOs
        lfo_1_rate = self.lfo_1_rate_adsr(note_on_duration, module_tensors=module_tensors)
        lfo_2_rate = self.lfo_2_rate_adsr(note_on_duration, module_tensors=module_tensors)
        lfo_1_amp = self.lfo_1_amp_adsr(note_on_duration, module_tensors=module_tensors)
        lfo_2_amp = self.lfo_2_amp_adsr(note_on_duration, module_tensors=module_tensors)

        # Compute LFOs with envelopes
        lfo_1 = self.control_vca(
            self.lfo_1(lfo_1_rate, module_tensors=module_tensors),
            lfo_1_amp,
        )
        lfo_2 = self.control_vca(
            self.lfo_2(lfo_2_rate, module_tensors=module_tensors),
            lfo_2_amp,
        )

        # ADSRs for Oscillators and noise
        adsr_1 = self.adsr_1(note_on_duration, module_tensors=module_tensors)
        adsr_2 = self.adsr_2(note_on_duration, module_tensors=module_tensors)

        # Mix all modulation signals
        (vco_1_pitch, vco_1_amp, vco_2_pitch, vco_2_amp, noise_amp) = self.mod_matrix(
            adsr_1, adsr_2, lfo_1, lfo_2, module_tensors=module_tensors
        )

        # Create signal and with modulations and mix together
        vco_1_out = self.vca(
            self.vco_1(
                midi_f0,
                self.control_upsample(vco_1_pitch),
                module_tensors=module_tensors,
            ),
            self.control_upsample(vco_1_amp),
        )
        vco_2_out = self.vca(
            self.vco_2(
                midi_f0,
                self.control_upsample(vco_2_pitch),
                module_tensors=module_tensors,
            ),
            self.control_upsample(vco_2_amp),
        )
        noise_out = self.vca(self.noise(), self.control_upsample(noise_amp))

        return self.mixer(
            vco_1_out, vco_2_out, noise_out, module_tensors=module_tensors
        )
