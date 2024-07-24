from .base import *

# 87 regression parameters
# 66 classification parameters
REG_NCLASS = 64

dict_tupled_regression_parameters ={
    'SimpleSynth': [
        ('adsr', 'attack'), ('adsr', 'decay'), ('adsr', 'sustain'), ('adsr', 'release'), ('adsr', 'alpha'), ('keyboard', 'midi_f0'), ('keyboard', 'duration'), ('vco', 'tuning'), ('vco', 'mod_depth'), ('vco', 'initial_phase'), ('vco', 'shape')
    ],
    'Synplant2': [
        ('keyboard', 'midi_f0'), ('keyboard', 'duration'), ('vol_adsr', 'attack'), ('vol_adsr', 'decay'), ('vol_adsr', 'sustain'), ('vol_adsr', 'release'), ('vol_adsr', 'alpha'), ('pitch_adsr', 'attack'), ('pitch_adsr', 'decay'), ('pitch_adsr', 'sustain'), ('pitch_adsr', 'release'), ('pitch_adsr', 'alpha'), ('vol_lfo', 'frequency'), ('vol_lfo', 'mod_depth'), ('vol_lfo', 'initial_phase'), ('vol_lfo', 'sin'), ('vol_lfo', 'tri'), ('vol_lfo', 'saw'), ('vol_lfo', 'rsaw'), ('vol_lfo', 'sqr'), ('pitch_lfo', 'frequency'), ('pitch_lfo', 'mod_depth'), ('pitch_lfo', 'initial_phase'), ('pitch_lfo', 'sin'), ('pitch_lfo', 'tri'), ('pitch_lfo', 'saw'), ('pitch_lfo', 'rsaw'), ('pitch_lfo', 'sqr'), ('vco_a', 'tuning'), ('vco_a', 'mod_depth'), ('vco_a', 'initial_phase'), ('vco_a', 'a_form'), ('vco_a', 'a_mod'), ('vco_a', 'a_noise'), ('vco_a', 'a_freq'), ('vco_b', 'tuning'), ('vco_b', 'mod_depth'), ('vco_b', 'initial_phase'), ('vco_b', 'a_form'), ('vco_b', 'a_mod'), ('vco_b', 'a_noise'), ('vco_b', 'a_freq'), ('mixer', 'vco_a'), ('mixer', 'vco_b')    
    ],
    'Voice': [
        ('keyboard', 'midi_f0'), ('keyboard', 'duration'), ('adsr_1', 'attack'), ('adsr_1', 'decay'), ('adsr_1', 'sustain'), ('adsr_1', 'release'), ('adsr_1', 'alpha'), ('adsr_2', 'attack'), ('adsr_2', 'decay'), ('adsr_2', 'sustain'), ('adsr_2', 'release'), ('adsr_2', 'alpha'), ('lfo_1', 'frequency'), ('lfo_1', 'mod_depth'), ('lfo_1', 'initial_phase'), ('lfo_1', 'sin'), ('lfo_1', 'tri'), ('lfo_1', 'saw'), ('lfo_1', 'rsaw'), ('lfo_1', 'sqr'), ('lfo_2', 'frequency'), ('lfo_2', 'mod_depth'), ('lfo_2', 'initial_phase'), ('lfo_2', 'sin'), ('lfo_2', 'tri'), ('lfo_2', 'saw'), ('lfo_2', 'rsaw'), ('lfo_2', 'sqr'), ('lfo_1_amp_adsr', 'attack'), ('lfo_1_amp_adsr', 'decay'), ('lfo_1_amp_adsr', 'sustain'), ('lfo_1_amp_adsr', 'release'), ('lfo_1_amp_adsr', 'alpha'), ('lfo_2_amp_adsr', 'attack'), ('lfo_2_amp_adsr', 'decay'), ('lfo_2_amp_adsr', 'sustain'), ('lfo_2_amp_adsr', 'release'), ('lfo_2_amp_adsr', 'alpha'), ('lfo_1_rate_adsr', 'attack'), ('lfo_1_rate_adsr', 'decay'), ('lfo_1_rate_adsr', 'sustain'), ('lfo_1_rate_adsr', 'release'), ('lfo_1_rate_adsr', 'alpha'), ('lfo_2_rate_adsr', 'attack'), ('lfo_2_rate_adsr', 'decay'), ('lfo_2_rate_adsr', 'sustain'), ('lfo_2_rate_adsr', 'release'), ('lfo_2_rate_adsr', 'alpha'), ('mod_matrix', 'adsr_1->vco_1_pitch'), ('mod_matrix', 'adsr_1->vco_1_amp'), ('mod_matrix', 'adsr_1->vco_2_pitch'), ('mod_matrix', 'adsr_1->vco_2_amp'), ('mod_matrix', 'adsr_1->noise_amp'), ('mod_matrix', 'adsr_2->vco_1_pitch'), ('mod_matrix', 'adsr_2->vco_1_amp'), ('mod_matrix', 'adsr_2->vco_2_pitch'), ('mod_matrix', 'adsr_2->vco_2_amp'), ('mod_matrix', 'adsr_2->noise_amp'), ('mod_matrix', 'lfo_1->vco_1_pitch'), ('mod_matrix', 'lfo_1->vco_1_amp'), ('mod_matrix', 'lfo_1->vco_2_pitch'), ('mod_matrix', 'lfo_1->vco_2_amp'), ('mod_matrix', 'lfo_1->noise_amp'), ('mod_matrix', 'lfo_2->vco_1_pitch'), ('mod_matrix', 'lfo_2->vco_1_amp'), ('mod_matrix', 'lfo_2->vco_2_pitch'), ('mod_matrix', 'lfo_2->vco_2_amp'), ('mod_matrix', 'lfo_2->noise_amp'), ('vco_1', 'tuning'), ('vco_1', 'mod_depth'), ('vco_1', 'initial_phase'), ('vco_2', 'tuning'), ('vco_2', 'mod_depth'), ('vco_2', 'initial_phase'), ('vco_2', 'shape'), ('mixer', 'vco_1'), ('mixer', 'vco_2'), ('mixer', 'noise')
    ],
}
# tupled_regression_parameters = dict_tupled_regression_parameters
__all__ = ['REG_NCLASS']

regression_parameters = lambda synth: [' '.join(t) for t in dict_tupled_regression_parameters[synth]]
key = 'SimpleSynth'

classification_parameters = []

# def get_torchsynth_descriptors():
#     descriptors = MemberDict()
#     descriptors |= {
#         key: MemberDict({
#             "serialize": lambda x:(x['value']-x['param'].min_value)/(x['param'].max_value-x['param'].min_value),
#             "unserialize": lambda x:x['value']*(x['param'].max_value-x['param'].min_value)+x['param'].min_value,
#         }) for key in regression_parameters
#     }
#     descriptors |= {
#         key: MemberDict({
#             "serialize": lambda x:x['value'],
#             "unserialize": lambda x:x['value'],
#         }) for key, classes in classification_parameters
#     }

#     return descriptors

def get_torchsynth_descriptors():
    descriptors = MemberDict()
    descriptors |= {
        key: MemberDict({
            "serialize": lambda x: x['value'],
            "unserialize": lambda x: x['value'],
        }) for key in regression_parameters(key)
    }

    return descriptors

class TorchSynthInterface(BaseInterface):
    regression_nclasses = REG_NCLASS
    regression_parameters = regression_parameters(key)
    classification_parameters = classification_parameters
    ordered_descriptors = get_torchsynth_descriptors()
    criteria = ParameterSpaceLoss(
        regression_parameters=regression_parameters,
        classification_parameters=classification_parameters,
        regression_nclasses=REG_NCLASS,
    )
    
    def __init__(self, *args, **kwargs):
        super(TorchSynthInterface, self).__init__(*args, **kwargs); self.set_algorithm(None)

    def set_algorithm(self, algorithm=None):
        self.algorithm = algorithm