import copy
import json
import os
from parlai.core.opt import Opt
from parlai.core.teachers import FixedDialogTeacher, DialogTeacher, ParlAIDialogTeacher
from .build import build
from parlai.utils.io import PathManager

class PersonaInferenceTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        dt = opt['datatype'].split(':')[0]
        opt['datafile'] = os.path.join(opt['datapath'], 'pbst', 'contextual_alignment', 'persona_inference', dt + '.json')
        self.id = 'persona_inference'
        super().__init__(opt, shared)

    def setup_data(self, path):
        # note that path is the value provided by opt['datafile']
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            self.persona_inference = json.load(data_file)
        for dialog in self.persona_inference:
            text = dialog['text']
            if 'labels' in dialog.keys() and 'label_candidates' in dialog.keys():
                labels = dialog['labels']
                label_candidates = dialog['label_candidates']
                yield {"text": text, "labels": labels, 'label_candidates': label_candidates}, True
            elif 'labels' in dialog.keys():
                labels = dialog['labels']
                yield {'text': text, 'labels': labels}, True
            else:
                yield {"text": text}, True


class DefaultTeacher(PersonaInferenceTeacher):
    pass

class LexicalRetrievalTeacher(PersonaInferenceTeacher):
    pass