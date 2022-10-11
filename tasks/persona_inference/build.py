import parlai.core.build_data as build_data
import os
import json
import random
from tqdm import tqdm

def build(opt):
    dpath = os.path.join(opt['datapath'], 'pbst', 'contextual_alignment', 'persona_inference')
    version = 'v0.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        seed_utterance_convai2_path = opt['datapath'] + '/pbst/' + 'seed_utterance_pairs_convai2.json'
        l_convai2_path = opt['datapath'] + '/pbst/' + 'leading_convai2_kb.json'
        f_convai2_path = opt['datapath'] + '/pbst/' + 'following_convai2_kb.json'

        seed_utterance_wow_path = opt['datapath'] + '/pbst/' + 'seed_utterance_pairs_wizard_of_wikipedia.json'
        l_wow_path = opt['datapath'] + '/pbst/' + 'leading_wizard_of_wikipedia_kb.json'
        f_wow_path = opt['datapath'] + '/pbst/' + 'following_wizard_of_wikipedia_kb.json'

        seed_utterance_empathy_path = opt['datapath'] + '/pbst/' + 'seed_utterance_pairs_empatheticdialogues.json'
        l_empathy_path = opt['datapath'] + '/pbst/' + 'leading_empatheticdialogues_kb.json'
        f_empathy_path = opt['datapath'] + '/pbst/' + 'following_empatheticdialogues_kb.json'

        # Make file
        persona_inference_train_list = []
        persona_inference_valid_list = []
        persona_inference_test_list = []
        with open(seed_utterance_convai2_path) as json_file:
            seed_utterance_convai2 = json.load(json_file)

        # leader's persona
        with open(l_convai2_path) as json_file:
            l_convai2 = json.load(json_file)

        # follower's persona
        with open(f_convai2_path) as json_file:
            f_convai2 = json.load(json_file)

        for i in tqdm(range(len(seed_utterance_convai2))):
            dialog_dict_l = {}
            dialog_dict_l['text'] = seed_utterance_convai2[i][0]
            dialog_dict_l['labels'] = l_convai2[i]
            # dialog_dict_l['label_candidates'] = l_convai2 + f_convai2

            dialog_dict_f = {}
            dialog_dict_f['text'] = seed_utterance_convai2[i][1]
            dialog_dict_f['labels'] = f_convai2[i]
            # dialog_dict_f['label_candidates'] = f_convai2 + l_convai2

            if i <= int(len(seed_utterance_convai2) * 0.1):
                persona_inference_train_list.append(dialog_dict_l)
                persona_inference_train_list.append(dialog_dict_f)
            elif i > int(len(seed_utterance_convai2) * 0.8) and i < int(len(seed_utterance_convai2) * 0.9):
                persona_inference_valid_list.append(dialog_dict_l)
                persona_inference_valid_list.append(dialog_dict_f)
            elif i > int(len(seed_utterance_convai2) * 0.9):
                persona_inference_test_list.append(dialog_dict_l)
                persona_inference_test_list.append(dialog_dict_f)

        with open(dpath + '/all_persona.json', "w") as json_file:
            json.dump(l_convai2 + f_convai2, json_file)
        print("Saved file at", dpath + '/all_persona.json')

        f = open(dpath + '/fixed_candidates.txt', 'w')
        for persona in l_convai2:
            f.write(escape(persona) + '\n')
        for persona in f_convai2:
            f.write(escape(persona) + '\n')
        f.close()

        with open(dpath + '/train.json', "w") as json_file:
            json.dump(persona_inference_train_list, json_file)
        print("Saved file at", dpath + '/train.json')

        # Due to storge issue, use train file as valid/test file
        with open(dpath + '/valid.json', "w") as json_file:
            json.dump(persona_inference_valid_list, json_file)
        print("Saved file at", dpath + '/valid.json')

        with open(dpath + '/test.json', "w") as json_file:
            json.dump(persona_inference_test_list, json_file)
        print("Saved file at", dpath + '/test.json')     

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
        
def escape(msg):
    txt = str(msg)
    txt = txt.replace('\t', '\\t')
    txt = txt.replace('\n', '\\n')
    txt = txt.replace('\r', '\\r')
    return txt